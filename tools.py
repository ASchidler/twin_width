import os
import sys

import networkx as nx
from pysat.card import CardEnc
from pysat.formula import CNF


def amo_commander(vars, vpool, m=2):
    formula = CNF()
    # Separate into list
    cnt = 0
    groups = []
    while cnt < len(vars):
        cg = []
        for i in range(0, min(m, len(vars) - cnt)):
            cg.append(vars[cnt + i])
        groups.append(cg)
        cnt += m

    cmds = []
    # Encode commanders
    for cg in groups:
        if len(cg) > 1:
            ncmd = vpool.id(f"amo_{vpool.top}")
            cmds.append(ncmd)
            cg.append(-ncmd)
            formula.extend(CardEnc.atmost(cg, bound=1, vpool=vpool))
            formula.extend(CardEnc.atleast(cg, bound=1, vpool=vpool))
        else:
            cmds.append(cg[0])

    # Recursive call?
    if len(cmds) < 2 * m:
        formula.extend(CardEnc.atmost(cmds, bound=1, vpool=vpool))
    else:
        formula.extend(amo_commander(cmds, vpool, m=m))

    return formula


def dot_export(g, u, v):
    def cln(name):
        return f"{name}".replace("(", "").replace(")", "").replace(",", "").replace(" ", "_")

    output1 = "strict graph dt {" + os.linesep

    for n in g.nodes:
        cl = 'green' if n == u or n == v else 'black'
        posstr = ""
        if isinstance(n, tuple) and len(n) == 2:
            posstr = f', pos="{n[0]},{n[1]}!"'
        output1 += f"n{cln(n)} [" \
                   f"shape=box, fontsize=11,width=0.3,height=0.2,fixedsize=true,style=filled,fontcolor=white," \
                   f"color={cl}, fillcolor={cl}{posstr}];{os.linesep}"

    for x, y in g.edges:
        cl = 'red' if 'red' in g[x][y] and g[x][y]['red'] else 'black'
        output1 += f"n{cln(x)} -- n{cln(y)} [color={cl}];{os.linesep}"

    # # Draw the linegraph
    # output2 = "strict graph dt {" + os.linesep
    # u, v = min(u, v), max(u, v)
    # for x, y in g.edges:
    #     x, y = min(x, y), max(x, y)
    #     color = 'green' if x == u and v == y else 'white'
    #     fillcolor = 'red' if 'red' in g[x][y] and g[x][y]['red'] else 'black'
    #     output2 += f"n{cln(x)}_{cln(y)} [" \
    #     f"shape=box, fontsize=11,style=filled,fontcolor={color}," \
    #     f"color={color}, fillcolor={fillcolor}];{os.linesep}"
    #
    # for n in g.nodes:
    #     for n1 in g[n]:
    #         x1, x2 = min(n1, n), max(n1, n)
    #         for n2 in g[n]:
    #             if n2 > n1:
    #                 cl = 'green' if n1 == u and n2 == v else 'black'
    #                 x3, x4 = min(n2, n), max(n2, n)
    #                 output2 += f"n{cln(x1)}_{cln(x2)} -- n{cln(x3)}_{cln(x4)} [color={cl}];{os.linesep}"

    return output1 + "}"


def find_modules(g):
    ordering = [x for x in g.nodes]
    m = None
    for _ in range(0, len(g.nodes)):
        m = _find_modules(g, ordering)
        if len(m) < len(g.nodes):
            # for x in m:
            #     if len(x) > 1:
            #         xs = set(x)
            #         nbs = [set(g.neighbors(cx)) - xs for cx in x]
            #
            #         if not all(cx == nbs[0] for cx in nbs):
            #             print("ERROR")
            return m
        ordering.insert(0, ordering.pop())

    return m


def _find_modules(g, p):
    class ListEntry:
        def __init__(self, elements, can_pivot=True, is_center=False):
            self.next_entry = None
            self.elements = elements
            self.can_pivot = can_pivot
            self.is_center = is_center

        def split(self, s, included_first):
            s1 = self.elements & s
            s2 = self.elements - s

            if len(s1) == 0 or len(s2) == 0:
                return False

            if included_first:
                new_entry = ListEntry(s2)
                self.elements = s1
            else:
                new_entry = ListEntry(s1)
                self.elements = s2

            self.can_pivot = True
            new_entry.next_entry = self.next_entry
            self.next_entry = new_entry
            return True

    c = p[0]
    lst = ListEntry({x for x in p if x not in g._adj[c] and x != c})
    lst.next_entry = ListEntry({c}, can_pivot=False, is_center=True)
    lst.next_entry.next_entry = ListEntry({x for x in p if x in g._adj[c]})

    changed = True
    while changed:
        changed = False
        centry = lst

        while centry is not None:
            if centry.can_pivot:

                for n in centry.elements:
                    oentry = lst
                    passed_other = False
                    passed_center = False
                    nb = set(g.neighbors(n))
                    while oentry is not None:
                        if oentry is centry:
                            passed_other = True
                        elif oentry.is_center:
                            passed_center = True
                        elif len(nb & oentry.elements) > 0:
                            if (passed_other and not passed_center) or (passed_center and not passed_other):
                                changed = oentry.split(nb, True) or changed
                            else:
                                changed = oentry.split(nb, False) or changed

                        oentry = oentry.next_entry

                centry.can_pivot = False
            centry = centry.next_entry

    modules = []
    c_lst = lst
    while c_lst is not None:
        modules.append(list(c_lst.elements))
        c_lst = c_lst.next_entry
    return modules


def check_result(g, od, mg):
    for u, v in g.edges:
        g[u][v]['red'] = False

    c_max = 0
    cnt = 0
    for n in od[:-1]:
        if n not in mg:
            return max(c_max, len(g.nodes))

        t = mg[n]
        tn = set(g.neighbors(t))
        tn.discard(n)
        nn = set(g.neighbors(n))

        for v in nn:
            if v != t:
                # Red remains, should edge exist
                if v in tn and g[n][v]['red']:
                    g[t][v]['red'] = True
                # Add non-existing edges
                if v not in tn:
                    g.add_edge(t, v, red=True)
        for v in tn:
            if v not in nn:
                g[t][v]['red'] = True
        g.remove_node(n)

        # Count reds...
        for u in g.nodes:
            cc = 0
            for v in g.neighbors(u):
                if g[u][v]['red']:
                    cc += 1
            c_max = max(c_max, cc)
        cnt += 1
    # print(f"Done {c_max}/{d}")
    return c_max


def prime_paley(p):
    G = nx.Graph()

    square_set = {(x ** 2) % p for x in range(1, p)}

    for x in range(p):
        for y in range(x+1, p):
            if y - x in square_set:
                G.add_edge(x, y)

    return G


def prime_square_paley(p):
    """Generates the paley graph for p^2"""
    # See: https://en.wikipedia.org/wiki/Finite_field
    G = nx.Graph()

    # Find non-square, i.e. quadratic non-residue
    p_squares = {(x*x) % p for x in range(1, p)}
    n_squares = {x for x in range(1, p) if x not in p_squares}
    m = next(iter(n_squares))

    # Compute elements of p^2
    elements = [(x, y) for x in range(p) for y in range(p)]
    squares = {((x*x + m*y*y) % p, (2 * x * y) % p) for x, y in elements}

    for x1 in range(p):
        for y1 in range(p):
            for x2 in range(p):
                for y2 in range(p):
                    if x1 != x2 or y1 != y2:
                        result = ((x2 - x1) % p, (y2 - y1) % p)
                        if result in squares:
                            G.add_edge((x1, y1), (x2, y2))

    return G


def rook(n):
    g = nx.Graph()

    for x1 in range(1, n+1):
        for y1 in range(1, n+1):
            for x2 in range(x1+1, n+1):
                g.add_edge((x1, y1), (x2, y1))
            for y2 in range(y1+1, n+1):
                g.add_edge((x1, y1), (x1, y2))

    return g


def line(n):
    g = nx.Graph()

    for i in range(1, n+1):
        for j in range(i+1, n+1):
            for k in range(1, n+1):
                if k != j:
                    if k > i:
                        g.add_edge((i, j), (i, k))
                    elif k < i:
                        g.add_edge((i, j), (k, i))
                if k != i:
                    if j > k:
                        g.add_edge((i, j), (k, j))
                    elif j < k:
                        g.add_edge((i, j), (j, k))

    return g


def solve_grid2(d1, d2, ub):
    # Create adjacency matrix
    adj = [[0 for _ in range(0, d1 * d2)] for _ in range(0, d1 * d2)]

    for i in range(0, d1):
        for j in range(0, d2):
            co1 = i * d1 + j

            for xoff, yoff in [(-1, 0), (1, 0), (0, -1), (-1, 0)]:
                if 0 < i + xoff < d1 and 0 < j + yoff < d2:
                    co2 = (i+xoff) * d1 + (j+yoff)
                    # This should set both directions
                    adj[co1][co2] = 1

    contracted = [False for _ in range(0, d1 * d2)]
    counts = [0 for _ in range(0, d1*d2)]
    od = []
    mg = {}

    if solve_grid2_(d1, d2, ub, adj, contracted, od, mg, counts):
        # Complete order
        missing = []
        for c_node in range(0, d1*d2):
            if not contracted[c_node]:
                missing.append((c_node // d1, c_node % d1))

        for i in range(0, len(missing)-1):
            od.append(missing[i])
            mg[missing[i]] = missing[i+1]

        g = nx.Graph()
        for i in range(0, d1):
            for j in range(0, d2):
                for xoff, yoff in [(-1, 0), (1, 0), (0, -1), (-1, 0)]:
                    if 0 <= i + xoff < d1 and 0 <= j + yoff < d2:
                        g.add_edge((i, j), (i+xoff, j+yoff))

        if check_result(g, od, mg) > ub:
            raise RuntimeError("Invalid solution found")
        for c_n in od:
            print(f"{c_n}: {mg[c_n]}")
        return True

    return False


def solve_grid2_(d1, d2, ub, adj, contracted, od, mg, counts):
    # Check if graph is fully contracted
    if d1 * d2 - len(od) <= ub:
        return True

    found_any = False
    for i in range(0, d1):
        if len(od) == 1 and i > 0:
            continue
        for j in range(0, d2):
            if len(od) == 1 and j > 0:
                continue
            cc = i * d1 + j
            if not contracted[cc]:
                # Search neighborhood. The way this works, the other index is always bigger
                for xoff, yoff in [(1, -1), (1, 0), (2, 0), (1, 1), (0, 1), (0, 2)]:
                    cc2 = (i + xoff) * d1 + (j + yoff)
                    if 0 <= i + xoff < d1 and 0 <= j + yoff < d2 and not contracted[cc2]:
                        found_any = True
                        reds = 0
                        new_red = []

                        for k in range(0, d1 * d2):
                            if contracted[k]:
                                continue
                            if adj[cc2][k] == 2:
                                reds += 1
                            elif adj[cc][k] == 2 and adj[cc2][k] <= 1:
                                reds += 1
                                new_red.append((cc2, k, adj[cc2][k]))
                                new_red.append((k, cc2, adj[cc2][k]))
                                if counts[k] == ub:
                                    reds = sys.maxsize
                                    break
                            elif adj[cc][k] == 1 and adj[cc2][k] == 0:
                                reds += 1
                                new_red.append((cc2, k, 0))
                                new_red.append((k, cc2, 0))
                                if counts[k] == ub:
                                    reds = sys.maxsize
                                    break
                            elif adj[cc][k] == 0 and adj[cc2][k] == 1:
                                reds += 1
                                new_red.append((cc2, k, 1))
                                new_red.append((k, cc2, 1))
                                if counts[k] == ub:
                                    reds = sys.maxsize
                                    break
                            if reds > ub:
                                break

                        if reds <= ub:
                            contracted[cc] = True
                            for ce1, ce2, _ in new_red:
                                adj[ce1][ce2] = 2
                                counts[ce1] += 1
                            od.append((i, j))
                            mg[(i, j)] = (i+xoff, j+yoff)
                            if solve_grid2_(d1, d2, ub, adj, contracted, od, mg, counts):
                                return True
                            contracted[cc] = False
                            od.pop()
                            mg.pop((i, j))
                            for ce1, ce2, p in new_red:
                                adj[ce1][ce2] = p
                                counts[ce1] -= 1
                        else:
                            print(f"Conflict {len(od)}")
    if not found_any:
        return False


def solve_quick(g, ub=sys.maxsize):
    nodes = {x: i for i, x in enumerate(g.nodes)}

    adj = [[0 for _ in range(0, len(nodes))] for _ in range(0, len(nodes))]
    for n in g.nodes:
        nid = nodes[n]
        for n2 in g.neighbors(n):
            adj[nid][nodes[n2]] = 1

    # Find degree two neighborhood
    nbs = [set() for _ in range(0, len(nodes))]

    for n in g.nodes:
        q = [(n, 0)]
        lst = nbs[nodes[n]]
        while q:
            c_n, d = q.pop()

            if d < 2:
                for n2 in g.neighbors(c_n):
                    lst.add(nodes[n2])
                    q.append((n2, d+1))

        lst.remove(nodes[n])

    for i, lst in enumerate(nbs):
        nbs[i] = [x for x in lst if x > i]

    contracted = [False for _ in range(0, len(nodes))]
    counts = [0 for _ in range(0, len(nodes))]
    od = []
    mg = {}
    solve_quick_(adj, nbs, contracted, od, mg, ub, counts)


def solve_quick_(adj, nbs, contracted, od, mg, ub, counts):
    if len(adj) - len(od) == 1:
        return max(counts), list(od), {x: y for x, y in mg.items()}

    best = None

    for i in range(0, len(adj)):
        if not contracted[i]:
            for j in nbs[i]:
                if not contracted[j]:
                    reds = 0
                    new_red = []

                    for k in range(0, len(adj)):
                        if contracted[k]:
                            continue
                        if adj[j][k] == 2:
                            reds += 1
                        elif adj[i][k] == 2 and adj[j][k] < 2:
                            reds += 1
                            new_red.append((j, k, adj[j][k]))
                            new_red.append((k, j, adj[j][k]))
                            if counts[k] == ub:
                                ub = sys.maxsize
                                break
                        elif adj[i][k] == 1 and adj[j][k] == 0:
                            reds += 1
                            new_red.append((j, k, 0))
                            new_red.append((k, j, 0))
                            if counts[k] == ub:
                                ub = sys.maxsize
                                break
                        elif adj[i][k] == 0 and adj[j][k] == 1:
                            reds += 1
                            new_red.append((j, k, 1))
                            new_red.append((k, j, 1))
                            if counts[k] == ub:
                                ub = sys.maxsize
                                break

                        if reds > ub:
                            break

                    if reds <= ub:
                        contracted[i] = True

                        for ce1, ce2, _ in new_red:
                            adj[ce1][ce2] = 2
                            counts[ce1] += 1
                        od.append(i)
                        mg[i] = j

                        result = solve_quick_(adj, nbs, contracted, od, mg, ub, counts)
                        if result is not None:
                            best = result
                            ub = result[0]

                        contracted[i] = False
                        od.pop()
                        mg.pop(i)
                        for ce1, ce2, p in new_red:
                            adj[ce1][ce2] = p
                            counts[ce1] -= 1
                    else:
                        print(f"Conflict {len(od)}")

    return best
