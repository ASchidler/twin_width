from pysat.formula import CNF
from pysat.card import CardEnc
import os
import networkx as nx
import random

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

    # Draw the linegraph
    output2 = "strict graph dt {" + os.linesep
    u, v = min(u, v), max(u, v)
    for x, y in g.edges:
        x, y = min(x, y), max(x, y)
        color = 'green' if x == u and v == y else 'white'
        fillcolor = 'red' if 'red' in g[x][y] and g[x][y]['red'] else 'black'
        output2 += f"n{cln(x)}_{cln(y)} [" \
        f"shape=box, fontsize=11,style=filled,fontcolor={color}," \
        f"color={color}, fillcolor={fillcolor}];{os.linesep}"

    for n in g.nodes:
        for n1 in g[n]:
            x1, x2 = min(n1, n), max(n1, n)
            for n2 in g[n]:
                if n2 > n1:
                    cl = 'green' if n1 == u and n2 == v else 'black'
                    x3, x4 = min(n2, n), max(n2, n)
                    output2 += f"n{cln(x1)}_{cln(x2)} -- n{cln(x3)}_{cln(x4)} [color={cl}];{os.linesep}"

    return output1 + "}", output2 + "}"


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


def solve_grid(g, ub):
    g2 = g.copy()
    for u, v in g2.edges:
        g2[u][v]["red"] = False

    od, mg = _solve_grid(g2, ub)

    if not check_result(g, od, mg):
        print("Error")

    print("Found")


def _solve_grid(g, ub):
    # find merges:
    contractions = []
    if len(g.nodes) <= ub:
        return list(g.nodes), {}

    for n in g.nodes:
        nbs = set()
        for n2 in g._adj[n]:
            nbs.add(n2)
            nbs.update(g._adj[n2])

        for cnb in nbs:
            if cnb > n:
                shared = g._adj[n].keys() & g._adj[cnb].keys()
                excl = g._adj[n].keys() ^ g._adj[cnb].keys()
                excl.discard(n)
                excl.discard(cnb)
                reds = []

                for cn in shared:
                    if g._adj[n][cn]["red"] or g._adj[cnb][cn]["red"]:
                       reds.append(cn)
                reds.extend(excl)

                if len(reds) <= ub:
                    contractions.append(((n, cnb),
                                         [x for x in reds if x not in g._adj[cnb]],
                                         [x for x in reds if x in g._adj[cnb] and not g._adj[cnb][x]["red"]]))

    for contr, newr, turnr in contractions:
        u, v = contr
        for x in newr:
            g.add_edge(x, v, red=True)
        for x in turnr:
            g[x][v]["red"] = True
        edges = [x for x in g._adj[u].items()]
        g.remove_node(u)

        result = _solve_grid(g, ub)
        if result is not None:
            result[0].append(u)
            result[1][u] = v
            return result

        for x in newr:
            g.remove_edge(x, v)
        for x in turnr:
            g[x][v]["red"] = False
        for x, y in edges:
            g.add_edge(u, x, red=y["red"])

    return None


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


def paley49():
    inp = "0111111100101010001011100010101000111010001010100"\
    "1011111011000111001000101001010011001110001100010"\
    "1101111101000101110001010010011010000100110011100"\
    "1110111000111011010000011100100100100110100101001"\
    "1111011011010000001111010100000111011001000001101"\
    "1111101010011000110100100011100101000001111010010"\
    "1111110100100100100110001101011000110001010100011"\
    "1010001011111111000101001010101010010001011101000"\
    "0100110101111101010010110001110001011001000111000"\
    "0110100110111110100101010001001110001110000010011"\
    "1001001111011100111000001110010100111010000011010"\
    "0001110111101110101000110100000110100001111100100"\
    "1001010111110101000110100110101001000110100000111"\
    "0110001111111000011011001001010001100100111000101"\
    "1101000101010001111111010001100010110010101100010"\
    "0111000110001010111110100110110010001100010101001"\
    "0010011001110011011110110100011100010100011010010"\
    "0011010010100111101111001001110100000011100011100"\
    "1100100000110111110110001110000011101101001010100"\
    "0000111101001011111011001010001101001001100100011"\
    "1000101010001111111100110001001001110010010001101"\
    "1010100101000110010100111111110100011000101000101"\
    "1100010010011001100011011111011100001010011100100"\
    "0011100011010010100011101111001001110100100111000"\
    "0101001100100100011101110111001101000111001101000"\
    "0001101000111001101001111011110010010101000000111"\
    "1010010100101001001101111101000011101000110011010"\
    "0100011011000110010011111110100010100011010010011"\
    "1001010110001011010001000101011111110101001010001"\
    "0110001010100101110001100100101111111000100100110"\
    "1010001101001000100110111000110111100111000110100"\
    "0001110001110000110101101000111011101010011001001"\
    "0110100101010011001000000111111101100011010001110"\
    "0100110010001100001110011010111110110100101001010"\
    "1001001000110110001010010011111111001000110110001"\
    "1000101110100010100011010100110001001111111001010"\
    "1100100011100001001101100010010100110111110110001"\
    "0111000001001101101000011100101001011011111010001"\
    "1101000001101010010010101001001110011101110001110"\
    "0000111110010000011100001101101010011110110110100"\
    "0011010000011110010101010010010001111111010100110"\
    "0010011100010101100010100011000110111111101001001"\
    "1100010100010110101001101000100101010100010111111"\
    "0101001110010011000100111000011000101001101011111"\
    "1010010011100000111000010011101000101101001101111"\
    "0011100110100001010010011010000111010010011110111"\
    "1010100000011100011011100100011010000011101111011"\
    "0100011001101010100100000111010011010010101111101"\
    "0001101001001101000111000101100100101100011111110"

    g = nx.Graph()
    for i in range(0, 49):
        for j in range(0, 49):
            idx = i * 49 + j
            if inp[idx] == "1":
                g.add_edge(i, j)

    return g
