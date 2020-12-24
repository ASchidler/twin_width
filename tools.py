from pysat.formula import CNF
from pysat.card import CardEnc
import os


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
        cl = 'red' if g[x][y]['red'] else 'black'
        output1 += f"n{cln(x)} -- n{cln(y)} [color={cl}];{os.linesep}"

    # Draw the linegraph
    output2 = "strict graph dt {" + os.linesep
    u, v = min(u, v), max(u, v)
    for x, y in g.edges:
        x, y = min(x, y), max(x, y)
        color = 'green' if x == u and v == y else 'white'
        fillcolor = 'red' if g[x][y]['red'] else 'black'
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
    return _find_modules(g, ordering)


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
                oentry = lst
                passed_other = False
                passed_center = False

                for n in centry.elements:
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
    while lst is not None:
        modules.append(list(lst.elements))
        lst = lst.next_entry
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
