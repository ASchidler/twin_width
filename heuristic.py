from sys import maxsize

def get_ub3(g):
    od = []
    mg = {}

    g = g.copy()

    max_red = 0

    g_nodes = sorted(g.nodes, reverse=True)
    reds = {n: set() for n in g_nodes}

    while len(g.nodes) > 1:
        c_min = (maxsize, maxsize, None)
        for i, u in enumerate(g_nodes):
            if g.has_node(u):
                nbs = set(g.neighbors(u))
                c_red = reds[u]
                for v in g_nodes[i+1:]:
                    if g.has_node(v):
                        vnbs = set(g.neighbors(v))
                        final = (vnbs ^ nbs) | reds[v] | c_red
                        total = len(final)
                        final -= c_red
                        final -= reds[v]

                        for w in final:
                            total = max(total, len(reds[w]) + 1)

                        if (total, len(final), (u, v)) < c_min:
                            c_min = (total, len(final), (u, v))

        u, v = c_min[2]
        od.append(u)
        mg[u] = v

        reds[v] = reds[v] | (set(g.neighbors(u)) ^ set(g.neighbors(v))) | reds[u]
        reds[v].discard(v)
        reds[v].discard(u)

        max_red = max(max_red, len(reds[v]))
        for w in reds[v]:
            reds[w].add(v)
            max_red = max(max_red, len(reds[w]))
            reds[w].discard(u)

        g.remove_node(u)

    return max_red


def get_ub2_polarity(g):
    additional_reds = {}
    for n1 in g.nodes:
        if n1 not in additional_reds:
            additional_reds[n1] = {}
        n1suc = set(g.successors(n1))
        n1pred = set(g.predecessors(n1))
        n1nb = n1suc | n1pred
        for n2 in g.nodes:
            if n2 > n1:
                n2suc = set(g.successors(n2))
                n2pred = set(g.predecessors(n2))
                n2nb = n1suc | n1pred

                reds = n1nb ^ n2nb
                reds.update(n1suc ^ n2pred)
                reds.update(n1pred ^ n2suc)
                reds.discard(n1)
                reds.discard(n2)
                additional_reds[n1][n2] = reds

    mg = {}
    od = []
    g = g.copy()
    for u, v in g.edges:
        g[u][v]['red'] = False

    ub = 0
    reds = {x: set() for x in g.nodes}

    while len(g.nodes) > ub:
        c_min = (maxsize, maxsize, None)
        for n1 in g.nodes:
            for n2 in g.nodes:
                if n2 > n1:
                    delta = ((set(g.neighbors(n1)) ^ set(g.neighbors(n2))))\
                            | (set(g.predecessors(n1)) ^ set(g.successors(n2))) \
                            | (set(g.successors(n1)) ^ set(g.predecessors(n2)))\
                            - {n1, n2}

                    new_test = len(delta)  # len((delta | reds[n1] | reds[n2]) - {n1, n2})

                    # Test if twin
                    if len(delta) == 0:
                        c_min = (0, 0, (n1, n2, delta))
                        break
                    elif (new_test, len(set(g.neighbors(n1))) + len(set(g.neighbors(n2))), (n1, n2, delta)) < c_min:
                        c_min = (new_test, len(delta), (n1, n2, delta))

        n1, n2, delta = c_min[2]
        od.append(n1)
        mg[n1] = n2

        for cn in delta:
            reds[cn].add(n2)
            reds[n2].add(cn)
            if g.has_edge(n2, cn):
                g[n2][cn]['red'] = True
            else:
                g.add_edge(n2, cn, red=True)

            if g.has_edge(cn, n2):
                g[cn][n2]['red'] = True
            else:
                g.add_edge(cn, n2, red=True)

        for cn in g.neighbors(n1):
            if cn != n2:
                g[n2][cn]['red'] = True
                g[cn][n2]['red'] = True
                reds[cn].add(n2)
                reds[n2].add(cn)
            reds[cn].discard(n1)

        g.remove_node(n1)

        # Count reds...
        for u in g.nodes:
            cc = 0
            for v in g.neighbors(u):
                if g[u][v]['red']:
                    cc += 1
            if cc != len(reds[u]):
                print(f"mismatch {cc < len(reds[u])}")
            ub = max(ub, cc)

    return ub
