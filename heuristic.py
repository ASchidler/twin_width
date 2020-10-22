from sys import maxsize


def get_ub(g):
    od = []
    mg = {}

    g = g.copy()
    for u, v in g.edges:
        g[u][v]['red'] = False

    c_max = 0
    while len(g.nodes) > 1:
        # Pick next node
        c_min = maxsize, (0, 0)

        for u in g.nodes:
            for v in g.neighbors(u):
                c_len = len(set(g.neighbors(u)) ^ set(g.neighbors(v)))
                if c_len < c_min[0]:
                    c_min = (c_len, (u, v))

        n = c_min[1][0]
        t = c_min[1][1]

        tn = set(g.neighbors(t))
        tn.discard(n)
        nn = set(g.neighbors(n))
        od.append(n)
        mg[n] = t

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

    return c_max
