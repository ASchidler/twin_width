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


def get_ub2(g):
    additional_reds = {}
    for n1 in g.nodes:
        if n1 not in additional_reds:
            additional_reds[n1] = {}
        n1nb = set(g.neighbors(n1))
        for n2 in g.nodes:
            if n2 > n1:
                n2nb = set(g.neighbors(n2))
                reds = n1nb ^ n2nb
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
                    delta = (set(g.neighbors(n1)) ^ set(g.neighbors(n2))) - {n1, n2}
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

        for cn in g.neighbors(n1):
            if cn != n2:
                g[n2][cn]['red'] = True
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
