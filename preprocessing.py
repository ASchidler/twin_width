def twin_merge(g):
    nodes = list(g.nodes)
    for n1 in nodes:
        nb1 = set(g.neighbors(n1))
        for n2 in nodes:
            if n1 < n2:
                nb2 = set(g.neighbors(n2))
                nbs = nb1 ^ nb2
                nbs.discard(n1)
                nbs.discard(n2)
                if len(nbs) == 0:
                    g.remove_node(n1)
                break


def clique_merge(g):
    nodes = list(g.nodes)
    for n in nodes:
        nbs = set(g.neighbors(n))
        is_clique = True
        for cn in nbs:
            if len(set(g.neighbors(cn)) & nbs) != len(nbs) - 1: # -1 for the node itself
                is_clique = False
                break

        if is_clique:
            for cn in nbs:
                nb2 = set(g.neighbors(cn))
                nbs = nbs ^ nb2
                nbs.discard(n)
                nbs.discard(cn)
                if len(nbs) == 0:
                    g.remove_node(n)
                    break


def path_merge(g):
    nodes = list(g.nodes)
    for n in nodes:
        nbs = list(g.neighbors(n))
        if len(nbs) == 2:
            if len(list(g.neighbors(nbs[0]))) == 2 or len(list(g.neighbors(nbs[1]))) == 2:
                g.add_edge(nbs[0], nbs[1])
                g.remove_node(n)
