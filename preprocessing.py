def twin_merge(g):
    changed = True
    while changed:
        changed = False
        nodes = sorted(list(g.nodes))
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
                        changed = True
                        break