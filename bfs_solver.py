import argparse as argp

import networkx as nx

import parser
import pynauty


ap = argp.ArgumentParser(description="Python implementation for computing and improving decision trees.")
ap.add_argument("instance", type=str)

args = ap.parse_args()

instance = args.instance
print(instance)

g = parser.parse(args.instance)[0]
node_map = {x: i+1 for i, x in enumerate(g.nodes)}
ng = nx.Graph()
for cn in node_map.values():
    ng.add_node(cn)
for cn1, cn2 in g.edges:
    ng.add_edge(node_map[cn1], node_map[cn2])
g = ng

current_list = {}
last_list = None


def partition_to_edges(c_part, cx, cy):
    pos = [set(), set()]
    neg = [set(), set()]
    all_nodes = set(g.nodes)

    for z in [ci for ci, ct in enumerate(c_part) if ct == cx]:
        pos[0].update({c_part[x] for x in g.neighbors(z)})
        neg[0].update({c_part[x] for x in (all_nodes - set(g.neighbors(z)))})

    for z in [ci for ci, ct in enumerate(c_part) if ct == cy]:
        pos[1].update({c_part[x] for x in g.neighbors(z)})
        neg[1].update({c_part[x] for x in all_nodes - set(g.neighbors(z))})

    creds = [pos[0] & neg[0], pos[1] & neg[1]]
    pos[0] = {x for x in pos[0] if x == c_part[x]}
    pos[1] = {x for x in pos[1] if x == c_part[x]}
    pos[0] -= creds[0]
    pos[1] -= creds[1]

    # Remove the contracted and contraction vertex from sets
    for centries in [*creds, *pos]:
        centries.difference_update({cx, cy})

    new_reds = pos[0] ^ pos[1]
    new_reds |= creds[0] | creds[1]
    new_reds.discard(cx)
    new_reds.discard(cy)


if last_list is None:
    gn = pynauty.Graph(len(g.nodes))
    for cn in g.nodes:
        gn.connect_vertex(cn - 1, [cb - 1 for cb in g.neighbors(cn)])
    grp = pynauty.autgrp(gn)
    orbits = grp[3]

    for cn1 in range(1, len(g.nodes)+1):
        partition = [i for i in range(0, len(g.nodes) + 1)]
        for cn2 in range(cn1+1, len(g.nodes)+1):
            # Do contraction
            pass

    print(orbits)
else:
    for centry in last_list:

        for cn in range(1, len(g.nodes)+1):
            if partition[cn] == cn:
               for cn2 in range(cn+1, len(g.nodes)+1):
                if partition[cn2] == cn2:
                    # Check contraction
                    pass


# Don't forget twins!
