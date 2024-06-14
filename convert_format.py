import sys

import networkx as nx

g = nx.Graph()
with open(sys.argv[1]) as inp:
    for cl in inp:
        if cl.startswith("p "):
            _, _, nodes, edges = cl.split(" ")
            nodes = int(nodes)
            edges = int(edges.strip())
            for ci in range(1, nodes+1):
                g.add_node(ci)

        elif cl.startswith("e "):
            _, n1, n2 = cl.split(" ")
            g.add_edge(int(n1), int(n2.strip()))

print(f"p tww {len(g.nodes)} {len(g.edges)}")
for cl in nx.generate_edgelist(g, data=False):
    print(cl)
