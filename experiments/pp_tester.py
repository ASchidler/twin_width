import lzma
import os
import networkx as nx
import tools
from networkx.algorithms import connected_components
import preprocessing as pp

inp_pth = "/Users/andre/Downloads/exact-public-2"

for cf in sorted(os.listdir(inp_pth)):
    if not cf.endswith(".xz"):
        continue
    edges = []
    with lzma.open(os.path.join(inp_pth, cf)) as inp:
        for i, cl in enumerate(inp):
            cl = cl.decode("ascii")
            if not cl.startswith("p ") and not cl.startswith("c "):
                edges.append([int(x) for x in cl.strip().split()])
    g = nx.from_edgelist(edges)
    print(f"{cf} {len(g.nodes)}")
    pp.twin_merge(g)
    print(f"{cf} {len(g.nodes)}")
    cp = nx.connected_components(g)
    print(f"{cf} {'/'.join(str(len(x)) for x in cp)}")
    # ms = tools.find_modules(g)
    # print(f"{cf} {'/'.join(str(len(x)) for x in ms if len(x) > 1)}")
    print("")

