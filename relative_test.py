import os
import resource

import pysat.solvers as slv
from networkx.generators.random_graphs import gnp_random_graph

import encoding2
import heuristic
import networkx as nx

resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024))
nodes = 125
p = 0.5

results = []

with open(f"ub_{nodes}.csv", "a") as outp:
    for _ in range(0, 100):
        g = gnp_random_graph(nodes, p)
        if not nx.is_connected(g):
            continue
        print(f"{len(g.nodes)} {len(g.edges)}")

        ub = heuristic.get_ub2(g)
        enc = encoding2.TwinWidthEncoding2(g)
        cb = enc.run(g, slv.Glucose3, ub, timeout=300)
        print(f"Finished, result: {ub}/{cb}")
        outp.write(f"{ub};{cb};{p}{os.linesep}")
        outp.flush()







