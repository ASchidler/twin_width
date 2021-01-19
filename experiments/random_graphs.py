import os
from multiprocessing import Pool

import networkx as nx
from networkx.generators.random_graphs import gnp_random_graph
from pysat.solvers import Cadical

import encoding
import heuristic

increment = 0.02
num_graphs = 100
g_min_size = 10
g_increment = 5
g_max_size = 40

results = {}

pool_size = 4


def compute_graph(args):
    c_p, c_g_size = args
    g = gnp_random_graph(c_g_size, c_p)
    components = list(nx.connected_components(g))
    tww = 0
    for c in components:
        cg = nx.Graph()
        for u in c:
            for v in g[u]:
                cg.add_edge(u, v)

        enc = encoding.TwinWidthEncoding()
        result = enc.run(cg, Cadical, heuristic.get_ub(cg), check=False, verbose=False)
        tww = max(tww, result)
    return tww


with open("random_results.csv", "w") as results_f:
    for g_size in range(g_min_size, g_max_size+1, g_increment):
        cp = increment
        results[g_size] = {}
        while cp < 1:
            with Pool(pool_size) as pool:
                c_total = 0
                for c_tww in pool.imap_unordered(compute_graph, ((cp, g_size) for _ in range(0, num_graphs))):
                    c_total += c_tww
                pool.close()
                pool.join()
                results[g_size][cp] = c_total / num_graphs
                print(f"{g_size} {cp} {results[g_size][cp]}")
                results_f.write(f"{g_size};{cp};{results[g_size][cp]}{os.linesep}")
                cp += increment
