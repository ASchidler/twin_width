import os
import sys
from multiprocessing import Pool

import networkx as nx
from networkx.generators.random_graphs import gnp_random_graph
from pysat.solvers import Cadical

import encoding, encoding2, encoding_lazy2
import heuristic

increment = 0.02
num_graphs = 100
g_min_size = 30
g_increment = 5
g_max_size = 50

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

        # enc = encoding.TwinWidthEncoding(g)
        enc = encoding2.TwinWidthEncoding2(g, cubic=2, sb_ord=False, sb_static=sys.maxsize, sb_static_full=True, sb_static_diff=True)
        # enc = encoding_lazy2.TwinWidthEncoding2(g, cubic=True, sb_ord=True)
        result = enc.run(cg, Cadical, heuristic.get_ub(cg), check=False, verbose=False)
        if isinstance(result, int):
            tww = max(tww, result)
        else:
            tww = max(tww, result[0])
    return tww


with open("random_results.csv", "w") as results_f:
    for g_size in range(g_min_size, g_max_size+1, g_increment):
        cp = increment
        results[g_size] = {}
        while cp < 1:
            with Pool(pool_size) as pool:
                c_total = 0
                finished = 0
                for c_tww in pool.imap_unordered(compute_graph, ((cp, g_size) for _ in range(0, num_graphs))):
                    c_total += c_tww
                    finished += 1
                    print(f"Finished {finished}")
                pool.close()
                pool.join()
                results[g_size][cp] = c_total / num_graphs
                print(f"{g_size} {cp} {results[g_size][cp]}")
                results_f.write(f"{g_size};{cp};{results[g_size][cp]}{os.linesep}")
                results_f.flush()
                cp += increment
