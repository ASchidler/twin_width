import os
import sys
import time
from multiprocessing import Pool

import networkx as nx
from networkx.generators.random_graphs import gnp_random_graph
from pysat.solvers import Cadical

import encoding, encoding2, encoding3
import heuristic

increment = 0.02
num_graphs = 100
g_min_size = 20
g_increment = 5
g_max_size = 20

results = {}

pool_size = 2

tenc = int(sys.argv[1])

def compute_graph(args):
    c_p, c_g_size = args
    g = gnp_random_graph(c_g_size, c_p)
    components = list(nx.connected_components(g))
    tww = 0
    start = time.time()
    for c in components:
        cg = nx.Graph()
        for u in c:
            for v in g[u]:
                cg.add_edge(u, v)

        if tenc == 0:
            enc = encoding.TwinWidthEncoding(g)
        elif tenc == 1:
            enc = encoding2.TwinWidthEncoding2(g, cubic=2, sb_ord=False, sb_static=0, sb_static_full=True, sb_static_diff=False)
        else:
            enc = encoding3.TwinWidthEncoding2(g, cubic=2, sb_ord=False, sb_static=0, sb_static_full=True, sb_static_diff=False)
        # enc = encoding_lazy2.TwinWidthEncoding2(g, cubic=True, sb_ord=True)
        result = enc.run(cg, Cadical, heuristic.get_ub(cg), check=False, verbose=False)
        if isinstance(result, int):
            tww = max(tww, result)
        else:
            tww = max(tww, result[0])
    return tww, (time.time() - start)


with open(f"random_results_{tenc}.csv", "w") as results_f:
    for g_size in range(g_min_size, g_max_size+1, g_increment):
        cp = increment
        results[g_size] = {}
        while cp < 1:
            with Pool(pool_size) as pool:
                c_total = 0
                finished = 0
                total_rt = 0
                for c_tww, c_rt in pool.imap_unordered(compute_graph, ((cp, g_size) for _ in range(0, num_graphs))):
                    c_total += c_tww
                    total_rt += c_rt
                    finished += 1
                pool.close()
                pool.join()
                results[g_size][cp] = c_total / num_graphs
                total_rt /= num_graphs
                print(f"{g_size} {cp} {results[g_size][cp]} {total_rt:.2f}")
                results_f.write(f"{g_size};{cp};{results[g_size][cp]};{total_rt}{os.linesep}")
                results_f.flush()
                cp += increment
