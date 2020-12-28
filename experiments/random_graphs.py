import os
import networkx as nx
from networkx.generators.random_graphs import gnp_random_graph
import encoding
from pysat.solvers import Cadical
import heuristic
from multiprocessing import Pool, Value

increment = 0.02
num_graphs = 100
g_min_size = 10
g_increment = 5
g_max_size = 40

results = {}

c_g_size = Value("i")
c_p = Value("f")
c_result = Value("i")

pool_size = 4


def compute_graph(i):
    g = gnp_random_graph(g_size, cp)
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
    c_result.value += tww


with open("random_results.csv", "w") as results_f:
    for g_size in range(g_min_size, g_max_size+1, g_increment):
        c_g_size.value = g_size
        cp = increment
        results[g_size] = {}
        while cp < 1:
            with Pool(pool_size) as pool:
                c_result.value = 0
                c_p.Value = cp
                pool.map(compute_graph, range(0, num_graphs))
                pool.close()
                pool.join()
                results[g_size][cp] = c_result.value / num_graphs
                print(f"{g_size} {cp} {results[g_size][cp]}")
                results_f.write(f"{g_size};{cp};{results[g_size][cp]}{os.linesep}")
                cp += increment
