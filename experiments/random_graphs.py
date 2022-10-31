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
g_min_size = 15
g_increment = 5
g_max_size = 15

results = {}

pool_size = 5

tenc = int(sys.argv[1])
output_path = f"random_results_{tenc}.csv"

def compute_graph(args):
    c_p, c_g_size = args
    g = gnp_random_graph(c_g_size, c_p)
    if c_p > 0.5:
        g = nx.complement(g)
    # for cl in nx.generate_edgelist(g):
    #     print(cl)

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
            enc = encoding3.TwinWidthEncoding2(g, cubic=2, sb_ord=True, sb_static=sys.maxsize, sb_static_full=True, sb_static_diff=False)
        # enc = encoding_lazy2.TwinWidthEncoding2(g, cubic=True, sb_ord=True)
        result = enc.run(cg, Cadical, heuristic.get_ub(cg), check=False, verbose=False)
        if isinstance(result, int):
            tww = max(tww, result)
        else:
            tww = max(tww, result[0])
    return tww, (time.time() - start), c_p, c_g_size

c_nodes = None
c_cp = None

if os.path.exists(output_path):
    with open(output_path) as inp:
        for cl in inp:
            fields = cl.split(";")
            c_nodes = int(fields[0])
            c_cp = float(fields[1])

if c_nodes is not None:
    print(f"Starting from {c_nodes}/{c_cp}")
else:
    c_nodes = g_min_size

c_nodes = max(c_nodes, g_min_size)


def enumerate_graphs(start_c, start_p):
    for g_size in range(start_c, g_max_size + 1, g_increment):
        if start_p is not None:
            cp = start_p + increment
            start_p = None
        else:
            cp = increment

        results[g_size] = {}
        while cp < 1:
            cp = round(cp, 2)
            for _ in range(0, num_graphs):
                yield cp, g_size
            cp += increment


with open(output_path, "a") as results_f:
    with Pool(pool_size) as pool:
        curr_cp = None
        curr_size = None

        for c_tww, c_rt, c_cp, c_cs in pool.imap(compute_graph, enumerate_graphs(c_nodes, c_cp)):
            if curr_cp != c_cp or curr_size != c_cs:
                if curr_cp is not None and curr_size is not None:
                    c_total /= num_graphs
                    total_rt /= num_graphs
                    print(f"{curr_size} {curr_cp} {c_total:.2f} {total_rt:.2f} Graphs: {finished}")
                    sys.stdout.flush()
                    results_f.write(f"{curr_size};{curr_cp};{c_total};{total_rt}{os.linesep}")
                    results_f.flush()

                c_total = 0
                finished = 0
                total_rt = 0
                curr_cp = c_cp
                curr_size = c_cs
            c_total += c_tww
            total_rt += c_rt
            finished += 1

        pool.close()
        pool.join()
