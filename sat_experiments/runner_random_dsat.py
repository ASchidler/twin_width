import sys
import time
import random

import networkx
import pysat.solvers as slv

import encoding as encoding
from networkx.generators.random_graphs import random_regular_graph
from multiprocessing import Pool

nodes = int(sys.argv[1])
threads = int(sys.argv[2])

print(f"Nodes: {nodes}")


def worker(wid):
    while True:
        if nodes % 4 == 1:
            g = random_regular_graph((nodes - 1) // 2, nodes, random.seed(time.time() + wid * 5))
        else:
            g = random_regular_graph((nodes - 1) // 2 + 1, nodes, random.seed(time.time() + wid * 5))

        enc = encoding.TwinWidthEncoding()
        cb = enc.run(g, slv.Cadical, (nodes - 1) // 2 + 5, verbose=False)
        print(f"Result: {cb}")
        if cb[0] >= (nodes - 1) // 2:
            print(f"Graph: {networkx.to_graph6_bytes(g)}")
        sys.stdout.flush()


with Pool(threads) as cp:
    cp.map(worker, range(0, threads))
