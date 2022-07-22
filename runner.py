import sys
import time
from threading import Timer

import networkx
import pysat.solvers as slv

import encoding3 as encoding
import encoding2
import encoding_signed_bipartite
import heuristic
import parser
import preprocessing
import tools
from networkx.generators.lattice import grid_2d_graph
from networkx.generators.random_graphs import gnp_random_graph
import resource
import treewidth

resource.setrlimit(resource.RLIMIT_AS, (16 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024))

#print(f"{tools.solve_grid2(7,7, 3)}")

instance = sys.argv[-1]
print(instance)

if instance.endswith(".cnf"):
    g = parser.parse_cnf(instance)
    ub = heuristic.get_ub2_polarity(g)
    print(f"UB {ub}")

    start = time.time()
    if len(sys.argv) > 2:
        cb = treewidth.solve(g.to_undirected(), len(g.nodes) - 1, slv.Glucose3, True)[1]
    else:
        enc = encoding_signed_bipartite.TwinWidthEncoding()
        cb = enc.run(g, slv.Cadical, ub)
else:
    g = parser.parse(instance)[0]

    print(f"{len(g.nodes)} {len(g.edges)}")
    preprocessing.twin_merge(g)

    if len(g.nodes) == 1:
        print("Done, width: 0")
        exit(0)

    # TODO: Deal with disconnected?
    ub = heuristic.get_ub(g)
    ub2 = heuristic.get_ub2(g)
    print(f"UB {ub} {ub2}")
    ub = min(ub, ub2)

    start = time.time()
    enc = encoding.TwinWidthEncoding()
    #enc = encoding2.TwinWidthEncoding2(g)

    cb = enc.run(g, slv.Cadical, ub)

print(f"Finished, result: {cb}")

