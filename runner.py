import sys
import time
from threading import Timer
import pysat.solvers as slv

import encoding3 as encoding
import encoding2
import heuristic
import parser
import preprocessing
import tools
from networkx.generators.lattice import grid_2d_graph
from networkx.generators.random_graphs import gnp_random_graph
import resource

resource.setrlimit(resource.RLIMIT_AS, (16 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024))

#print(f"{tools.solve_grid2(7,7, 3)}")

instance = sys.argv[1]

g = parser.parse(instance)[0]

# print(f"{len(g.nodes)} {len(g.edges)}")
#
# x = tools.find_modules(g)

#
def interrupt():
    exit(1)

timer = Timer(300, interrupt)
timer.start()

print(f"{len(g.nodes)} {len(g.edges)}")
preprocessing.twin_merge(g)

if len(g.nodes) == 1:
    print("Done, width: 0")
    exit(0)
#g = tools.prime_paley(73)
g = grid_2d_graph(7, 7)
# g = gnp_random_graph(75, 0.5)
# TODO: Deal with disconnected?
ub = heuristic.get_ub(g)
ub2 = heuristic.get_ub2(g)
print(f"UB {ub} {ub2}")
ub = min(ub, ub2)

start = time.time()
enc = encoding.TwinWidthEncoding()
#enc = encoding2.TwinWidthEncoding2(g)
ub = 3
cb = enc.run(g, slv.Cadical, ub)

print(f"Finished, result: {cb}")
timer.cancel()
