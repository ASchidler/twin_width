import sys
import time

import pysat.solvers as slv

import encoding3 as encoding
import heuristic
import parser
import preprocessing
import tools

print(f"{tools.solve_grid2(7,7, 3)}")

instance = sys.argv[1]

g = parser.parse(instance)[0]
import networkx as nx
d1 = 8
d2 = 8
g = nx.Graph()
for i in range(0, d1):
    for j in range(0, d2):
        for xoff, yoff in [(-1, 0), (1, 0), (0, -1), (-1, 0)]:
            if 0 <= i + xoff < d1 and 0 <= j + yoff < d2:
                g.add_edge((i, j), (i+xoff, j+yoff))
print(f"{tools.solve_quick(g, 3)}")

print(f"{len(g.nodes)} {len(g.edges)}")

x = tools.find_modules(g)

#
print(f"{len(g.nodes)} {len(g.edges)}")
preprocessing.twin_merge(g)

if len(g.nodes) == 1:
    print("Done, width: 0")
    exit(0)
g = tools.prime_paley(53)
# TODO: Deal with disconnected?
ub = heuristic.get_ub(g)
ub2 = heuristic.get_ub2(g)
print(f"UB {ub} {ub2}")
ub = min(ub, ub2)

start = time.time()
enc = encoding.TwinWidthEncoding()
# enc = encoding2.TwinWidthEncoding2(g)
cb = enc.run(g, slv.Cadical, ub)

print(f"Finished, result: {cb}")

