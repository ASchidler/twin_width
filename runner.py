import os
import sys
import time

import networkx as nx
import pysat.solvers as slv

import encoding
import encoding2
import heuristic
import parser
import preprocessing
from networkx.generators.lattice import grid_2d_graph
import tools

path1 = "/home/asc/Dev/graphs/treewidth_benchmarks/twlib-graphs/all"
path2 = "/home/asc/Dev/TCW_TD_to_SAT/inputs/famous"
instance = sys.argv[1]

if os.path.exists(os.path.join(path1, instance)):
    path = os.path.join(path1, instance)
else:
    path = os.path.join(path2, instance)

g = parser.parse(path)[0]
from networkx.generators.expanders import paley_graph
from networkx import Graph, from_graph6_bytes

#g = tools.prime_square_paley(7)
#tools.solve_paley(g)
g = tools.prime_paley(13)
# g = tools.line(6)
#g = tools.rook(5)

#g = from_graph6_bytes(bytes('ECZW', encoding="ascii"))
# from networkx.generators.lattice import grid_2d_graph
#g = grid_2d_graph(10, 10)
#g = tools.paley49()
print(f"{len(g.nodes)} {len(g.edges)}")

x = tools.find_modules(g)

#
print(f"{len(g.nodes)} {len(g.edges)}")
preprocessing.twin_merge(g)
# print(f"{len(g.nodes)} {len(g.edges)}")
# preprocessing.clique_merge(g)
# print(f"{len(g.nodes)} {len(g.edges)}")
# preprocessing.path_merge(g)
# print(f"{len(g.nodes)} {len(g.edges)}")

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

