import sys
import time

import pysat.solvers as slv

import encoding
import heuristic
import parser
import preprocessing
import tools

instance = sys.argv[1]

g = parser.parse(instance)[0]

print(f"{len(g.nodes)} {len(g.edges)}")

x = tools.find_modules(g)

#
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

