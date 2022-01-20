import sys
import time
import pysat.solvers as slv

import encoding
import encoding2
import heuristic
import parser
import preprocessing
import resource

# Use to limit memory to avoid killing the workstation
#resource.setrlimit(resource.RLIMIT_AS, (16 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024))

instance = sys.argv[1]
g = parser.parse(instance)[0]

print(f"Nodes: {len(g.nodes)} Edges: {len(g.edges)}")
preprocessing.twin_merge(g)

if len(g.nodes) == 1:
    print("Done, width: 0")
    exit(0)

ub = heuristic.get_ub(g)
ub2 = heuristic.get_ub2(g)
print(f"Upper Bound 1: {ub}, Upper Bound 2: {ub2}")
ub = min(ub, ub2)

start = time.time()
enc = encoding.TwinWidthEncoding()

# Encoding 2 is (much) slower but can be used for larger graphs
#enc = encoding2.TwinWidthEncoding2(g)

cb = enc.run(g, slv.Cadical, ub)

print(f"Finished, Twin-Width: {cb[0]}")
print("Contractions:")
for cn in cb[1][:-1]:
    print(f"{cn} => {cb[2][cn]}")
