import parser
import encoding
import encoding2
import os
import sys
import heuristic
import preprocessing
import time
import pysat.solvers as slv

path = "/home/asc/Dev/TCW_TD_to_SAT/inputs/famous"
graphs = []
results = {}

for cf in os.listdir(path):
    if os.path.isfile(os.path.join(path, cf)):
        instance = os.path.join(path, cf)
        g = parser.parse(os.path.join(path, instance))[0]
        graphs.append((g, cf))

graphs.sort(key=lambda x: (len(x[0].nodes), len(x[0].edges)))
for g, cf in graphs:
    print(f"{cf}: {len(g.nodes)}")
    preprocessing.twin_merge(g)
    if len(g.nodes) == 1:
        print("Finished, result: 0\n\n")
        results[cf] = 0
        continue

    ub = min(heuristic.get_ub(g), heuristic.get_ub2(g))

    cb = ub

    start = time.time()
    enc = encoding.TwinWidthEncoding()
    result = enc.run(g, slv.Cadical, ub)

    print(f"Finished, result: {result}\n\n")
    results[cf] = result

results = [(k, v) for k, v in results.items()]
results.sort()
for k, v in results:
    print(f"{k}: {v}")
