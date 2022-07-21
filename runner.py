import os
import subprocess
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


#print(f"{tools.solve_grid2(7,7, 3)}")

instance = sys.argv[-1]
print(instance)

output_graphs = False
if any(x == "-l" for x in sys.argv[1:-1]):
    output_graphs = True

if instance.endswith(".cnf"):
    g = parser.parse_cnf(instance)
    ub = heuristic.get_ub2_polarity(g)
    print(f"UB {ub}")

    start = time.time()
    if len(sys.argv) > 2 and not output_graphs:
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

if output_graphs:
    instance_name = os.path.split(instance)[-1]
    mg = cb[2]
    for u, v in g.edges:
        g[u][v]["red"] = False

    for i, n in enumerate(cb[1]):
        if n not in mg:
            t = None
            n = None
        else:
            t = mg[n]
        with open(f"{instance_name}_{i}.dot", "w") as f:
            f.write(tools.dot_export(g, n, t, True))
        with open(f"{instance_name}_{i}.png", "w") as f:
            subprocess.run(["dot", "-Tpng", f"{instance_name}_{i}.dot"], stdout=f)

        if n is None:
            break

        tn = set(g.neighbors(t))
        tn.discard(n)
        nn = set(g.neighbors(n))

        for v in nn:
            if v != t:
                # Red remains, should edge exist
                if v in tn and g[n][v]['red']:
                    g[t][v]['red'] = True
                # Add non-existing edges
                if v not in tn:
                    g.add_edge(t, v, red=True)
        for v in tn:
            if v not in nn:
                g[t][v]['red'] = True
        g.remove_node(n)
