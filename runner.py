import os
import subprocess
import sys
import time
from threading import Timer

import networkx
import pysat.solvers as slv

import encoding as encoding
import encoding_lazy as lazy
import encoding_lazy2 as lazy2
import encoding5 as encoding2

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
issat = False

output_graphs = False
if any(x == "-l" for x in sys.argv[1:-1]):
    output_graphs = True

if instance.endswith(".cnf"):
    issat = True
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
    # g = parser.parse(instance)[0]
    g = tools.prime_paley(29)
    # g = tools.prime_square_paley(9)

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
    #enc = encoding.TwinWidthEncoding()
    # enc = lazy.TwinWidthEncoding()
    # enc = encoding2.TwinWidthEncoding2(g)
    enc = lazy2.TwinWidthEncoding2(g)
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
            f.write(tools.dot_export(g, n, t, issat))
        with open(f"{instance_name}_{i}.png", "w") as f:
            subprocess.run(["dot", "-Tpng", f"{instance_name}_{i}.dot"], stdout=f)

        if n is None:
            break

        if issat:
            tns = set(g.successors(t))
            tnp = set(g.predecessors(t))
            nns = set(g.successors(n))
            nnp = set(g.predecessors(n))

            nn = nns | nnp
            tn = tns | tnp
        else:
            nn = set(g.neighbors(n))
            tn = set(g.neighbors(t))

        for v in nn:
            if v != t:
                # Red remains, should edge exist
                if (v in g[n] and g[n][v]['red']) or v not in tn or (issat and v in nns and v not in tns) or (
                        issat and v in nnp and v not in tnp):
                    if g.has_edge(t, v):
                        g[t][v]['red'] = True
                    elif g.has_edge(v, t):
                        g[v][t]['red'] = True
                    else:
                        g.add_edge(t, v, red=True)

        for v in tn:
            if v not in nn:
                if g.has_edge(t, v):
                    g[t][v]['red'] = True
                else:
                    g.add_edge(t, v, red=True)

                if g.has_edge(v, t):
                    g[v][t]['red'] = True
                else:
                    g.add_edge(v, t, red=True)

        if issat:
            for u in list(g.successors(n)):
                g.remove_edge(n, u)
            for u in list(g.predecessors(n)):
                g.remove_edge(u, n)
        else:
            for u in list(g.neighbors(n)):
                g.remove_edge(u, n)
        g.nodes[n]["del"] = True
