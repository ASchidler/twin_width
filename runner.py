import os
import subprocess
import sys
import time
from threading import Timer

import networkx
import networkx as nx
import pysat.solvers as slv

import encoding as encoding
import encoding2 as encoding2
import encoding3
import encoding_lazy2 as lazy2

import encoding_signed_bipartite
import heuristic
import parser
import preprocessing
import tools
from networkx.generators.lattice import grid_2d_graph
from networkx.generators.random_graphs import gnp_random_graph
import resource
import treewidth
import argparse as argp


ap = argp.ArgumentParser(description="Python implementation for computing and improving decision trees.")
ap.add_argument("instance", type=str)
ap.add_argument("-e", dest="encoding", action="store", default=0, choices=[0, 1, 2, 3], type=int,
                help="The encoding to use (0=Relative, 1=Absolute, 2=Absolute Incremental, 3=Absolute with cardinality trick")
ap.add_argument("-c", dest="cubic", action="store_true", default=False,
                help="Cubic mode, reduces the number of clauses from n^4 to n^3 (only available for absolute encodings)"
                )
ap.add_argument("-t", dest="contraction", action="store_true", default=False,
                help="Use contraction hints.")
ap.add_argument("-f", dest="contraction_full", action="store_true", default=False,
                help="Use full contraction symmetry hints (requires -t).")
ap.add_argument("-l", dest="contraction_limit", action="store", default=sys.maxsize, type=int,
                help="Limit the number of contractions for which contraction hints are used (only available for absolute encodings)"
                )
ap.add_argument("-i", dest="contraction_diff", action="store_true", default=False,
                help="Use additional contraction hints with additional clauses based on the red degree of the vertex (only available for absolute encodings)")
ap.add_argument("-o", dest="order", action="store_true", default=False,
                help="Use order symmetry breaking (only available for absolute encodings).")
ap.add_argument("-d", dest="draw", action="store_true", default=False,
                help="Draw the contraction steps (uses dot).")
ap.add_argument("-v", dest="verbose", action="store_true", default=False,
                help="Verbose mode.")
ap.add_argument("-m", dest="memory", action="store", default=0, type=int,
                help="Limit maximum memory usage in GB, useful to avoid memouts when testing.")

args = ap.parse_args()

if args.memory > 0:
    resource.setrlimit(resource.RLIMIT_AS, (args.memory * 1024 * 1024 * 1024, args.memory * 1024 * 1024 * 1024))

#print(f"{tools.solve_grid2(7,7, 3)}")

instance = sys.argv[-1]
print(instance)
issat = False


if instance.endswith(".cnf"):
    issat = True
    g = parser.parse_cnf(instance)
    ub = heuristic.get_ub2_polarity(g)
    print(f"UB {ub}")

    start = time.time()

    if len(sys.argv) > 2 and not args.draw:
        cb = treewidth.solve(g.to_undirected(), len(g.nodes) - 1, slv.Glucose3, True)[1]
    else:
        enc = encoding_signed_bipartite.TwinWidthEncoding()
        cb = enc.run(g, slv.Cadical, ub)
else:
    g = parser.parse(args.instance)[0]

    if len(g.nodes) == 1:
        print("Done, width: 0")
        exit(0)

    # TODO: Deal with disconnected?
    print(f"{len(g.nodes)} {len(g.edges)}")
    preprocessing.twin_merge(g)
    ub = heuristic.get_ub(g)
    ub2 = heuristic.get_ub2(g)
    print(f"UB {ub} {ub2}")
    ub = min(ub, ub2)

    start = time.time()
    if args.encoding == 0:
        enc = encoding.TwinWidthEncoding(use_sb_static=args.contraction, use_sb_static_full=args.contraction_full)
    elif args.encoding == 1:
        enc = encoding2.TwinWidthEncoding2(g, sb_ord=args.order, sb_static=0 if not args.contraction else args.contraction_limit, sb_static_full=args.contraction_full,
                                           cubic=2 if args.cubic else 0, sb_static_diff=args.contraction_diff)
    elif args.encoding == 2:
        enc = lazy2.TwinWidthEncoding2(g, sb_ord=args.order, sb_static=0 if not args.contraction else args.contraction_limit)
    else:
        enc = encoding3.TwinWidthEncoding2(g, sb_ord=args.order,
                                           sb_static=0 if not args.contraction else args.contraction_limit,
                                           sb_static_full=args.contraction_full,
                                           cubic=2, sb_static_diff=args.contraction_diff, break_g_symmetry=True)

    cb = enc.run(g, slv.Cadical, ub, verbose=args.verbose, write=False)

print(f"Finished")
print(f"{cb}")

if args.draw:
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
