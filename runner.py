import os
import subprocess
import sys
import time

import networkx as nx
import pysat.solvers as slv

import encoding as encoding
import encoding2 as encoding2
import encoding3 as encoding3
import encoding_lazy2 as lazy2

import encoding_signed_bipartite
import heuristic
import parser
import preprocessing
import tools
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
ap.add_argument("-j", dest='maxsat', type=str, default=None, help="Export MaxSAT encoding to file.")
ap.add_argument("-p", dest="sep_cards", action="store_true", default=False, help="Store cardinalities separately")


args = ap.parse_args()

if args.memory > 0:
    resource.setrlimit(resource.RLIMIT_AS, (args.memory * 1024 * 1024 * 1024, args.memory * 1024 * 1024 * 1024))

instance = args.instance
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
        cb = enc.run(g, slv.Cadical153, ub)

    print(f"Finished")
    print(f"{cb}")
else:
    g = parser.parse(args.instance)[0]

    if len(g.nodes) == 1:
        print("Finished")
        print("[0, [0], {}]")
        exit(0)

    eliminations = []
    parents = {}
    max_tww = None
    single_nodes = []

    print(f"{len(g.nodes)} {len(g.edges)}")
    twins = preprocessing.twin_merge(g)
    for parent, child in twins:
        eliminations.append(child)
        parents[child] = parent

    cub = heuristic.get_ub3(g)
    print(f"UB {cub}")

    start = time.time()

    comps = list(nx.connected_components(g))

    for cci, cc in enumerate(comps):
        print(f"Component {cci} ({len(cc)} Nodes)")
        if len(cc) == 1:
            single_nodes.append(next(iter(cc)))
        else:
            cc = list(cc)
            n_map = {x: i+1 for i, x in enumerate(cc)}
            r_map = {i: x for x, i in n_map.items()}

            ng = nx.Graph()
            for n1, n2 in g.edges:
                if n1 in n_map:
                    ng.add_edge(n_map[n1], n_map[n2])

            ub = heuristic.get_ub3(ng)

            if args.encoding == 0:
                enc = encoding.TwinWidthEncoding(use_sb_static=args.contraction, use_sb_static_full=args.contraction_full)
            elif args.encoding == 1:
                enc = encoding2.TwinWidthEncoding2(ng, sb_ord=args.order, sb_static=0 if not args.contraction else args.contraction_limit, sb_static_full=args.contraction_full,
                                                   cubic=2 if args.cubic else 0, sb_static_diff=args.contraction_diff)
            elif args.encoding == 2:
                enc = lazy2.TwinWidthEncoding2(ng, sb_ord=args.order, sb_static=0 if not args.contraction else args.contraction_limit)
            else:
                enc = encoding3.TwinWidthEncoding2(ng, sb_ord=args.order,
                                                   sb_static=0 if not args.contraction else args.contraction_limit,
                                                   sb_static_full=args.contraction_full,
                                                   cubic=2, sb_static_diff=args.contraction_diff, break_g_symmetry=True)

            cb = enc.run(ng, slv.Cadical153, ub-1, verbose=args.verbose, write=True, steps_limit=None)

            remaining = set(ng.nodes)
            remaining -= set(cb[1])
            remaining = list(remaining)
            eliminations.extend([r_map[x] for x in cb[1]])

            if len(remaining) >= 1:
                for ck, cv in cb[2].items():
                    parents[r_map[ck]] = r_map[cv]
                for cn in remaining[:-1]:
                    eliminations.append(r_map[cn])
                    parents[r_map[cn]] = r_map[remaining[-1]]

                single_nodes.append(r_map[remaining[-1]])

            if max_tww is None or cb[0] > max_tww:
                max_tww = cb[0]

    for cn in single_nodes[:-1]:
        eliminations.append(cn)
        parents[cn] = single_nodes[-1]

    print(f"Finished")
    print(f"({max_tww}, {eliminations}, {parents})")

if args.draw:
    instance_name = os.path.split(instance)[-1]
    mg = cb[2] if issat else parents
    for u, v in g.edges:
        g[u][v]["red"] = False

    for i, n in enumerate(cb[1] if issat else eliminations):
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
