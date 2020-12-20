import parser
import encoding
import encoding2
import encoding3
import encoding4
import encoding5
import os
import sys
import heuristic
import preprocessing
import networkx as nx
import networkx.algorithms.components.biconnected as bc
import time
import pysat.solvers as slv

path = "/home/asc/Dev/TCW_TD_to_SAT/inputs/famous"

for cf in os.listdir(path):
    if os.path.isfile(os.path.join(path, cf)):
        print(cf)
        instance = os.path.join(path, cf)

        g = parser.parse(os.path.join(path, instance))[0]
        print(f"{len(g.nodes)} {len(g.edges)}")
        preprocessing.twin_merge(g)
        print(f"{len(g.nodes)} {len(g.edges)}")
        preprocessing.clique_merge(g)
        print(f"{len(g.nodes)} {len(g.edges)}")
        preprocessing.path_merge(g)
        print(f"{len(g.nodes)} {len(g.edges)}")
        if len(g.nodes) == 1:
            print("Done, width: {1}")
            continue

        ub = heuristic.get_ub(g)
        ub2 = heuristic.get_ub2(g)
        print(f"UB {ub} {ub2}")
        ub = min(ub, ub2)

        sg = [list(g.nodes)]
        print(f"{len(list(bc.biconnected_components(g)))} components")
        # sg = list(bc.biconnected_components(g))
        # sg.sort(key=lambda x:len(x), reverse=True)
        cb = ub
        #print(f"{len(sg)} components")
        for csg in sg:
            if len(csg) > cb:
                cg = nx.Graph()
                for n1 in csg:
                    for n2 in csg:
                        if n1 < n2:
                            if g.has_edge(n1, n2):
                                cg.add_edge(n1, n2)

                start = time.time()
                enc = encoding.TwinWidthEncoding()
                # enc = encoding2.TwinWidthEncoding2(g)
                formula = enc.encode(g, ub)
                print(f"Created encoding in {time.time() - start}")

                with slv.Cadical() as solver:
                    solver.append_formula(formula)
                    for i in range(ub, -1, -1):
                        start = time.time()

                        if solver.solve(enc.get_card_vars(i, solver)):
                            cb = i
                            print(f"Found {i}")
                            enc.decode(solver.get_model(), g, i)
                        else:
                            print(f"Failed {i}")
                            print(f"Finished cycle in {time.time() - start}")
                            break

        print(f"Finished, result: {cb}\n\n")

