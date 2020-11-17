import parser
import encoding
import encoding2
import encoding3
import encoding4
import encoding5
import os
import sat_tools
import sys
import heuristic
import preprocessing
import networkx as nx
import networkx.algorithms.components.biconnected as bc

path = "/home/asc/Dev/graphs/treewidth_benchmarks/twlib-graphs/all"
instance = sys.argv[1]

g = parser.parse(os.path.join(path, instance))[0]
print(f"{len(g.nodes)} {len(g.edges)}")
preprocessing.twin_merge(g)
print(f"{len(g.nodes)} {len(g.edges)}")
preprocessing.clique_merge(g)
print(f"{len(g.nodes)} {len(g.edges)}")
preprocessing.path_merge(g)
print(f"{len(g.nodes)} {len(g.edges)}")
ub = heuristic.get_ub(g)
ub2 = heuristic.get_ub2(g)
print(f"UB {ub} {ub2}")
ub = min(ub, ub2)

sg = [list(g.nodes)]
# sg = list(bc.biconnected_components(g))
# sg.sort(key=lambda x:len(x), reverse=True)
cb = ub
print(f"{len(sg)} components")
for csg in sg:
    if len(csg) > cb:
        cg = nx.Graph()
        for n1 in csg:
            for n2 in csg:
                if n1 < n2:
                    if g.has_edge(n1, n2):
                        cg.add_edge(n1, n2)

        #st = sat_tools.SatRunner(encoding.TwinWidthEncoding, sat_tools.GlucoseSolver())
        #st = sat_tools.SatRunner(encoding.TwinWidthEncoding, sat_tools.CadicalSolver())
        st = sat_tools.SatRunner(encoding.TwinWidthEncoding, sat_tools.KissatSolver())
        #st = sat_tools.SatRunner(encoding.TwinWidthEncoding, sat_tools.MiniSatSolver())
        #st = sat_tools.SatRunner(encoding2.TwinWidthEncoding2, sat_tools.CadicalSolver())
        #st = sat_tools.SatRunner(encoding3.TwinWidthEncoding2, sat_tools.CadicalSolver())
        #st = sat_tools.SatRunner(encoding5.TwinWidthEncoding2, sat_tools.CadicalSolver())
        r, _ = st.run(cb, cg)
        cb = min(r, cb)

print(f"Finished, result: {cb}")

