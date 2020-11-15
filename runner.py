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

path = "/home/asc/Dev/graphs/treewidth_benchmarks/twlib-graphs/all"
instance = sys.argv[1]


g = parser.parse(os.path.join(path, instance))[0]
print(f"{len(g.nodes)} {len(g.edges)}")
ub = heuristic.get_ub(g)
print(f"UB {ub}")

#st = sat_tools.SatRunner(encoding.TwinWidthEncoding, sat_tools.GlucoseSolver())
#st = sat_tools.SatRunner(encoding.TwinWidthEncoding, sat_tools.CadicalSolver())
#st = sat_tools.SatRunner(encoding.TwinWidthEncoding, sat_tools.KissatSolver())
#st = sat_tools.SatRunner(encoding.TwinWidthEncoding, sat_tools.MiniSatSolver())
#st = sat_tools.SatRunner(encoding2.TwinWidthEncoding2, sat_tools.CadicalSolver())
st = sat_tools.SatRunner(encoding3.TwinWidthEncoding2, sat_tools.CadicalSolver())
#st = sat_tools.SatRunner(encoding5.TwinWidthEncoding2, sat_tools.CadicalSolver())
r, _ = st.run(ub, g)
