import parser
import encoding
import encoding2
import os
import sat_tools
import sys
import heuristic

path = "/home/aschidler/Downloads/twlib-filtered"
instance = sys.argv[1]

g = parser.parse(os.path.join(path, instance))[0]
print(f"{len(g.nodes)} {len(g.edges)}")
ub = heuristic.get_ub(g)
print(f"UB {ub}")

#st = sat_tools.SatRunner(encoding.TwinWidthEncoding, sat_tools.GlucoseSolver())
st = sat_tools.SatRunner(encoding2.TwinWidthEncoding2, sat_tools.GlucoseSolver())
r, _ = st.run(ub, g)


