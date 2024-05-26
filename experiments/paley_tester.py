import sys

from networkx.generators.lattice import grid_2d_graph
import tools
import pysat.solvers as slv
import encoding as encoding
import encoding_lazy2 as lazy
import encoding2
import encoding3
import tools
import heuristic

g = tools.prime_paley(73)
# g = tools.prime_paley(37)

ub = heuristic.get_ub(g)
ub2 = heuristic.get_ub2(g)
ub = min(ub, ub2)
print(len(g.nodes))

ub = 36

enc = encoding.TwinWidthEncoding(use_sb_static=False, use_sb_static_full=False)
# enc = encoding2.TwinWidthEncoding2(g, cubic=2, sb_ord=False, sb_static=False, sb_static_full=False, is_grid=False)
# enc = encoding3.TwinWidthEncoding2(g, cubic=2, sb_static=0, sb_ord=False, sb_static_full=True, sb_static_diff=False)
result = enc.run(g, slv.Cadical, ub, write=True)

print(f"Result: {result[0]}")

# encl = lazy.TwinWidthEncoding2(g, cubic=True, sb_ord=True, sb_static=len(g.nodes)//2, sb_red=False, use_sb_static_full=True)
# encl.run(g, solver=slv.Cadical, start_bound=3)


