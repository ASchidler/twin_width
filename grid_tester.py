import sys

from networkx.generators.lattice import grid_2d_graph
import tools
import pysat.solvers as slv
import encoding as encoding
import encoding_lazy2 as lazy
import encoding2

g = grid_2d_graph(7, 7)
#tools.solve_grid(g, 3)

enc = encoding.TwinWidthEncoding(use_sb_static=True, use_sb_static_full=True)
enc = encoding2.TwinWidthEncoding2(g, cubic=2, sb_ord=False, sb_static=len(g.nodes) // 2, sb_static_full=True)
encl = lazy.TwinWidthEncoding2(g, cubic=True, sb_ord=False, sb_static=1, sb_red=False, use_sb_static_full=False)
encl.run(g, solver=slv.Cadical, start_bound=3)

