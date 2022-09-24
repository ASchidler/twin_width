import sys

from networkx.generators.lattice import grid_2d_graph
import tools
import pysat.solvers as slv
import encoding as encoding
import encoding_lazy2 as lazy
import encoding2

# g, steps_limit = grid_2d_graph(7, 7), 35
g, steps_limit = grid_2d_graph(9, 6), 40
# g, steps_limit = grid_2d_graph(6, 6), 33

#enc = encoding.TwinWidthEncoding(use_sb_static=True, use_sb_static_full=True)
enc = encoding2.TwinWidthEncoding2(g, cubic=2, sb_ord=False, sb_static=2 * len(g.nodes)//3, sb_static_full=True, is_grid=True)
enc.run(g, solver=slv.Cadical, start_bound=3)

# encl = lazy.TwinWidthEncoding2(g, cubic=True, sb_ord=True, sb_static=len(g.nodes)//2, sb_red=False, use_sb_static_full=True)
# encl.run(g, solver=slv.Cadical, start_bound=3)


