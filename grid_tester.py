import sys

from networkx.generators.lattice import grid_2d_graph
import tools
import pysat.solvers as slv
import encoding as encoding
import encoding_lazy2 as lazy
import encoding2

g = grid_2d_graph(7, 7)
#tools.solve_grid(g, 3)

enc = encoding.TwinWidthEncoding()
enc = encoding2.TwinWidthEncoding2(g, cubic=2, sb_ord=True, sb_static=sys.maxsize)
encl = lazy.TwinWidthEncoding2(g, cubic=True, sb_ord=True)
encl.run(g, solver=slv.Cadical, start_bound=3)

