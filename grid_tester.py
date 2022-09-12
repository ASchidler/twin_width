from networkx.generators.lattice import grid_2d_graph
import tools
import pysat.solvers as slv
import encoding as encoding
import encoding_lazy2 as lazy
import encoding2

g = grid_2d_graph(6, 6)
#tools.solve_grid(g, 3)

enc = encoding.TwinWidthEncoding()
enc = encoding5.TwinWidthEncoding2(g, cubic=1, sb_ord=True)
encl = lazy.TwinWidthEncoding2(g, cubic=False, sb_ord=False)
enc.run(g, solver=slv.Cadical, start_bound=3)

