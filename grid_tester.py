from networkx.generators.lattice import grid_2d_graph
import tools
import pysat.solvers as slv
import encoding_lazy as encoding
import encoding_lazy2 as lazy
import encoding5

g = grid_2d_graph(6, 6)
#tools.solve_grid(g, 3)

enc = encoding.TwinWidthEncoding()
enc = encoding5.TwinWidthEncoding2(g)
encl = lazy.TwinWidthEncoding2(g)
encl.run(g, solver=slv.Glucose3, start_bound=3)

