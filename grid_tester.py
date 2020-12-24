from networkx.generators.lattice import grid_2d_graph
import tools
import pysat.solvers as slv
import encoding

g = grid_2d_graph(7, 6)
#tools.solve_grid(g, 3)

enc = encoding.TwinWidthEncoding()
enc.run(g, solver=slv.Cadical, start_bound=3)

