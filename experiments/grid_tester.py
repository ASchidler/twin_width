import sys

from networkx.generators.lattice import grid_2d_graph
import tools
import pysat.solvers as slv
import encoding as encoding
import encoding_lazy2 as lazy
import encoding2
import encoding3

# g, steps_limit = grid_2d_graph(7, 7), 35
g, steps_limit = grid_2d_graph(9, 6), 44 # 46 (43)
# g, steps_limit = grid_2d_graph(8, 6), 50
    
print(len(g.nodes))

# enc = encoding.TwinWidthEncoding(use_sb_static=False, use_sb_static_full=False)
# enc = encoding2.TwinWidthEncoding2(g, cubic=2, sb_ord=False, sb_static=0, sb_static_full=False, is_grid=False)

# enc.run(g, solver=slv.Cadical, start_bound=2, steps_limit=steps_limit)

enc = encoding3.TwinWidthEncoding2(g, cubic=2, sb_static=sys.maxsize, sb_ord=True, sb_static_full=True, sb_static_diff=False, is_grid=True)
result = enc.run(g, slv.Cadical, 3, steps_limit=steps_limit)


# encl = lazy.TwinWidthEncoding2(g, cubic=True, sb_ord=True, sb_static=len(g.nodes)//2, sb_red=False, use_sb_static_full=True)
# encl.run(g, solver=slv.Cadical, start_bound=3)


