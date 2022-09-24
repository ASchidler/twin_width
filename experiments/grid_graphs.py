import os
import sys
from multiprocessing import Pool
from copy import copy

import networkx as nx
from networkx.generators.lattice import grid_2d_graph
from pysat.solvers import Cadical
from collections import defaultdict

import encoding
import encoding2
import encoding_lazy2
import heuristic
import queue

dimensions_x = 9
dimensions_y = 6

min_steps = 5
max_steps = 40

targets = []

g = grid_2d_graph(dimensions_x, dimensions_y)

results = {}

pool_size = 4

enc = encoding2.TwinWidthEncoding2(g)
gn = enc.remap_graph(g)

for cn1, cn2 in gn.edges:
    gn[cn1][cn2]["red"] = False

rmap = {y: x for (x, y) in enc.node_map.items()}

def generate_instances(cgg):
    q = []
    q.append((cgg, (1, 2), [], defaultdict(bool), set()))
    q.append((cgg, (1, 4), [], defaultdict(bool), set()))
    q.append((cgg, (1, 5), [], defaultdict(bool), set()))

    while q:
        cg, cm, cs, changed, at_limit = q.pop()
        cs = list(cs)
        cs.append(cm)
        changed = copy(changed)

        if len(cs) >= min_steps:
            yield cs
            continue

        cg = cg.copy()
        n1, n2 = cm
        n1nb = set(cg.neighbors(n1))
        n2nb = set(cg.neighbors(n2))
        n1nb.discard(n2)
        n2nb.discard(n1)
        nbdiff = n1nb ^ n2nb

        changed[n1] = True
        changed[n2] = True

        for cn in n1nb:
            if cg[n1][cn]["red"]:
                nbdiff.add(cn)

        for cn in nbdiff:
            changed[cn] = True
            if not cg.has_edge(n2, cn):
                cg.add_edge(n2, cn, red=True)
            else:
                cg[n2][cn]["red"] = True
        cg.remove_node(n1)

        for cn in cg.nodes:
            rdg = 0
            for cn2 in cg.neighbors(cn):
                if cg[cn][cn2]["red"]:
                    rdg += 1

        candidates = []

        for n1 in cg.nodes:
            if len(cs) == 1 and cs[0][0] == 1 and cs[0][1] == 5:
                rnode = rmap[n1]
                midx = dimensions_x // 2 + dimensions_x % 2 - 1
                midy = dimensions_y // 2 + dimensions_y % 2 - 1
                if rnode[0] > midx or rnode[1] > midy:
                    continue

            for n2 in cg.nodes:
                if n1 < n2:
                    n1nb = set(cg.neighbors(n1))
                    n2nb = set(cg.neighbors(n2))
                    n1nb.discard(n2)
                    n2nb.discard(n1)
                    nbdiff = n1nb ^ n2nb

                    if len(nbdiff) > 3:
                        continue

                    nonlex = any(x[0] > n1 for x in cs)
                    went_limit = False
                    if nonlex and not changed[n1] and not changed[n2] and all(changed[x] == False for x in nbdiff):
                        continue

                    b_rdg = 0
                    for cn in cg.neighbors(n2):
                        if cg[n2][cn]["red"]:
                            b_rdg += 1

                    rdg = 0
                    for cn in n2nb - nbdiff:
                        if cg[n2][cn]["red"]:
                            rdg += 1
                    for cn in n1nb - nbdiff:
                        if cg[n1][cn]["red"]:
                            rdg += 1

                    if rdg + len(nbdiff) > 3:
                        continue

                    if b_rdg < 3 and rdg + len(nbdiff) == 3:
                        went_limit = True

                    exceeded = False
                    for cn in nbdiff:
                        if (cg.has_edge(n1, cn) and cg[n1][cn]["red"]) or (cg.has_edge(n2, cn) and cg[n2][cn]["red"]):
                            continue

                        rdg = 0
                        for cnb in cg.neighbors(cn):
                            if cg[cn][cnb]["red"]:
                                rdg += 1

                        if rdg >= 3:
                            exceeded = True
                            break

                        if rdg == 2:
                            went_limit = True

                    if exceeded:
                        continue

                    if not went_limit:
                        at_limit_new = at_limit
                        nonlex = False
                        for myn1, _ in reversed(cs):
                            if myn1 in at_limit:
                                break
                            if myn1 > n1:
                                nonlex = True
                                break
                        if nonlex:
                            continue
                    else:
                        at_limit_new = set(at_limit)
                        at_limit_new.add(n1)
                    candidates.append((cg, (n1, n2), cs, changed, at_limit_new))

        if len(cs) >= min_steps:
            targets.append(cs)
            print(f"{len(targets)}")
        else:
            for ce in candidates:
                q.append(ce)


def compute_graph(args):
    ord = [x for x, y in args]
    mg = {x: y for x, y in args}

    cenc = encoding2.TwinWidthEncoding2(g, cubic=2, sb_ord=True, sb_static=2 * len(g.nodes) // 3, sb_static_full=True,
                                       is_grid=True)

    result = cenc.run(g, solver=Cadical, start_bound=3, i_od=ord, i_mg=mg, verbose=False, steps_limit=max_steps)

    if isinstance(result, int):
        return args
    else:
        return result

# for cx in generate_instances(gn):
#     print(cx)
# exit(0)

with open(f"dimensions_{dimensions_x}_{dimensions_y}.done", "a") as outp:
    with Pool(pool_size) as pool:
        for tww in pool.imap_unordered(compute_graph, generate_instances(gn)):
            if isinstance(tww, list):
                print(f"Tried {tww}")
                outp.write(f"{tww}"+os.linesep)
            else:
                print(f"Succeeded {tww}")
                outp.write(f"!Success {tww}")
                exit(0)



