import ctypes
import os
import sys
from multiprocessing import Pool, Process
from copy import copy
from pebble import ProcessPool
from itertools import combinations

import networkx as nx
from networkx.generators.lattice import grid_2d_graph
from pysat.solvers import Cadical
from collections import defaultdict
from concurrent.futures import TimeoutError

import encoding
import encoding2
import encoding_lazy2
import heuristic
import queue

min_steps = 3

dimensions_x, dimensions_y, max_steps = 7, 7, 35
# dimensions_x, dimensions_y, max_steps = 9, 6, 40

outp_file = f"dimensions_{dimensions_x}_{dimensions_y}.done"

timeout = 60

targets = []

g = grid_2d_graph(dimensions_x, dimensions_y)

results = {}

pool_size = 4

enc = encoding2.TwinWidthEncoding2(g)
gn = enc.remap_graph(g)

for cn1, cn2 in gn.edges:
    gn[cn1][cn2]["red"] = False

rmap = {y: x for (x, y) in enc.node_map.items()}

done = {}

def add_done(seq):
    c_d = done
    for cen in seq[:-1]:
        if cen not in c_d:
            c_d[cen] = {}
        c_d = c_d[cen]
        if c_d == 1:
            break

    if c_d != 1:
        c_d[seq[-1]] = 1

if os.path.exists(outp_file):
    with open(outp_file) as inp:
        for cline in inp:
            cline = cline.strip()
            if cline.startswith("["):
                cline = cline[1:-1]
                entries = cline.split(")")[:-1]
                entries = [x[x.index("(")+1:] for x in entries]
                entries2 = []
                for ce in entries:
                    fields = ce.split(",")
                    x, y = int(fields[0].strip()), int(fields[1].strip())
                    entries2.append((x, y))
                add_done(entries2)

def generate_instances(cgg, c_minsteps, created):
    q = []
    q.append((cgg, (1, 2), [], defaultdict(bool), set(), []))
    q.append((cgg, (1, 4), [], defaultdict(bool), set(), []))
    q.append((cgg, (1, 5), [], defaultdict(bool), set(), []))

    while q:
        cg, cm, cs, changed, at_limit, decreased = q.pop()
        cs = list(cs)
        decreased = list(decreased)
        new_decreased = set()
        cs.append(cm)
        changed = copy(changed)

        if len(cs) >= c_minsteps:
            created.append(cs)
            yield cs
            continue

        c_done = done
        for centry in cs:
            if c_done == 1:
                break
            if centry in c_done:
                c_done = c_done[centry]
            else:
                c_done = None
                break

        cg = cg.copy()
        n1, n2 = cm
        n1nb = set(cg.neighbors(n1))
        n2nb = set(cg.neighbors(n2))
        n1nb.discard(n2)
        n2nb.discard(n1)
        nbdiff = n1nb ^ n2nb

        changed[n1] = True
        changed[n2] = True

        if cg.has_edge(n1, n2) and cg[n1][n2]["red"]:
            if all(cg.has_edge(n2, x) and cg[n2][x]["red"] for x in nbdiff):
                new_decreased.add(n2)

        for cn in n1nb:
            if cg[n1][cn]["red"]:
                nbdiff.add(cn)

        for cn in nbdiff:
            if cg.has_edge(n1, cn) and cg.has_edge(n2, cn) and cg[n2][cn]["red"] and cg[n1][cn]["red"]:
                new_decreased.add(cn)

            changed[cn] = True
            if not cg.has_edge(n2, cn):
                cg.add_edge(n2, cn, red=True)
            else:
                cg[n2][cn]["red"] = True
        cg.remove_node(n1)
        decreased.append(new_decreased)

        for cn in cg.nodes:
            rdg = 0
            for cn2 in cg.neighbors(cn):
                if cg[cn][cn2]["red"]:
                    rdg += 1

        cnodes = sorted(cg.nodes, reverse=True)
        for ci, n1 in enumerate(cnodes):
            if len(cs) == 1 and cs[0][0] == 1 and cs[0][1] == 5:
                rnode = rmap[n1]
                midx = dimensions_x // 2 + dimensions_x % 2 - 1
                midy = dimensions_y // 2 + dimensions_y % 2 - 1
                if rnode[0] > midx or rnode[1] > midy:
                    continue

            for n2 in cnodes[:ci]:
                if n1 < n2:
                    # if c_done is not None and (n1, n2) in c_done and c_done[(n1, n2)] == 1:
                    #     continue

                    n1nb = set(cg.neighbors(n1))
                    n2nb = set(cg.neighbors(n2))
                    n1nb.discard(n2)
                    n2nb.discard(n1)
                    nbdiff = n1nb ^ n2nb

                    if len(nbdiff) > 3 or (c_done is not None and (n1, n2) in c_done and c_done[(n1, n2)] == 1):
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
                    went_limit = []
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
                            went_limit.append(cn)

                    if exceeded:
                        continue

                    if went_limit:
                        at_limit_new = set(at_limit)
                        at_limit_new.add(n1)
                    else:
                        at_limit_new = at_limit

                    nonlex = False
                    for i, (myn1, _) in enumerate(reversed(cs)):
                        if any(x in decreased[-i] for x in went_limit):
                            break

                        if myn1 > n1:
                            nonlex = True
                            break
                    if nonlex:
                        continue

                    indices = []
                    for i, past_entry in enumerate(cs):
                        if i > 0 and past_entry[1] == n1:
                            indices.append(i)

                    if len(indices) > 0:
                        for cnum in range(1, len(indices)+1):
                            for cc in combinations(indices, cnum):
                                ncs = list(cs)
                                for cidx in cc:
                                    ncs[cidx] = (ncs[cidx][0], n2)
                                ncs.append((n1, n2))
                                add_done(ncs)

                    q.append((cg, (n1, n2), cs, changed, at_limit_new, decreased))


def compute_graph(args):
    ord = [x for x, y in args]
    mg = {x: y for x, y in args}

    cenc = encoding2.TwinWidthEncoding2(g, cubic=2, sb_ord=True, sb_static=2 * len(g.nodes) // 3, sb_static_full=True,
                                       is_grid=True)

    # cenc = encoding.TwinWidthEncoding(use_sb_static=True, use_sb_static_full=True)
    result = cenc.run(g, solver=Cadical, start_bound=3, i_od=ord, i_mg=mg, verbose=False, steps_limit=max_steps)

    if isinstance(result, int) or len(result) == 2:
        return args
    else:
        return result

if __name__ == '__main__':
    # cnt = 1
    # for cx in generate_instances(gn, min_steps, []):
    #     print(f"{cnt} {cx}")
    #     cnt += 1
    # exit(0)

    with open(outp_file, "a") as outp:
        with open(outp_file+".to", "a") as outp_to:
            # with Pool(pool_size) as pool:
            #     for tww in pool.imap_unordered(compute_graph, generate_instances(gn), chunksize=100):
            # if isinstance(tww, list):
            #     print(f"Tried {tww}")
            #     outp.write(f"{tww}" + os.linesep)
            #     outp.flush()
            # else:
            #     print(f"Succeeded {tww}")
            #     outp.write(f"! {tww}")
            #     outp.flush()
            #     exit(0)

            for c_steps in range(min_steps, max_steps):
                c_created = []
                cnt = 0
                with ProcessPool(max_workers=pool_size) as pool:
                    future = pool.map(compute_graph, generate_instances(gn, c_steps, c_created), chunksize=1, timeout=timeout)
                    it = future.result()
                    while True:
                        try:
                            tww = next(it)
                            if isinstance(tww, list):
                                print(f"Tried {tww}")
                                outp.write(f"{tww}" + os.linesep)
                                add_done(tww)
                                outp.flush()
                            else:
                                print(f"Succeeded {tww}")
                                outp.write(f"! {tww}")
                                outp.flush()
                                exit(0)
                        except StopIteration as ee:
                            break
                        except TimeoutError as error:
                            print(f"Timeout {c_created[cnt]}")
                            outp_to.write(f"{c_created[cnt]}{os.linesep}")
                            outp_to.flush()
                        finally:
                            cnt += 1
                print(f"Finished {c_steps}")


