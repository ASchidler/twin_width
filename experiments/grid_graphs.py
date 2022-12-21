import os
import sys
from collections import defaultdict
from concurrent.futures import TimeoutError
from copy import copy

from networkx.generators.lattice import grid_2d_graph
from pebble import ProcessPool
from pysat.solvers import Cadical

import encoding
import encoding2
import encoding3
import encoding_lazy2

min_steps = 4
timeout = 30
use_seen = False
verify = False
pool_size = 3
enc_count = 1
consolidate = False

if len(sys.argv) > 1 and int(sys.argv[1]) > 0:
    dimensions_x, dimensions_y, max_steps = 7, 7, 35
else:
    dimensions_x, dimensions_y, max_steps = 9, 6, 44

#outp_file = f"/media/asc/4ED79AA0509E44AA/dimensions_{dimensions_x}_{dimensions_y}.done"
outp_file = f"dimensions_{dimensions_x}_{dimensions_y}.done"

targets = []

g = grid_2d_graph(dimensions_x, dimensions_y)

results = {}

enc = encoding2.TwinWidthEncoding2(g)
gn = enc.remap_graph(g)

for cn1, cn2 in gn.edges:
    gn[cn1][cn2]["red"] = False

rmap = {y: x for (x, y) in enc.node_map.items()}

done = {}
verified = set()
seen = set()

children = defaultdict(int)

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


def check_seen(c_part):
    nk = []
    for ck, cv in sorted(c_part.items()):
        nk.append(tuple(cv))

    nk = tuple(nk)

    if nk in seen:
        return True
    else:
        seen.add(nk)

    return False


target_files = [outp_file] if not verify else [outp_file, outp_file + ".verified"]

# Load previous state of done partial assignments
for i, c_output in enumerate(target_files):
    if os.path.exists(c_output):
        with open(c_output) as inp:
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
                    if i == 0:
                        add_done(entries2)

                        if use_seen:
                            new_parts = {}

                            for (n1, n2) in entries2:
                                if n2 not in new_parts:
                                    new_parts[n2] = [n2]

                                if n1 in new_parts:
                                    new_parts[n2].extend(new_parts[n1])
                                    new_parts.pop(n1)
                                else:
                                    new_parts[n2].append(n1)
                            for cv in new_parts.values():
                                cv.sort()

                            check_seen(new_parts)
                    else:
                        verified.add(tuple(entries2))

# Consolidate file, remove subsumed partial assignments
if consolidate:
    q = [([k], v) for k, v in done.items()]
    entries = []

    while q:
        c_entry, c_entry_v = q.pop()
        if c_entry_v == 1:
            if tuple(c_entry) not in verified:
                entries.append(c_entry)
        else:
            for ck, cv in c_entry_v.items():
                nl = list(c_entry)
                nl.append(ck)
                q.append((nl, cv))

    entries.sort(key=lambda x: (len(x), x))
    with open(outp_file+".consolidated", "a") as outp:
        for c_entry in entries:
            outp.write(f"{c_entry}{os.linesep}")

    exit(0)

def generate_instances(cgg, c_minsteps, created):
    with open(outp_file, "a") as outp:
        q = []

        for ci in range(1, len(cgg.nodes)+1):
            for cj in range(ci+1, len(cgg.nodes) + 1):
                if (ci, cj) not in done or done[(ci, cj)] != 1:
                    q.append((cgg, (ci, cj), [], defaultdict(bool), {cj: [ci, cj]}, defaultdict(list), defaultdict(list), set()))
        # q.append((cgg, (1, 2), [], defaultdict(bool), {2: [1, 2]}, defaultdict(list), defaultdict(list), set()))
        # q.append((cgg, (1, 4), [], defaultdict(bool), {4: [1, 4]}, defaultdict(list), defaultdict(list), set()))
        # q.append((cgg, (1, 5), [], defaultdict(bool), {5: [1, 5]}, defaultdict(list), defaultdict(list), set()))

        while q:
            cg, cm, cs, changed, parts, transferred, lowered, justified = q.pop()
            cs = list(cs)
            cs.append(cm)

            if len(cs) >= c_minsteps:
                created.append(cs)
                yield cs
                continue

            changed = copy(changed)
            transferred = defaultdict(list, [(x, list(y)) for x, y in transferred.items()])
            lowered = defaultdict(list, [(x, list(y)) for x, y in lowered.items()])
            justified = set(justified)

            # Load correct part of the done entries
            c_done = done
            for centry in cs:
                if c_done == 1:
                    break
                if centry in c_done:
                    c_done = c_done[centry]
                else:
                    c_done = None
                    break

            # Perform contraction
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

            # Check which vertices decreased in degree...
            if cg.has_edge(n1, n2) and cg[n1][n2]["red"]:
                lowered[n2].extend(transferred[(n1, n2)])
                lowered[n2].append(n1)

            for cn in nbdiff:
                if cg.has_edge(n1, cn) and cg.has_edge(n2, cn) and cg[n2][cn]["red"] and cg[n1][cn]["red"]:
                    lowered[cn].append(transferred[(min(n1, cn), max(n1, cn))])
                    lowered[cn].append(transferred[(min(n2, cn), max(n2, cn))])
                    lowered[cn].append(n1)

                if cg.has_edge(n1, cn) and cg[n1][cn]["red"]:
                    transferred[(min(n1, cn), max(n1, cn))].append(n1)

                changed[cn] = True
                if not cg.has_edge(n2, cn):
                    cg.add_edge(n2, cn, red=True)
                else:
                    cg[n2][cn]["red"] = True

            cg.remove_node(n1)

            cnodes = sorted(cg.nodes, reverse=True)
            for ci, n1 in enumerate(cnodes):
                # diagonally = True
                # for xn1, xn2 in cs:
                #     xn1 = rmap[xn1]
                #     xn2 = rmap[xn2]
                #     if xn1[0] != xn1[1] or xn2[0] != xn2[1]:
                #         diagonally = False
                #         break

                if len(cs) == 1 and cs[0][0] == 1 and cs[0][1] == 5:
                    rnode = rmap[n1]
                    midx = dimensions_x // 2 + dimensions_x % 2 - 1
                    midy = dimensions_y // 2 + dimensions_y % 2 - 1
                    if rnode[0] > midx or rnode[1] > midy:
                        continue

                for n2 in cnodes[:ci]:
                    if n1 < n2:
                        if c_done is not None and (n1, n2) in c_done and c_done[(n1, n2)] == 1:
                            continue

                        n1nb = set(cg.neighbors(n1))
                        n2nb = set(cg.neighbors(n2))
                        n1nb.discard(n2)
                        n2nb.discard(n1)
                        nbdiff = n1nb ^ n2nb

                        if len(nbdiff) > 3:
                            continue

                        # Enforce lex order whenever the contraction does not influence any previously influenced vertex
                        nonlex = any(x[0] > n1 for x in cs)
                        if nonlex and not changed[n1] and not changed[n2] and all(changed[x] == False for x in nbdiff):
                            continue

                        # Check if red degree of n2 would be exceeded
                        rdg = 0
                        for cn in n2nb - nbdiff:
                            if cg[n2][cn]["red"]:
                                rdg += 1
                        for cn in n1nb & n2nb:
                            if cg[n1][cn]["red"] and not cg[n2][cn]["red"]:
                                rdg += 1

                        if rdg + len(nbdiff) > 3:
                            continue

                        went_limit = []
                        ordg = 0
                        for cn in n2nb:
                            if cg[n2][cn]["red"]:
                                ordg += 1
                        if ordg < 3 and rdg + len(nbdiff) == 3:
                            went_limit.append(n2)

                        # Check if any adjacent vertex would be exceeded
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
                                went_limit.append(cn)

                        if exceeded:
                            continue

                        # Check if non-lexicographic order is justified
                        nonlex = False

                        for myn1, _ in reversed(cs):
                            if myn1 > n1 and myn1 not in justified:
                                if any(myn1 in lowered[x] for x in went_limit):
                                    justified.add(myn1)
                                else:
                                    nonlex = True
                                    break

                        if nonlex:
                            continue

                        new_parts = parts
                        if use_seen:
                            new_parts = {x: list(y) for x, y in parts.items()}
                            if n2 not in new_parts:
                                new_parts[n2] = [n2]

                            if n1 in new_parts:
                                new_parts[n2].extend(new_parts[n1])
                                new_parts.pop(n1)
                            else:
                                new_parts[n2].append(n1)
                            new_parts[n2].sort()

                            # If we already have a queue entry with a low enough tww that has the same partitioning, skip
                            if check_seen(new_parts):
                                continue
                        children[tuple(cs)] += 1
                        q.append((cg, (n1, n2), cs, changed, new_parts, transferred, lowered, justified))

            if tuple(cs) not in children:
                cx = tuple(cs[:-1])
                while True:
                    children[cx] -= 1
                    if children[cx] == 0:
                        outp.write(f"{list(cx)}{os.linesep}")
                        children.pop(cx)
                        print(f"Removed {cx}")
                        cx = tuple(cx[:-1])
                    else:
                        break

# if enc == 0:
#     cenc = encoding2.TwinWidthEncoding2(g, cubic=2, sb_ord=True, sb_static=1, sb_static_full=False,
#                                         is_grid=False)
#     cenc = encoding3.TwinWidthEncoding2(g, cubic=2, sb_ord=False, sb_static=1)
# elif enc == 1:
#     cenc = encoding2.TwinWidthEncoding2(g, cubic=2, sb_ord=False, sb_static=0, sb_static_full=False,
#                                         is_grid=False)
# else:
#     cenc = encoding.TwinWidthEncoding(use_sb_static=True, use_sb_static_full=True)


solver = None
def compute_graph(args):
    global solver
    ord = [x for x, y in args]
    mg = {x: y for x, y in args}
    if solver is None:
        solver = encoding3.TwinWidthEncoding2(g, cubic=2, sb_ord=True, sb_static=sys.maxsize, sb_static_full=True,
                                              break_g_symmetry=True)
        solver.start_incremental(g, solver=Cadical, start_bound=3, verbose=False, steps_limit=max_steps)

    result = solver.run_incremental(ord, mg, False)

    # result = cenc.run(g, solver=Cadical, start_bound=3, i_od=ord, i_mg=mg, verbose=True, steps_limit=max_steps)

    if isinstance(result, int) or len(result) == 2:
        return args
    else:
        return result


def generate_verification():
    q = [([k], v) for k, v in done.items()]

    while q:
        c_entry, c_entry_v = q.pop()
        if c_entry_v == 1:
            if tuple(c_entry) not in verified:
                yield c_entry
        else:
            for ck, cv in c_entry_v.items():
                nl = list(c_entry)
                nl.append(ck)
                q.append((nl, cv))


if __name__ == '__main__':
    # for csteps in range(min_steps, max_steps):
    #     cnt = 1
    #     for cx in generate_instances(gn, csteps, []) if not verify else generate_verification():
    #         print(f"{csteps}:{cnt} {cx}")
    #         cnt += 1
    #     exit(0)

    with open(outp_file if not verify else outp_file+".verified", "a") as outp:
        with open(outp_file+".to", "a") as outp_to:
            for c_steps in range(min_steps, max_steps):
                c_created = []
                cnt = 0
                with ProcessPool(max_workers=pool_size) as pool:
                    if not verify:
                        future = pool.map(compute_graph, generate_instances(gn, c_steps, c_created), chunksize=1, timeout=timeout)
                    else:
                        future = pool.map(compute_graph, ((x, i) for x in generate_verification() for i in range(0, enc_count)), chunksize=1, timeout=timeout)

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
                                outp.write(f"! {tww}" + os.linesep)
                                outp.flush()
                                exit(0)
                        except StopIteration as ee:
                            break
                        except TimeoutError as error:
                            if not verify:
                                print(f"Timeout {c_created[cnt // enc_count]}")
                                outp_to.write(f"{c_created[cnt // enc_count]}{os.linesep}")
                                outp_to.flush()
                            else:
                                print("Timeout")
                        finally:
                            cnt += 1
                print(f"Finished {c_steps}")


