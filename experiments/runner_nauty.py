import os
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool

import networkx
from pysat.solvers import Cadical195

import encoding
import experiments.nauty_limits as nl
import heuristic
import tools

solver = Cadical195
pool_size = 4

if len(sys.argv) > 2:
    start_at = int(sys.argv[1])


def run_nauty(bnd):
    m = bnd * (bnd-1) // 4 if bnd % 4 <= 1 else (bnd * (bnd-1)) // 4 + 1
    popen = subprocess.Popen(["bin/geng", "-c", str(bnd), f"0:{m}"], stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, ["bin/geng", "-c", bnd])


def add_runner(g_str):
    g = networkx.from_graph6_bytes(bytes(g_str.strip(), encoding="ascii"))
    modules = tools.find_modules(g)
    if len(modules) < len(g.nodes):
        return None, None, len(g.edges)

    enc = encoding.TwinWidthEncoding()
    if len(g.nodes) <= 2:
        result = 0
    else:
        ub = heuristic.get_ub3(g)
        result = enc.run(g, Cadical195, ub, verbose=False, check=False)

    return result, g_str, len(g.edges)


c_smallest = -1
for v in nl.nauty_counts.values():
    for k2, v2 in v.items():
        if v2 > 0:
            c_smallest = max(c_smallest, k2)

for i in range(nl.finished+1, 12):
    totals = 0
    prime = 0
    counts = defaultdict(lambda: 0)

    with Pool(processes=pool_size) as pool:
        new_smallest = c_smallest
        m = i * (i - 1) // 4 if i % 4 <= 1 else (i * (i - 1)) // 4 + 1

        for p_result, p_str, p_edges in pool.imap_unordered(add_runner, run_nauty(i), chunksize=pool_size):
            # This will be off, as we don't know how many complementary graphs will be connected?
            totals += 2 if (i % 4 <= 1 or p_edges < m) and p_result is not None else 1

            if p_result is not None:
                factor = 2
                if p_edges == m:
                    if i % 4 <= 1:
                        factor = 1
                    else:
                        factor = 0

                # Self complementary / does not have a second one
                prime += 1 * factor
                counts[p_result] += 1 * factor
                if p_result > c_smallest:
                    if p_result not in nl.nauty_smallest:
                        nl.nauty_smallest[p_result] = []
                    nl.nauty_smallest[p_result].append(p_str)
                    print(f"{p_result} {p_str}")
                    new_smallest = max(new_smallest, p_result)

        pool.close()
        pool.join()

    c_smallest = new_smallest
    nl.nauty_total[i] = totals
    nl.nauty_prime[i] = prime
    nl.nauty_counts[i] = {}
    for ci in range(0, i):
        nl.nauty_counts[i][ci] = counts[ci]

    print(f"Finished {i}")
    line = f"{i},\t{nl.nauty_total[i]},\t{nl.nauty_prime[i]}"
    for k, c in nl.nauty_counts[i].items():
        line += f",\t{k:} {c}"
    print(line)

    with open("experiments/nauty_limits.py", "w") as nf:
        def print_dict(cd, sep_lines=True):
            for k, v in cd.items():
                if isinstance(v, defaultdict) or isinstance(v, dict):
                    nf.write(f"{k}: " + "{")
                    print_dict(v, sep_lines=False)
                    nf.write("}, " + (os.linesep if sep_lines else " "))
                elif isinstance(v, list):
                    nf.write(f"{k}: [")
                    for entry in v:
                        if isinstance(entry, str):
                            nf.write(f"'{entry.strip()}',")
                        else:
                            nf.write(f"{entry},")
                    nf.seek(nf.tell() - 1)
                    nf.write("],"+os.linesep)
                else:
                    if isinstance(v, str):
                        nf.write(f"{k}: '{v}'," + (os.linesep if sep_lines else " "))
                    else:
                        nf.write(f"{k}: {v}," + (os.linesep if sep_lines else " "))

            if len(cd) > 0:
                nf.seek(nf.tell() - 1 - (len(os.linesep) if sep_lines else 1))

        nf.write(f"finished = {i}" + os.linesep)
        nf.write(f"nauty_smallest = " + "{" + os.linesep)
        print_dict(nl.nauty_smallest)
        nf.write(os.linesep + "}" + os.linesep)
        nf.write(f"nauty_counts = " + "{" + os.linesep)
        print_dict(nl.nauty_counts)
        nf.write(os.linesep + "}" + os.linesep)
        nf.write(f"nauty_prime = " + "{" + os.linesep)
        print_dict(nl.nauty_prime)
        nf.write(os.linesep + "}" + os.linesep)
        nf.write(f"nauty_total = " + "{" + os.linesep)
        print_dict(nl.nauty_total)
        nf.write(os.linesep + "}" + os.linesep)
