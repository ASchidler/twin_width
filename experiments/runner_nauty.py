import networkx
import os
from pysat.solvers import Cadical
import preprocessing
import encoding
import heuristic
import experiments.nauty_limits as nl
import subprocess
import sys
from collections import defaultdict
import threading
import tools
import time
from multiprocessing import Pool, Value, Array, Queue

solver = Cadical
pool_size = 4

if len(sys.argv) > 2:
    start_at = int(sys.argv[1])


def run_nauty(bnd):
    popen = subprocess.Popen(["bin/geng", "-c", str(bnd)], stdout=subprocess.PIPE, universal_newlines=True)
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
        return None, None

    enc = encoding.TwinWidthEncoding()
    if len(g.nodes) <= 2:
        result = 0
    else:
        ub = min(heuristic.get_ub(g), heuristic.get_ub2(g))
        result = enc.run(g, Cadical, ub, verbose=False, check=False)

    return result, g_str


c_smallest = -1
for v in nl.nauty_counts.values():
    for k2, v2 in v.items():
        if v2 > 0:
            c_smallest = max(c_smallest, k2)

for i in range(nl.finished+1, 10):
    totals = 0
    prime = 0
    counts = defaultdict(lambda: 0)

    with Pool(processes=pool_size) as pool:
        new_smallest = c_smallest

        for p_result, p_str in pool.imap_unordered(add_runner, run_nauty(i), chunksize=pool_size):
            totals += 1
            if p_result is not None:
                prime += 1
                counts[p_result] += 1
                if p_result > c_smallest:
                    if p_result not in nl.nauty_smallest:
                        nl.nauty_smallest[p_result] = []
                    nl.nauty_smallest[p_result].append(p_str)
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
