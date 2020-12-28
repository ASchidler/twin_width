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
c_smallest = Value("i")
c_smallest.value = -1
prime = Value("i")
total = Value("i")
counts = None
queue = Queue()

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


def add_runner(c_str):
    total.value += 1

    g = networkx.from_graph6_bytes(bytes(c_str.strip(), encoding="ascii"))
    modules = tools.find_modules(g)
    if len(modules) < len(g.nodes):
        return

    enc = encoding.TwinWidthEncoding()
    if len(g.nodes) <= 2:
        result = 0
    else:
        ub = min(heuristic.get_ub(g), heuristic.get_ub2(g))
        result = enc.run(g, Cadical, ub, verbose=False, check=False)

    prime.value += 1
    counts[result] += 1
    if result > c_smallest.value:
        queue.put((result, c_str))


for v in nl.nauty_counts.values():
    for k2, v2 in v.items():
        if v2 > 0:
            c_smallest.value = max(c_smallest.value, k2)

for i in range(nl.finished+1, 10):
    counts = Array("i", [0 for _ in range(0, i)])
    prime.value = 0
    total.value = 0

    # TODO: Smallest?
    with Pool(processes=pool_size) as pool:
        pool.map(add_runner, run_nauty(i))
        done = False

        last = 0
        while last < total.value:
            time.sleep(5)
            last = total.value
            print(f"{last} entries done")

        pool.close()
        pool.join()
        nl.nauty_total[i] = total.value
        nl.nauty_prime[i] = prime.value
        nl.nauty_counts[i] = {}
        for ci in range(0, i):
            nl.nauty_counts[i][ci] = counts[ci]
        while not queue.empty():
            c_result, c_str = queue.get()
            if c_result not in nl.nauty_smallest:
                nl.nauty_smallest[c_result] = []
            nl.nauty_smallest[c_result].append(c_str)
            c_smallest.value = max(c_smallest.value, c_result)

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
