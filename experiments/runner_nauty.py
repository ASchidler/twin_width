import networkx
import os
from pysat.solvers import Cadical
import preprocessing
import encoding
import heuristic
import experiments.nauty_limits as nl
import subprocess
import sys
from multiprocessing.pool import ThreadPool
from collections import defaultdict
import threading

solver = Cadical
pool_size = 4
lock1 = threading.Lock()
lock2 = threading.Lock()

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


def add_result(g6, bnd):
    lock1.acquire()
    nl.nauty_total[bnd] += 1
    lock1.release()

    g = networkx.from_graph6_bytes(bytes(g6.strip(), encoding="ascii"))
    before = len(g.nodes)
    preprocessing.twin_merge(g)
    if before > len(g.nodes):
        return

    enc = encoding.TwinWidthEncoding()
    if len(g.nodes) <= 2:
        result = 0
    else:
        ub = min(heuristic.get_ub(g), heuristic.get_ub2(g))
        result = enc.run(g, Cadical, ub, verbose=False, check=False)

    # Since we parallelize only calls with the same bound and any single operation is threadsafe, this is ok
    if result not in nl.nauty_smallest:
        nl.nauty_smallest[result] = f"{g6.strip()}"

    lock2.acquire()
    nl.nauty_prime[bnd] += 1
    nl.nauty_counts[bnd][result] += 1
    lock2.release()


for i in range(nl.finished+1, 33):
    pool = ThreadPool(processes=4)
    nl.nauty_total[i] = 0
    nl.nauty_prime[i] = 0
    nl.nauty_counts[i] = defaultdict(lambda: 0)
    pool.map(lambda x: add_result(x, i), run_nauty(i))
    pool.close()
    pool.join()

    print(f"Finished {i}")

    with open("experiments/nauty_limits.py", "w") as nf:
        def print_dict(cd, sep_lines=True):
            for k, v in cd.items():
                if isinstance(v, defaultdict) or isinstance(v, dict):
                    nf.write(f"{k}: " + "{")
                    print_dict(v, sep_lines=False)
                    nf.write("}, " + (os.linesep if sep_lines else " "))
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
