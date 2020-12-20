import networkx
import os
from pysat.solvers import Cadical
import preprocessing
import encoding
import heuristic
import experiments.nauty_limits as nl
import subprocess
import sys

solver = Cadical

start_at = 2
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


for i in range(nl.finished+1, 33):
    for cgs in run_nauty(i):
        g = networkx.from_graph6_bytes(bytes(cgs.strip(), encoding="ascii"))
        preprocessing.twin_merge(g)
        enc = encoding.TwinWidthEncoding()
        if len(g.nodes) <= 2:
            result = 0
        else:
            ub = min(heuristic.get_ub(g), heuristic.get_ub2(g))
            result = enc.run(g, Cadical, ub, verbose=False, check=False)

        if result not in nl.nauty_limits:
            nl.nauty_limits[result] = i

    print(f"Finished {i}")
    with open("experiments/nauty_limits.py", "w") as nf:
        nf.write(f"finished = {i}" + os.linesep)
        nf.write("nauty_limits = {")
        for k, v in nl.nauty_limits.items():
            nf.write(f"{k}: {v},"+os.linesep)

        nf.seek(nf.tell() - 2)
        nf.write(os.linesep+ "}" +os.linesep)

