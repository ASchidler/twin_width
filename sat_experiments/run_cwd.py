import io
import os
import sys
from threading import Timer

import parser
import subprocess
from pysat.solvers import Cadical153, Glucose4

instance = sys.argv[1]
print(os.path.split(instance)[-1])

signed = True
timeout = 60

g = parser.parse_cnf(instance)
mapping = {n: i+1 for i, n in enumerate(g.nodes)}

fname = os.path.join(sys.argv[2],f"cwd_{os.getpid()}.arc")
with open(fname, "w") as outp:
    if signed:
        outp.write(f"p arc {len(g.nodes)} {len(g.edges)}{os.linesep}")
    else:
        outp.write(f"p edge {len(g.nodes)} {len(g.edges)}{os.linesep}")

    for u, v in g.edges:
        if signed:
            outp.write(f"a {mapping[u]} {mapping[v]}{os.linesep}")
        else:
            outp.write(f"e {mapping[u]} {mapping[v]}{os.linesep}")

c_bound = 2
last_unsat = 0
last_sat = sys.maxsize

while c_bound > last_unsat:
    if signed:
        proc = subprocess.Popen(["bin/dcwd", fname, str(c_bound)], stdout=subprocess.PIPE)
    else:
        proc = subprocess.Popen(["bin/cwd", fname, str(c_bound)], stdout=subprocess.PIPE)

    with Glucose4() as slv:
        for line in io.TextIOWrapper(proc.stdout):
            if line.startswith("c") or line.startswith("%") or line.startswith("p"):
                continue
            vars = [int(x) for x in line.strip().split(" ")[:-1]]
            slv.add_clause(vars)

        print(f"Running {c_bound}")
        print(f"{slv.nof_vars()}/{slv.nof_clauses()}")


        def interrupt(s):
            s.interrupt()

        if c_bound != last_sat - 1:
            timer = Timer(timeout, interrupt, [slv])
            timer.start()

        result = slv.solve_limited(expect_interrupt=True)
        if result is True:
            print(f"SAT Width: {c_bound}")
            sys.stdout.flush()
            last_sat = c_bound
            c_bound -= 1
        elif result is False:
            print(f"UNSAT Width: {c_bound}")
            sys.stdout.flush()
            last_unsat = c_bound
            c_bound += 1
        else:
            print(f"Timeout: {c_bound}")
            c_bound += 1
            timeout *= 2

print(f"Finished successful, Clique Width: {last_sat}")
