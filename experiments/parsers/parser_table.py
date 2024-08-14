import os
import sys

import parser_mappings as pm
from collections import defaultdict


class SolverResult:
    def __init__(self):
        self.solved = 0
        self.memory = 0
        self.time = 0
        self.bound_solved = 0
        self.counter = 0


target_set = 1
files = ["tww-pace", "tww-long", "tww-twlib", "tww-twlong"]
solver_results = defaultdict(lambda: [SolverResult() for _ in range(0, 4)])

for cfi, cf in enumerate(files):
    sat_solved = set()
    sat_bound_solved = set()

    with open(cf +".csv") as inp:
        for i, cl in enumerate(inp):
            fd = cl.strip().split(",")
            if i == 0:
                column_names = fd
            else:
                lb = int(fd[5]) if fd[5] != "" else 0
                ub = int(fd[6]) if fd[6] != "" else sys.maxsize
                last_time = None
                last_memory = None
                last_solved = False

                for ci, cn in enumerate(column_names):
                    cs = cn.split(" ")[0].split("-")[-1]
                    if cs == '':
                        continue
                    cs = pm.resolve_solver(cs)

                    if cn.endswith(" tww"):
                        solver_results[cs][cfi].counter += 1
                        last_solved = False
                        if fd[ci] != "":
                            if cs.startswith("sat"):
                                sat_solved.add(fd[0])
                            last_solved = True
                            solver_results[cs][cfi].solved += 1
                            solver_results[cs][cfi].time += last_time
                    elif cn.endswith(" best"):
                        if last_solved or lb == ub or (fd[ci] != "" and int(fd[ci]) == lb):
                            solver_results[cs][cfi].bound_solved += 1
                            if cs.startswith("sat"):
                                sat_bound_solved.add(fd[0])
                    elif cn.endswith(" Time"):
                        last_time = float(fd[ci])
                    elif cn.endswith(" Mem"):
                        solver_results[cs][cfi].memory += float(fd[ci])
    solver_results["sat"][cfi].solved = len(sat_solved)
    solver_results["sat"][cfi].bound_solved = len(sat_bound_solved)
    solver_results["sat"][cfi].counter = 1

solvers_ordered = sorted((x for x in solver_results.keys() if pm.order_solver(x) is not None),
                         key=lambda x: pm.order_solver(x))

for ck in solvers_ordered:
    sys.stdout.write(f"{pm.map_solver(ck)}")
    # sys.stdout.write(f"{ck}")

    cvi = 2 * target_set
    cvs = solver_results[ck][cvi]
    cvl = solver_results[ck][cvi+1]

    cvs_val = []
    cvl_val = []
    if cvs.solved == 0:
        cvs_val = ["-", "-", "-", "-"]
    else:
        cvs_val.append(str(cvs.solved))
        cvs_val.append(str(cvs.bound_solved))
        cvs_val.append(f"{round(cvs.time / cvs.solved / 60, 2):.2f}")
        cvs_val.append(f"{round(cvs.memory / cvs.counter / 1024 / 1024, 2):.2f}")

    if cvl.solved == 0:
        cvl_val = ["-", "-", "-", "-"]
    else:
        cvl_val.append(str(cvl.solved))
        cvl_val.append(str(cvl.bound_solved))
        cvl_val.append(f"{round(cvl.time / cvl.solved / 60, 2):.2f}")
        cvl_val.append(f"{round(cvl.memory / cvl.counter / 1024 / 1024, 2):.2f}")

        for cidx in range(0, len(cvs_val)):
            sys.stdout.write(f"&{cvs_val[cidx]}/&{cvl_val[cidx]}")

    sys.stdout.write("\\\\")
    sys.stdout.write(os.linesep)

