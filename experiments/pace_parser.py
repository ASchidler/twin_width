import sys
import tarfile as tf
import os
from collections import defaultdict

input = "logs/tww-pace.tar.bz2"
instance_path = "/Users/andre/Downloads/exact-private/"


def xstr(s):
    if s is None:
        return ''
    return str(s)


class ExperimentResult:
    def __init__(self):
        self.tww = None
        self.runtime = None
        self.memory = None
        self.best_tww = None


class InstanceResult:
    def __init__(self):
        self.vertices: int | None = None
        self.edges: int | None = None
        self.max_component_vertices: int | None = None
        self.max_component_edges: int | None = None
        self.twin_width: int | None = None
        self.lb = 0
        self.ub = None


experiments = set()
results = defaultdict(lambda: defaultdict(ExperimentResult))
instance_results = defaultdict(InstanceResult)

with tf.open(input) as ctf:
    for clf in ctf:
        name_fields = clf.name.split(".")
        experiments.add(name_fields[0])
        instance = int(name_fields[-1])
        cetf = ctf.extractfile(clf)

        instance_object = instance_results[instance]

        if name_fields[1].startswith("v"):
            for i, cln in enumerate(cetf):
                cln = cln.decode('ascii').strip()
                if not cln.strip().startswith("#"):
                    if cln.strip().startswith("WCTIME"):
                        results[instance][name_fields[0]].runtime = float(cln.split("=")[1].strip())
                    elif cln.strip().startswith("MAXVM"):
                        results[instance][name_fields[0]].memory = float(cln.split("=")[1].strip())
        elif name_fields[1].startswith("s"):
            only_line = True
            only_field = None
            finished_seen = False
            time_seen = False

            o_lb = 0
            o_ub = None
            c_best = None
            o_best = None
            g_ub = None
            found_ub = True
            max_width = None

            lines = []

            for i, cln in enumerate(cetf):
                cln = cln.decode('ascii').strip()
                lines.append(cln)
                if only_field is None and only_line:
                    try:
                        only_field = int(cln.strip())
                    except ValueError:
                        only_line = False
                else:
                    only_line = False

                if cln.strip().startswith("Final Result:"):
                    results[instance][name_fields[0]].tww = int(cln.split(":")[1].strip())
                    if instance_object.twin_width is None:
                        instance_object.twin_width = results[instance][name_fields[0]].tww
                    else:
                        assert results[instance][name_fields[0]].tww == instance_object.twin_width
                elif cln.strip().find("Edges") > -1 and cln.strip().find("Vertices") > -1:
                    fields = cln.strip().split(" ")
                    if fields[1] == "Vertices" and fields[3] == "Edges":
                        instance_object.vertices = int(fields[0])
                        instance_object.edges = int(fields[2])
                elif cln.strip() == "Finished":
                    finished_seen = True
                elif finished_seen and cln.strip().startswith("("):
                    cln = cln[1:-1]
                    # Dirty fix for wrong output
                    if instance == 200:
                        cln = cln[cln.index(",")+1:cln.index("]")]
                        cfs = cln.split(",")
                        if len(cfs) == 19999:
                            results[instance][name_fields[0]].tww = 0
                    else:
                        results[instance][name_fields[0]].tww = int(cln.split(",")[0])
                elif cln.strip().startswith("Found ") and not cln.strip().startswith("Found subgraph") and len(cln.split(" ")) == 2:
                    new_sol = int(cln.strip().split(" ")[1])
                    c_best = new_sol
                elif cln.strip().startswith("Improved bound to "):
                    new_sol = int(cln.strip().split(" ")[-1])
                    c_best = new_sol
                elif cln.strip().startswith("Running component with"):
                    found_ub = False
                    fields = cln.strip().split(" ")
                    assert fields[4] == "vertices"
                    assert fields[7] == "edges."

                    if instance_object.max_component_vertices is None or instance_object.max_component_vertices < int(fields[3]):
                        instance_object.max_component_vertices = int(fields[3])
                    if instance_object.max_component_edges is None or instance_object.max_component_edges < int(fields[6]):
                        instance_object.max_component_edges = int(fields[6])

                    if c_best is not None:
                        if o_best is None:
                            o_best = 0
                        if max_width is None:
                            max_width = 0
                        o_best = max(o_best, c_best)
                        max_width = max(max_width, c_best)
                        c_best = None
                elif cln.strip().startswith("Component"):
                    if c_best is not None:
                        if o_best is None:
                            o_best = 0
                        o_best = max(o_best, c_best)
                    c_best = g_ub
                elif cln.strip().startswith("UB:"):
                    found_ub = True
                    if o_ub is None:
                        o_ub = 0
                    c_best = int(cln.strip().split(" ")[1])
                    o_ub = max(o_ub, int(cln.strip().split(" ")[1]))
                elif cln.strip().startswith("UB"):
                    g_ub = int(cln.strip().split(" ")[1])
                    c_best = g_ub
                elif cln.strip().startswith("LB"):
                    o_lb = max(o_lb, int(cln.strip().split(" ")[-1]))
                #TODO: Check that solved by bounds checks all components
            if only_line and only_field is not None:
                results[instance][name_fields[0]].tww = only_field

            if o_ub is not None and found_ub and (instance_object.ub is None or instance_object.ub > o_ub):
                instance_object.ub = o_ub

            if not found_ub and max_width is not None:
                if max_width == c_best:
                    print(f"Could be solved {name_fields[0]}/{instance}")

            instance_object.lb = max(instance_object.lb, o_lb)
            if c_best is not None:
                if o_best is None:
                    o_best = 0
                o_best = max(o_best, c_best)
            if results[instance][name_fields[0]].tww is not None:
                assert o_best is None or o_best == g_ub or o_best == results[instance][name_fields[0]].tww
                results[instance][name_fields[0]].best_tww = results[instance][name_fields[0]].tww
            elif found_ub:
                results[instance][name_fields[0]].best_tww = o_best

counter = defaultdict(int)
experiments = sorted(experiments)

with open(os.path.split(input)[-1].replace(".tar.bz2", "") +".csv", "w") as outp:
    outp.write("Instance,Vertices,Edges,Max Vertices,Max Edges,LB,UB,TWW")
    for ce in experiments:
        outp.write(f",{ce} Time,{ce} Mem,{ce} tww,{ce} best")
    outp.write(os.linesep)

    for ck in sorted(results.keys()):
        instance_obj = instance_results[ck]
        outp.write(f"{ck},{xstr(instance_obj.vertices)},{xstr(instance_obj.edges)},{xstr(instance_obj.max_component_vertices)},"
                   f"{xstr(instance_obj.max_component_edges)},{xstr(instance_obj.lb)},{xstr(instance_obj.ub)},{xstr(instance_obj.twin_width)}")
        for ce in experiments:
            cr = results[ck][ce]
            outp.write(f",{round(cr.runtime if cr.runtime is not None else -1, 2)},{round(cr.memory if cr.memory is not None else -1, 2)},{xstr(cr.tww)},{xstr(cr.best_tww)}")
            if cr.tww is not None:
                counter[ce] += 1
        outp.write(os.linesep)

for ck, cv in counter.items():
    print(f"{ck} {cv}")
