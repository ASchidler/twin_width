import sys
import tarfile as tf
import os
from collections import defaultdict

input = "logs/tww-twlib.tar.bz2"
instance_path = "/Users/andre/Downloads/exact-private/"


class ExperimentResult:
    def __init__(self):
        self.tww = None
        self.runtime = None
        self.memory = None
        self.best_tww = None


class InstanceResult:
    def __init__(self, vertices: int, edges: int):
        self.vertices = vertices
        self.edges = edges


experiments = set()
results = defaultdict(lambda: defaultdict(ExperimentResult))

with tf.open(input) as ctf:
    for clf in ctf:
        name_fields = clf.name.split(".")
        experiments.add(name_fields[0])
        instance = int(name_fields[-1])
        cetf = ctf.extractfile(clf)

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

            for i, cln in enumerate(cetf):
                cln = cln.decode('ascii').strip()
                if only_field is None and only_line:
                    try:
                        only_field = int(cln.strip())
                    except ValueError:
                        only_line = False
                else:
                    only_line = False

                if cln.strip().startswith("Final Result:"):
                    results[instance][name_fields[0]].tww = int(cln.split(":")[1].strip())
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
                elif cln.strip().startswith("Found ") and not cln.strip().startswith("Found subgraph"):
                    results[instance][name_fields[0]].best_tww = int(cln.split(" ")[1].strip())
                # "Improved bound to 8"
                # "Component n"
                # "Found"
                # "Running component with 8 vertices and 13 edges."
                #TODO: Check that solved by bounds checks all components
            if only_line and only_field is not None:
                results[instance][name_fields[0]].tww = only_field

counter = defaultdict(int)
experiments = sorted(experiments)

with open("tww-pace.csv", "w") as outp:
    outp.write("Instance")
    for ce in experiments:
        outp.write(f",{ce} Time, {ce} Mem,{ce} tww")
    outp.write(os.linesep)

    for ck in sorted(results.keys()):
        outp.write(f"{ck}")
        for ce in experiments:
            cr = results[ck][ce]
            outp.write(f",{round(cr.runtime if cr.runtime is not None else -1, 2)},{round(cr.memory if cr.memory is not None else -1, 2)},{cr.tww if cr.tww is not None else ''}")
            if cr.tww is not None:
                counter[ce] += 1
        outp.write(os.linesep)

for ck, cv in counter.items():
    print(f"{ck} {cv}")
