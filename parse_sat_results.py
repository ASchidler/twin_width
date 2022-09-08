import os
import sys
import tarfile as tf
from collections import defaultdict

results = defaultdict(dict)
instances = []

with open("sat_instances.txt") as inp:
    for cl in inp:
        instances.append(os.path.split(cl.strip().split(" ")[-1])[-1])

for cf in os.listdir("logs"):
    with tf.open("logs/" + cf) as tar_file:
        for ctf in tar_file:
            cetf = tar_file.extractfile(ctf)
            decomp = os.path.split(ctf.name)[-1].split(".")[0].split("-")[1]

            bound = sys.maxsize
            optimal = False
            instance = None
            tree_last = None

            for i, cln in enumerate(cetf):
                cln = cln.decode('ascii').strip()
                if i == 0 and decomp != "tree":
                    instance = cln.strip()
                elif i == 0:
                    inst_idx = int(os.path.split(ctf.name)[-1].split(".")[-1])
                    instance = instances[inst_idx-1]
                else:
                    if decomp != "cwd" and cln.startswith("UB"):
                        bound = int(cln.split(" ")[1])
                    elif decomp == "sat" and cln.startswith("Found "):
                        bound = int(cln.split(" ")[1])
                    elif cln.startswith("Finished"):
                        optimal = True
                    elif decomp == "tree" and cln.startswith("Searching"):
                        tree_last = int(cln.split(" ")[-1])
                    elif decomp == "tree" and cln.startswith("Found"):
                        bound = tree_last
                    elif decomp == "cwd" and cln.startswith("SAT"):
                        bound = int(cln.split(" ")[-1])

            if bound != sys.maxsize and (instance not in results or decomp not in results[instance] or results[instance][decomp][0] >= bound):
                results[instance][decomp] = (bound, 1 if optimal else 0, 1)

results_aggr = defaultdict(dict)
aggregating = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))

for c_inst, inst_results in results.items():
    if c_inst.startswith("uf20") or c_inst.startswith("ur_"):
        c_inst = c_inst[0:c_inst.replace("-", "_").rindex("_")]
        for c_decomp, c_decomp_result in inst_results.items():
            aggregating[c_inst][c_decomp][0] += c_decomp_result[0]
            aggregating[c_inst][c_decomp][1] += c_decomp_result[1]
            aggregating[c_inst][c_decomp][2] += 1
    else:
        results_aggr[c_inst] = inst_results
for c_inst, inst_results in aggregating.items():
    for c_decomp, c_decomp_result in inst_results.items():
        results_aggr[c_inst][c_decomp] = [c_decomp_result[0] / c_decomp_result[2], c_decomp_result[1], c_decomp_result[2]]

for c_k in sorted(results_aggr.keys()):
    c_l = c_k
    for c_decomp in ["sat", "cwd", "tree"]:
        if c_decomp in results_aggr[c_k]:
            c_l += f"& {round(results_aggr[c_k][c_decomp][0],1)}&{results_aggr[c_k][c_decomp][1]}/{results_aggr[c_k][c_decomp][2]}"
        else:
            c_l += "&-&-"
    c_l += "\\\\"
    c_l = c_l.replace("_", "\_")
    print(c_l)
print("Done")