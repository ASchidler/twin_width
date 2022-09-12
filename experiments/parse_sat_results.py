import tarfile as tf
import os
from collections import defaultdict
import parser

instances = []

results = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, {}]))

with open("sat_instances.txt") as inp:
    for cl in inp:
        instances.append(os.path.split(cl.strip().split(" ")[-1])[-1])

for cf in os.listdir("logs"):
    if not (cf.endswith("cwd-2.tar.bz2") or cf.endswith("sat-2.tar.bz2") or cf.endswith("tree-3.tar.bz2")):
        continue
    with tf.open("logs/"+cf) as ctf:
        for clf in ctf:
            if clf.name.split(".")[1].startswith("o"):
                decomp = clf.name.split(".")[0].split("-")[1]
                cetf = ctf.extractfile(clf)
                instance_name = None
                bound = -1
                optimal = False
                last_try = 0
                original_name = None

                for i, cln in enumerate(cetf):
                    cln = cln.decode('ascii').strip()

                    if i == 0:
                        if cln.startswith("UB"):
                            instance_name = instances[int(clf.name.split(".")[-1])-1]
                        else:
                            instance_name = os.path.split(cln)[-1]

                        if instance_name.endswith(".all.q") or instance_name == "runner.py":
                            break
                        instance_name = instance_name[:-4] # remove .cnf
                        original_name = instance_name
                        if instance_name.startswith("uf20-") or instance_name.startswith("ur_"):
                            instance_name = instance_name[:instance_name.replace("_", "-").rindex("-")]

                    if cln.startswith("UB") or (decomp == "sat" and cln.startswith("Found")):
                        bound = int(cln.split(" ")[1])
                    elif decomp != "cwd" and cln.startswith("Finished, result"):
                        optimal = True
                    elif decomp == "tree" and cln.startswith("Searching"):
                        last_try = int(cln.split(" ")[-1])
                    elif decomp == "tree" and cln.startswith("Found solution"):
                        bound = last_try
                    elif decomp == "cwd" and cln.startswith("SAT "):
                        bound = int(cln.split(" ")[-1])
                    elif decomp == "cwd" and cln.find("Clique Width:") > -1:
                        bound = int(cln.split(" ")[-1])
                        optimal = True

                if instance_name is not None and bound != -1:
                    results[instance_name][decomp][0] += bound
                    if optimal:
                        results[instance_name][decomp][1] += 1
                    results[instance_name][decomp][2] += 1
                    if original_name != instance_name:
                        results[instance_name][decomp][3][original_name] = (bound, optimal)

files = []
for r, d, f in os.walk("sat_experiments"):
    for cf in f:
        if cf.endswith(".cnf"):
            files.append(os.path.join(r, cf))

with open("signed_results.csv", "w") as outp:
    outp.write("Instance;Nodes;Edges;stww;optimal;scwd;optimal;tw;optimal"+os.linesep)
    for c_instance in sorted(results.keys()):
        nodes = 0
        edges = 0
        if c_instance.startswith("uf20") or c_instance.startswith("ur_"):
            cnt = 0
            for x in (x for x in files if os.path.split(x)[-1].startswith(c_instance)):
                g = parser.parse_cnf(x)
                nodes += len(g.nodes)
                edges += len(g.edges)
                cnt += 1
            nodes = round(nodes / cnt, 0)
            edges = round(edges / cnt, 0)
        else:
            g = parser.parse_cnf(next(iter(x for x in files if x.endswith(c_instance + ".cnf"))))
            nodes = len(g.nodes)
            edges = len(g.edges)

        c_l = "&"+ c_instance.replace("_", "\_")
        c_l += f"&{nodes}&{edges}"

        for decomp in ["sat", "cwd", "tree"]:
            if decomp in results[c_instance]:# and (decomp == "sat" or instance_name == "uf20" or results[c_instance][decomp][1] == results[c_instance][decomp][2]):
                c_r = results[c_instance][decomp]
                c_l += f"&{'*' if decomp == 'sat' and c_r[1] != c_r[2] else ''}{round(c_r[0]/c_r[2], 1)}"
            else:
                c_l += "&-"

        if len(results[c_instance]["tree"][3]) == 0:
            outp.write(f"{c_instance};{nodes};{edges}")
            for decomp in ["sat", "cwd", "tree"]:
                if decomp in results[c_instance]:
                    c_r = results[c_instance][decomp]
                    outp.write(f";{c_r[0]};{c_r[1] == 1}")
                else:
                    outp.write(";;")
            outp.write(os.linesep)
        else:
            for c_k in sorted(results[c_instance]["tree"][3].keys()):
                g = parser.parse_cnf(next(iter(x for x in files if x.endswith(c_k + ".cnf"))))
                nodes = len(g.nodes)
                edges = len(g.edges)
                outp.write(f"{c_k};{nodes};{edges}")

                for decomp in ["sat", "cwd", "tree"]:
                    if decomp in results[c_instance] and c_k in results[c_instance][decomp][3]:
                        c_r = results[c_instance][decomp][3][c_k]
                        outp.write(f";{c_r[0]};{c_r[1]}")
                    else:
                        outp.write(";;")
                outp.write(os.linesep)

        c_l += "\\\\"
        print(c_l)



