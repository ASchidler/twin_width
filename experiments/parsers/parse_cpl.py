import matplotlib.pyplot as plt
import numpy as np
import parser_mappings as pm
from collections import defaultdict

prefixes = ["tww-long", "tww-twlong"]
target_prefix = prefixes[1]
target_solvers = {"cdo", "cod", "co", "sat0", "sat1c", "winner"}
penalty_time = 7200

names = []
data = defaultdict(list)

linestyles = [
        {"c": "green",   "marker": "H", "ms": 5, "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "green",   "mew": 0.75},
        {"c": "red",     "marker": "^", "ms": 5, "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "red",     "mew": 0.75},
        {"c": "blue",    "marker": "x", "ms": 5, "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "blue",    "mew": 0.75},
        {"c": "brown",   "marker": "+", "ms": 5, "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "brown",   "mew": 0.75},
        {"c": "orange",  "marker": "D", "ms": 5, "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "orange",  "mew": 0.75},
        {"c": "magenta", "marker": "*", "ms": 5, "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "magenta", "mew": 0.75},
        {"c": "cyan",    "marker": "o", "ms": 5, "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "cyan",    "mew": 0.75},
        {"c": "black",   "marker": "d", "ms": 5, "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "black",   "mew": 0.75},
        {"c": "#666aee", "marker": "v", "ms": 5, "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "#666aee", "mew": 0.75},
        {"c": "grey",    "marker": ">", "ms": 5, "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "grey",    "mew": 0.75},
        {"c": "green",   "marker": "^", "ms": 5, "ls": "--", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "green",   "mew": 0.75},
        {"c": "red",     "marker": "H", "ms": 5, "ls": "--", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "red",     "mew": 0.75},
        {"c": "blue",    "marker": "+", "ms": 5, "ls": "--", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "blue",    "mew": 0.75},
        {"c": "brown",   "marker": "x", "ms": 5, "ls": "--", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "brown",   "mew": 0.75},
        {"c": "orange",  "marker": "*", "ms": 5, "ls": "--", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "orange",  "mew": 0.75},
        {"c": "magenta", "marker": "D", "ms": 5, "ls": "--", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "magenta", "mew": 0.75},
        {"c": "cyan",    "marker": "d", "ms": 5, "ls": "--", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "cyan",    "mew": 0.75},
        {"c": "black",   "marker": "o", "ms": 5, "ls": "--", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "black",   "mew": 0.75},
        {"c": "#666aee", "marker": ">", "ms": 5, "ls": "--", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "#666aee", "mew": 0.75},
        {"c": "grey",    "marker": "v", "ms": 5, "ls": "--", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "grey",    "mew": 0.75},
        {"c": "green",   "marker": "v", "ms": 5, "ls": ":", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "green",   "mew": 0.75},
        {"c": "red",     "marker": ">", "ms": 5, "ls": ":", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "red",     "mew": 0.75},
        {"c": "blue",    "marker": "o", "ms": 5, "ls": ":", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "blue",    "mew": 0.75},
        {"c": "brown",   "marker": "d", "ms": 5, "ls": ":", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "brown",   "mew": 0.75},
        {"c": "orange",  "marker": "D", "ms": 5, "ls": ":", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "orange",  "mew": 0.75},
        {"c": "magenta", "marker": "*", "ms": 5, "ls": ":", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "magenta", "mew": 0.75},
        {"c": "cyan",    "marker": "x", "ms": 5, "ls": ":", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "cyan",    "mew": 0.75},
        {"c": "black",   "marker": "+", "ms": 5, "ls": ":", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "black",   "mew": 0.75},
        {"c": "#666aee", "marker": "H", "ms": 5, "ls": ":", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "#666aee", "mew": 0.75},
        {"c": "grey",    "marker": "^", "ms": 5, "ls": ":", "lw": 1, "alpha": 0.7, "mfc": "white", "mec": "grey",    "mew": 0.75}
    ]

target_cols = []
with open(target_prefix +".csv") as inp:
    for i, cl in enumerate(inp):
        fd = cl.strip().split(",")
        if i == 0:
            for j in range(1, len(fd)):
                if fd[j].endswith("Time") and any(fd[j].startswith(target_prefix + "-"+ x +" ") for x in target_solvers):
                    target_cols.append((pm.resolve_solver(fd[j].split(" ")[0]), j))
        else:
            for cdn, cdi  in target_cols:
                if fd[cdi+2] == "":
                    data[cdn].append(penalty_time)
                else:
                    data[cdn].append(float(fd[cdi]))

max_value = 0
for d in data.values():
    d.sort()
    # for ci in range(1, len(d)):
    #     d[ci] += d[ci-1]
    max_value = max(max_value, d[-1])

names = sorted(data.keys())

coords = []
for ck in names:
    d = data[ck]
    d.append(max_value)
    coords.append(np.array(d))
    coords.append(np.arange(1, len(d) + 1))  # xs (separate for each line)

lines = plt.plot(*coords, zorder=3)
for i, l in enumerate(lines):
    plt.setp(l, **linestyles[i])

plt.ylabel("Instance")
plt.xlabel("Runtime [s]")
names.sort(key=lambda x: pm.order_solver(x))
plt.legend([pm.map_solver(s) for s in names])
# plt.gcf().subplots_adjust(left=0.15)
# plt.gcf().subplots_adjust(bottom=0.15)
plt.plot()
# plt.xlim(0, 425)
plt.savefig(f"cactus_{target_prefix}.pdf")
plt.show()
