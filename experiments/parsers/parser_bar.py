vertex_bins = [25, 50, 100, 250]
density_bins = [round(0.1 * x, 1) for x in range(1, 8)]
tww_bins = [3,5,10,25]

target = 2
if target == 0:
    target_bins = vertex_bins
elif target == 1:
    target_bins = density_bins
elif target == 2:
    target_bins = tww_bins

bins = [[0,0] for _ in range(0, len(target_bins)+1)]

import matplotlib.pyplot as plt
import numpy as np

for x in ["tww-twlong.csv", "tww-long.csv"]:
    with open(x) as inp:
        for i, cl in enumerate(inp):
            fd = cl.strip().split(",")
            if i > 0:
                if fd[1] == "" or fd[2] == "":
                    continue
                vertices = int(fd[3]) if fd[3] != "" else int(fd[1])
                edges = int(fd[4]) if fd[4] != "" else int(fd[2])
                solved = fd[7] != ""
                if target == 0:
                    val = vertices
                elif target == 1:
                    val = vertices * (vertices-1) / 2
                    val = edges / val
                elif target == 2:
                    if not solved:
                        if fd[6] == "" or fd[5] == "":
                            continue
                        val = (int(fd[6]) + int(fd[5])) // 2
                    else:
                        val = int(fd[7])

                for ci, cv in enumerate(target_bins):
                    if val < cv:
                        bins[ci][0 if solved else 1] += 1
                        break
                    elif ci == len(target_bins) - 1:
                        bins[-1][0 if solved else 1] += 1

ind = np.arange(len(bins))
fig = plt.subplots(figsize =(10, 7))
p1 = plt.bar(ind, [x[0] for x in bins], 0.35, color="green")
p2 = plt.bar(ind, [x[1] for x in bins], 0.35, bottom =  [x[0] for x in bins], color=red)

plt.ylabel('Instances')
plt.xticks(ind, [*[f"$<${x}" for x in target_bins], f"$\geq${target_bins[-1]}"])
# plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Solved', 'Unsolved'))

plt.show()