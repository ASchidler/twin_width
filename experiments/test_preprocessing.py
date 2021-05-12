import parser
import os
import tools
import matplotlib.pyplot as plt
import time

import networkx as nx
from networkx.generators.random_graphs import gnp_random_graph

# for inst in os.listdir("instances"):
#     g, _ = parser.parse(os.path.join("instances", inst))
#     mods = tools.find_modules(g)
#     ml = max(len(m) for m in mods)
#
#     print(f"{inst} {len(mods)} {ml}")

# mods = 0
# ml = 0
# for _ in range(0, 10):
#     g = gnp_random_graph(60, 0.5)
#     ms = tools.find_modules(g)
#     ml += max(len(m) for m in ms)
#     mods += len(ms)
#
# print(f"{mods/10} {ml/10}")

#twdir = "/home/asc/Dev/graphs/treewidth_benchmarks/twlib-graphs/all"
twdir = "/home/asc/Dev/graphs/treewidth_benchmarks/transit-tw/gtfs-data-exchange"
sizes = []
reduced = []

for inst in sorted(os.listdir(twdir),
                       key=lambda x: os.path.getsize(os.path.join(twdir, x))):
    if inst.find("-minor-") >= 0:
        continue

    g, _ = parser.parse(os.path.join(twdir, inst))
    if len(g.nodes) > 20000:
        continue

    start_time = time.time()
    mods = tools.find_modules(g)
    # if time.time() - start_time > 3600:
    #     break
    if mods is None:
        continue
    ml = max(len(m) for m in mods)
    to_process = len(g.nodes)

    for cm in mods:
        to_process -= len(cm) - 1

    to_process = max(to_process, ml)
    sizes.append(len(g.nodes))
    reduced.append(to_process)
    print(f"{inst} {len(g.nodes)} {to_process}")

# for r, d, f in os.walk("/home/asc/Dev/graphs/treewidth_benchmarks/twlib-graphs/twlib-graphs/probabilistic_networks"):
#     for inst in f:
#         if inst.endswith(".bz2"):
#             g, _ = parser.parse(os.path.join(r, inst))
#             mods = tools.find_modules(g)
#             if mods is None:
#                 continue
#             ml = max(len(m) for m in mods)
#
#             print(f"{inst} {len(mods)} {ml}")

max_s = max(sizes)
max_r = max(reduced)

fig, ax = plt.subplots(figsize=(4.5, 3), dpi=80)
ax.set_axisbelow(True)
#ax = fig.add_axes([0,0,1,1])
ax.scatter(reduced, sizes, marker=".")
ax.set_xlabel('Reduced')
ax.set_ylabel('Nodes')
ax.axline([0, 0], [1, 1], linestyle="--", color="grey")
#ax.set_title('scatter plot')
plt.rcParams["legend.loc"] = 'lower left'
plt.rcParams['savefig.pad_inches'] = 0
plt.autoscale(tight=True)
# plt.xscale("log")
# plt.yscale("log")
plt.xlim(0, max_r + 5)
plt.ylim(0, max_s + 5)

plt.plot(color='black', linewidth=0.5, linestyle='dashed', markersize=2)
plt.savefig("reduction_data_1.pdf", bbox_inches='tight')
plt.show()
