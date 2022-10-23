import sys
import matplotlib.pyplot as plot

target_size = 25
runtime_graph = False
logfiles = ["random_results_0.csv", "random_results_z1.csv"]
log_names = ["Relative", "Absolute"]
log_data = []

for clogfile in logfiles:
    with open(clogfile) as logfile:
        log_data.append({})
        for ln in logfile:
            fields = ln.strip().split(";")
            if fields[0] not in log_data[-1]:
                log_data[-1][fields[0]] = []
            log_data[-1][fields[0]].append((float(fields[1]), float(fields[2]), float(fields[3])))

for cdata in log_data:
    for cv in cdata.values():
        cv.sort()


fig = plot.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
markers = ["x", "^", "P", "."]
ax.set_xlabel('p', labelpad=-5)
names = []

if runtime_graph:
    ax.set_ylabel('Runtime [s]')
    for i, cdata in enumerate(log_data):
        target_data = cdata[str(target_size)]
        scale = [round(x[0], 2) for x in target_data]
        ax.scatter(scale, [x[2] for x in target_data], marker=markers.pop())
        names.append(log_names[i])
else:
    data = log_data[0]

    for ck, cv in data.items():
        if int(ck) > target_size:
            continue
        scale = [round(x[0], 2) for x in cv]
        ax.scatter(scale, [x[1] for x in cv], marker=markers.pop())
        names.append(ck + " nodes")
    ax.set_ylabel('tww')

ax.legend(names)
#ax.set_title('Randomly generated graphs')
ax.grid(True)

plot.rcParams['savefig.pad_inches'] = 0
plot.savefig(f"random_{'tww' if not runtime_graph else f'rt{target_size}'}.pdf", bbox_inches='tight')
plot.show()

print("Done")
