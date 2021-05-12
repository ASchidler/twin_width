import sys
import matplotlib.pyplot as plot

data = {}

with open(sys.argv[1]) as logfile:
    for ln in logfile:
        fields = ln.strip().split()
        if fields[0] not in data:
            data[fields[0]] = []
        data[fields[0]].append((float(fields[1]), float(fields[2])))

for cv in data.values():
    cv.sort()

scale = [round(x[0], 2) for x in next(iter(data.values()))]


fig = plot.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
markers = ["x", "^", "P"]
names = []

for ck, cv in data.items():
    ax.scatter(scale, [x[1] for x in cv], marker=markers.pop())
    names.append(ck + " nodes")
ax.legend(names)
ax.set_xlabel('p', labelpad=-5)
ax.set_ylabel('tww')

#ax.set_title('Randomly generated graphs')
ax.grid(True)

plot.rcParams['savefig.pad_inches'] = 0
plot.savefig("random.pdf", bbox_inches='tight')
plot.show()

print("Done")
