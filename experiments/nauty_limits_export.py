import os
import subprocess

import networkx as nx
from pysat.solvers import Cadical

import encoding
import experiments.nauty_limits as nl
import heuristic
import sys


def dot_export(g, mg):
    output1 = "strict graph tww {" + os.linesep

    for n in g.nodes:
        output1 += f"{n} [" \
                   f"shape=circle, fontsize=11,width=0.3,height=0.2,fixedsize=true,style=filled,fontcolor=white," \
                   f"color=black, fillcolor=grey];{os.linesep}"

    for x, y in g.edges:
        if (x < y and y == mg[x]) or (y < x and x == mg[y]):
            output1 += f"{x} -- {y} [color=black,style=dotted];{os.linesep}"
        else:
            output1 += f"{x} -- {y} [color=black];{os.linesep}"

    for x, y in mg.items():
        if not g.has_edge(x, y):
            output1 += f"{x} -- {y} [color=green, style=dotted];{os.linesep}"

    return output1 + "}"


for k, v in nl.nauty_smallest.items():
    for i, c_str in enumerate(v):
        g = nx.from_graph6_bytes(bytes(c_str.strip(), encoding="ascii"))

        if len(g.nodes) > 2:
            ub = heuristic.get_ub(g)
            ub2 = heuristic.get_ub2(g)
            ub = min(ub, ub2)

            lb = 0
            glb = g.copy()
            clb = sys.maxsize
            cmn = None
            for n1 in glb:
                n1nb = set(glb.neighbors(n1))
                for n2 in glb:
                    if n1 < n2:
                        n2nb = set(glb.neighbors(n2))
                        reds = (n1nb ^ n2nb) - {n1, n2}
                        if len(reds) < clb:
                            clb = len(reds)
                            cmn = n1
                lb = max(lb, clb)
            print(f"{lb}")
            enc = encoding.TwinWidthEncoding()
            cb, od, mg = enc.run(g, Cadical, ub, check=True)

            translate = {}
            for ij in range(0, len(od)):
                translate[od[ij]] = ij + 1
            mg = {translate[x]: translate[y] for x, y in mg.items()}

            g2 = nx.Graph()
            for cu, cv in g.edges:
                g2.add_edge(translate[cu], translate[cv])

            with open(f"smallest_{k}_{i}.dot", "w") as f:
                f.write(dot_export(g2, mg))
            with open(f"smallest_{k}_{i}.png", "w") as f:
                subprocess.run(["dot", "-Kfdp", "-Tpng", f"smallest_{k}_{i}.dot"], stdout=f)


keys = [x for x in nl.nauty_total.keys()]
widths = set()
for x in nl.nauty_counts.values():
    for ck, cv in x.items():
        if cv > 0:
            widths.add(ck)
widths = list(widths)
widths.sort()
keys.sort()

header = "$\Card{V}$&\t connected&\t prime"
for cw in widths:
    header += f"&\t{cw}"

print("\\begin{tabular}{lrr"+ "".join(("r" for _ in range(0, len(widths)))) +"}")
print("\\toprule")
print(f"{header}\\\\")
print("\\midrule")

for k in keys:
    line = f"{k}&\t {nl.nauty_total[k]}&\t {nl.nauty_prime[k]}"
    for cw in widths:
        if cw in nl.nauty_counts[k]:
            line += f"&\t{nl.nauty_counts[k][cw]}"
        else:
            line += f"&\t0"
    line += "\\\\"
    print(line)
print("\\bottomrule")
print("\\end{tabular}")
