import os
import networkx as nx
import bz2
import shutil

base_path = "twlib-graphs"
target = "twlib-small"

for r, d, f in os.walk(base_path):
    for cf in f:
        if cf.endswith(".bz2") and not cf.endswith("tgz.bz2"):
            cg = bz2.BZ2File(os.path.join(r, cf), "r")
            found = False
            outside = False
            vertices = set()
            nodes = 0
            cfile = ""
            header = ""

            for cl in cg:
                cl = cl.decode("ascii").strip()

                if cl.startswith("p"):
                    data = cl.split(" ")
                    nodes = int(data[2])
                    found = True
                    header = f"p edge {nodes} {data[3]}" + os.linesep
                    if 20 > nodes or nodes > 200:
                        outside = True
                        break
                if found:
                    if cl.startswith("e") or cl.startswith("a"):
                        cfields = cl.split()
                        n1 = int(cfields[1])
                        n2 = int(cfields[2])

                        vertices.add(n1)
                        vertices.add(n2)

                        cfile += f"e {n1} {n2}"+os.linesep

            outside = outside or 20 > len(vertices) or len(vertices) > 200

            if not found:
                print(f"Error {cf}")
            elif not outside and any(int(x) > nodes or (int(x) == nodes and 0 in vertices) for x in vertices):
                print(f"Wrong indexed {cf}")
            elif not outside:
                if len(vertices) != nodes:
                    print(f"Missing vertices {cf}")
                if not outside and 0 in vertices:
                    for cc in range(nodes, 0, -1):
                        cfile = cfile.replace(f" {cc-1} ", f" {cc} ").replace(f" {cc-1}{os.linesep}", f" {cc}{os.linesep}")
                    print(f"Corrected indices {cf}")

                nf = bz2.BZ2File(os.path.join(target, cf), "w")
                nf.write((header + cfile).encode('utf-8'))
                nf.close()

            cg.close()
