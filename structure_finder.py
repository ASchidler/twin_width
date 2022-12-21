import sys

from pysat.formula import CNF, IDPool
from pysat.solvers import Cadical
from pysat.card import CardEnc
import networkx as nx
import tools
import subprocess

red_deg = 5
size = 8

pool = IDPool()
def red(x, y):
    if x < y:
        return pool.id(f"red_{x}_{y}")
    else:
        return pool.id(f"red_{y}_{x}")

def new_red(x, y, z):
    return pool.id(f"nred_{min(x, y)}_{max(x, y)}_{z}")

def black(x, y):
    return pool.id(f"black_{min(x, y)}_{max(x, y)}")

def c1(x, y, z):
    return pool.id(f"c1_{min(x, y)}_{max(x, y)}_{z}")
def c2(x, y, z):
    return pool.id(f"c2_{min(x, y)}_{max(x, y)}_{z}")

with Cadical() as slv:
    for n in range(0, size):
        slv.append_formula(CardEnc.atmost([red(n2, n) for n2 in range(0, size) if n != n2], bound=red_deg, vpool=pool))

        for n2 in range(n+1, size):
            for n3 in range(0, size):
                if n != n3 and n2 != n3:
                    slv.add_clause([-black(n, n3), black(n2, n3), c1(n, n2, n3)])
                    slv.add_clause([-c1(n, n2, n3), black(n, n3)])
                    slv.add_clause([-c1(n, n2, n3), -black(n2, n3)])
                    slv.add_clause([black(n, n3), -black(n2, n3), c2(n, n2, n3)])
                    slv.add_clause([-c2(n, n2, n3), -black(n, n3)])
                    slv.add_clause([-c2(n, n2, n3), black(n2, n3)])

                    slv.add_clause([-new_red(n, n2, n3), red(n, n3), red(n2, n3), c1(n, n2, n3), c2(n, n2, n3)])
                    slv.add_clause([-red(n, n3), new_red(n, n2, n3)])
                    slv.add_clause([-red(n2, n3), new_red(n, n2, n3)])
                    slv.add_clause([-c1(n, n2, n3), new_red(n, n2, n3)])
                    slv.add_clause([-c2(n, n2, n3), new_red(n, n2, n3)])

                    for n4 in range(0, n3):
                        if n4 != n and n4 != n2:
                            slv.add_clause([red(n, n3), black(n, n3), red(n2, n4), black(n2, n4)])
                            # slv.add_clause([-black(n, n3), red(n2, n4), black(n2, n4)])


            slv.append_formula(CardEnc.atleast([new_red(n, n2, n3) for n3 in range(0, size) if n != n3 and n2 != n3], bound=red_deg+1, vpool=pool))

    result = True

    import networkx as nx
    cnt = 0
    while result:
        result = slv.solve()

        if not result:
            print("UNSAT")
        else:
            cnt += 1
            print("SAT")

            model = slv.get_model()
            model = [x > 0 for x in model]
            model.insert(0, None)

            clause = []

            g = nx.Graph()

            for n in range(0, size):
                reds = 0
                for n2 in range(0, size):
                    if n != n2:
                        if model[red(n, n2)]:
                            g.add_edge(n, n2, red=True)
                            sys.stdout.write(f"r")
                            clause.append(-red(n, n2))
                            reds += 1
                        elif model[black(n, n2)]:
                            sys.stdout.write(f"b")
                            clause.append(-black(n, n2))
                            g.add_edge(n, n2, red=False)
                        else:
                            sys.stdout.write(" ")
                    else:
                        sys.stdout.write("x")
                print(f" {reds}")

            instance_name = f"structure_{red_deg}_{size}_{cnt}"

            with open(f"{instance_name}.dot", "w") as f:
                f.write(tools.dot_export(g, None, None, False))
            with open(f"{instance_name}.png", "w") as f:
                subprocess.run(["circo", "-Tpng", f"{instance_name}.dot"], stdout=f)

            slv.add_clause(clause)
            print("")
            if cnt > 10:
                exit(0)
