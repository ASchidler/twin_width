import sys

from pysat.formula import CNF, IDPool
from pysat.solvers import Cadical
from pysat.card import CardEnc
import networkx as nx
import tools
import subprocess

red_deg = 3
size = 6
depth = 1

pool = IDPool()
def red(x, y):
    if x < y:
        return pool.id(f"red_{x}_{y}")
    else:
        return pool.id(f"red_{y}_{x}")

def new_red(x, y, z1, z2):
    return pool.id(f"nred_{min(x, y)}_{max(x, y)}_{min(z1, z2)}_{max(z1,z2)}")

def black(x, y):
    return pool.id(f"black_{min(x, y)}_{max(x, y)}")

def c1(x, y, z):
    return pool.id(f"c1_{min(x, y)}_{max(x, y)}_{z}")
def c2(x, y, z):
    return pool.id(f"c2_{min(x, y)}_{max(x, y)}_{z}")



with Cadical() as slv:
    def atmost_constr(cv, bnd):
        cid = pool.top
        exceeded = pool.id(f"{cid}_amo_exceeded")

        for i, v in enumerate(cv):
            slv.add_clause([-v, pool.id(f"{cid}_amo_{i}_1")])

        for cb in range(1, bnd+1):
            for i, v in enumerate(cv[1:]):
                slv.add_clause([-pool.id(f"{cid}_amo_{i}_{cb}"), pool.id(f"{cid}_amo_{i+1}_{cb}")])

        for cb in range(2, bnd+1):
            for i, v in enumerate(cv[1:]):
                slv.add_clause([-pool.id(f"{cid}_amo_{i}_{cb-1}"), -v, pool.id(f"{cid}_amo_{i+1}_{cb}")])

        for i, v in enumerate(cv[1:]):
            slv.add_clause([-pool.id(f"{cid}_amo_{i}_{bnd}"), -v, exceeded])

        return exceeded

    def descend(t1, t2, cd):
        varname = f"nred"
        for i in range(0, len(t1)):
            varname += f"_{t1[i]}_{t2[i]}"

        if cd >= depth:
            cardvards = []
            for x in range(0, size):
                if all(x != v for v in t1) and x == t2[-1]:
                    cvars = [-pool.id(varname + f"_{min(x, z)}_{max(x, z)}") for z in range(0, size) if z != x and all(z != v for v in t1)]
                    # exc = atmost_constr(cvars, size-red_deg-2)
                    # cardvards.append(-exc)
                    CardEnc.atleast([-x for x in cvars], bound=red_deg + 1, vpool=pool)
            # slv.add_clause(cardvards)

            return

        for x in range(0, size):
            if any(v == x for v in t1):
                continue

            for y in range(x + 1, size):
                if any(v == y for v in t1):
                    continue
                slv.append_formula(
                    CardEnc.atmost([pool.id(varname + f"_{z}") for z in range(0, size) if all(z != v for v in t1) and z != t2[-1]], bound=red_deg, vpool=pool))

                for z in range(0, size):
                    if x == z or y == z or any(v == z for v in t1):
                        continue

                    oldvar1 = pool.id(varname + f"_{x}_{z}")
                    oldvar2 = pool.id(varname + f"_{y}_{z}")
                    nvar = pool.id(varname + f"_{x}_{y}_{z}")

                    slv.add_clause([-nvar, oldvar1, oldvar2, c1(x, y, z), c2(x, y, z)])
                    slv.add_clause([-oldvar1, nvar])
                    slv.add_clause([-oldvar2, nvar])
                    slv.add_clause([-c1(x, y, z), nvar])
                    slv.add_clause([-c2(x, y, z), nvar])

                descend([*t1, x], [*t2, y], cd + 1)


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

                    slv.add_clause([-new_red(n, n2, n2, n3), red(n, n3), red(n2, n3), c1(n, n2, n3), c2(n, n2, n3)])
                    slv.add_clause([-red(n, n3), new_red(n, n2, n2, n3)])
                    slv.add_clause([-red(n2, n3), new_red(n, n2, n2, n3)])
                    slv.add_clause([-c1(n, n2, n3), new_red(n, n2, n2, n3)])
                    slv.add_clause([-c2(n, n2, n3), new_red(n, n2, n2, n3)])

                    for n4 in range(n3+1, size):
                        if n != n4 and n2 != n4:
                            slv.add_clause([-new_red(n, n2, n3, n4), red(n3, n4)])
                            slv.add_clause([-red(n3, n4), new_red(n, n2, n3, n4)])

                    descend([n], [n2], 1)

                    # Break some symmetries
                    for n4 in range(0, n3):
                        if n4 != n and n4 != n2:
                            slv.add_clause([red(n, n3), black(n, n3), red(n2, n4), black(n2, n4)])
                            # slv.add_clause([-black(n, n3), red(n2, n4), black(n2, n4)])


            # slv.append_formula(CardEnc.atleast([new_red(n, n2, n3) for n3 in range(0, size) if n != n3 and n2 != n3], bound=red_deg+1, vpool=pool))

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

            slv.add_clause(clause)
            print("")

            # instance_name = f"structure_{red_deg}_{size}_{cnt}_{depth}"
            #
            # with open(f"{instance_name}.dot", "w") as f:
            #     f.write(tools.dot_export(g, None, None, False))
            # with open(f"{instance_name}.png", "w") as f:
            #     subprocess.run(["circo", "-Tpng", f"{instance_name}.dot"], stdout=f)
            #
            # if cnt > 10:
            #     exit(0)
