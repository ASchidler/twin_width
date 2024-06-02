from pysat.formula import CNF, IDPool
from pysat.card import CardEnc
from pysat.solvers import Cadical153, Glucose4
from networkx.generators.lattice import grid_2d_graph
import networkx as nx
import encoding


def encode_partition(g, bound, steps=1, start=0):
    # for c_parts in range(bound + 10, len(g.nodes)):
    for c_parts in range(len(g.nodes) - 1 - start, bound + 2 + steps - 1, -1):
        max_steps = min(steps, c_parts - bound - 1)
        formula = CNF()
        pool = IDPool()
        for c_step in range(0, max_steps):
            # Last vertex is in the first partition
            formula.append([pool.id(f"p_{c_step}_{0}_{len(g.nodes)}")])

            # Each vertex in exactly one partition
            for i in range(1, len(g.nodes) + 1):
                formula.append([pool.id(f"p_{c_step}_{cp}_{i}") for cp in range(0, c_parts + c_step)])
                # for cp in range(i, c_parts + c_step):
                #     formula.append([-pool.id(f"p_{c_step}_{cp}_{i}")])

                for cp1 in range(0, c_parts + c_step):
                    for cp2 in range(cp1+1, c_parts + c_step):
                        formula.append([-pool.id(f"p_{c_step}_{cp1}_{i}"), -pool.id(f"p_{c_step}_{cp2}_{i}")])

            # Each partition must contain a vertex
            for cp in range(0, c_parts + c_step):
                formula.append([pool.id(f"p_{c_step}_{cp}_{i}") for i in range(1, len(g.nodes)+1)])

            # Symmetry breaking, always pick highest possible class
            if c_step == 0:
                for i in range(1, len(g.nodes)+1):
                    for cp1 in range(1, c_parts + c_step):
                        higher_vertices = [pool.id(f"p_{c_step}_{cp1}_{j}") for j in range(i + 1, len(g.nodes) + 1)]
                        formula.append([-pool.id(f"x_{c_step}_{cp1}_{i}"), *higher_vertices])
                        for cp2 in range(0, cp1):
                            higher_vertices = [pool.id(f"p_{c_step}_{cp2}_{j}") for j in range(i+1, len(g.nodes)+1)]
                            formula.append([-pool.id(f"p_{c_step}_{cp1}_{i}"), pool.id(f"x_{c_step}_{cp1}_{i}"), *higher_vertices])
            # else:
            #     for i in range(1, len(g.nodes)+1):
            #         for cp1 in range(0, c_parts + c_step - 1):
            #             higher_vertices = [pool.id(f"p_{c_step}_{cp1}_{j}") for j in range(i + 1, len(g.nodes) + 1)]
            #             formula.append([-pool.id(f"m_{c_step}_{cp1}"), -pool.id(f"p_{c_step}_{c_parts + c_step-1}_{i}"), *higher_vertices])

            # Create red edges
            for i in range(1, len(g.nodes)+1):
                for j in range(1, len(g.nodes)+1):
                    for k in range(1, len(g.nodes)+1):
                        if i != j and i != k and j != k:
                            if g.has_edge(i, k) and not g.has_edge(j, k):
                                for cp in range(0, c_parts + c_step):
                                    formula.append([-pool.id(f"p_{c_step}_{cp}_{i}"), -pool.id(f"p_{c_step}_{cp}_{j}"), pool.id(f"p_{c_step}_{cp}_{k}"), pool.id(f"c_{c_step}_{cp}_{k}")])

            for i in range(1, len(g.nodes)+1):
                for cp1 in range(0, c_parts + c_step):
                    for cp2 in range(0, c_parts + c_step):
                        if cp1 != cp2:
                            formula.append([-pool.id(f"p_{c_step}_{cp1}_{i}"), -pool.id(f"c_{c_step}_{cp2}_{i}"), pool.id(f"r_{c_step}_{min(cp1, cp2)}_{max(cp1, cp2)}")])

            # Constrain cardinality
            for cp1 in range(0, c_parts):
                formula.extend(CardEnc.atmost([pool.id(f"r_{c_step}_{min(cp1, cp2)}_{max(cp1, cp2)}") for cp2 in range(0, c_parts + c_step) if cp1 != cp2],
                                              bound=bound, vpool=pool))

            if c_step > 0:
                formula.append([pool.id(f"m_{c_step}_{cp}")for cp in range(0, c_parts + c_step - 1)])
                for cp1 in range(0, c_parts + c_step - 1):
                    for cp2 in range(cp1+1, c_parts + c_step - 1):
                        formula.append([-pool.id(f"m_{c_step}_{cp1}"), -pool.id(f"m_{c_step}_{cp2}")])

                for cp in range(0, c_parts + c_step - 1):
                    for cn in range(1, len(g.nodes)+1):
                        formula.append([-pool.id(f"m_{c_step}_{cp}"), -pool.id(f"p_{c_step}_{c_parts + c_step - 1}_{cn}"), pool.id(f"p_{c_step-1}_{cp}_{cn}")])
                        formula.append([-pool.id(f"p_{c_step}_{cp}_{cn}"), pool.id(f"p_{c_step-1}_{cp}_{cn}")])

        with Cadical153() as slv:
            counter = 0
            slv.append_formula(formula)
            while slv.solve():
                counter += 1
                model = slv.get_model()

                mg = {}
                tord = []

                last_part = []
                for target_steps in range(max_steps - 1, -1, -1):
                    parts = list()
                    for cp in range(0, c_parts + target_steps):
                        parts.append({i for i in range(1, len(g.nodes)+1) if model[pool.id(f"p_{target_steps}_{cp}_{i}")-1] > 0})

                    if target_steps == 0:
                        for i, cp in enumerate(parts):
                            for cp2 in parts[i+1:c_parts]:
                                assert(max(cp) > max(cp2))

                    if len(last_part) > 0:
                        npart = next(iter(cp for cp in parts if tord[-1] in cp))
                        target = max(npart)
                        mg[tord[-1]] = target

                        opart1 = last_part[-1]
                        opart2 = next(iter(cp for cp in last_part if target in cp))
                        assert(max(opart1) != max(opart2))
                        assert(max(opart1) < max(opart2))

                        for i in range(0, len(parts)):
                            if max(parts[i]) != target:
                                assert(parts[i] == last_part[i])

                    tord.append(max(parts[-1]))
                    last_part = parts

                enc = encoding.TwinWidthEncoding()
                # if not isinstance(enc.run(g, Cadical153, bound, lb=bound, parts=last_part), int):
                if not isinstance(enc.run(g, Cadical153, bound, lb=bound, i_mg=mg, i_od=tord), int):
                    print("Found witness")
                    break
                else:
                    print("Unsat")

                # print(f"{c_parts}: {counter}")
                # model = slv.get_model()
                #
                # slv.add_clause([-pool.id(f"p_0_{cp}_{n}") for cp in range(0, c_parts) for n in range(1, len(g.nodes)+1) if model[pool.id(f"p_0_{cp}_{n}") + 1] > 0])
                break

            if counter == 0:
                print(f"Failed {c_parts}")
            else:
                print(f"Success {c_parts}")

    print("Done")


g = grid_2d_graph(7, 7)

rev = {x: i+1 for i, x in enumerate(g.nodes)}

gp = nx.Graph()
for u, v in g.edges:
    gp.add_edge(rev[u], rev[v])

encode_partition(gp, 3, 5, 0)
