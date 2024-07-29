import sys
import networkx as nx

import parser
import encoding
import encoding2
import time

from pysat.solvers import Glucose4, Cadical195

inp_file = sys.argv[1]
graph_file = sys.argv[2]
bound = int(sys.argv[3])

use_twins = True
use_conditional = False


class SubGraph:
    def __init__(self, nodes, reds):
        self.nodes = nodes
        self.reds = reds


overall_start = time.time()

g = parser.parse(graph_file)[0]
rg = nx.Graph()
for cn in rg.nodes:
    rg.add_node(cn)
sub_graphs = []
node_map = [None for _ in range(0, len(g.nodes) + 1)]
requested_partitions = set()

twin_offset = 0
twins = set()


def get_partition(l_contractions):
    c_partition = [[x] if x in g.nodes else [] for x in range(0, max(g.nodes)+1)]

    for l_cc in l_contractions:
        c_partition[l_cc[1]].extend(c_partition[l_cc[0]])
        c_partition[l_cc[0]].clear()
    c_partition = [el for x in c_partition for el in sorted(x, reverse=True) if len(x) > 1]
    return ",".join((str(x) for x in c_partition))


# First scan for requested partitions
# with open(inp_file) as proof_file:
#     for line_no, cl in enumerate(proof_file):
#         if cl.startswith("C "):
#             fd = cl.strip()[2:].split(":")
#             if fd[1] == "C":
#                 contractions = [x[1:-1].split(",") for x in fd[0].split(" ")]
#                 contractions = [(int(x[0]) + 1, int(x[1]) + 1) for x in contractions]
#                 partition = get_partition(contractions)
#                 requested_partitions.add(partition)
#             if fd[1] == "O":
#                 contractions = [x[1:-1].split(",") for x in fd[0].split(" ")]
#                 contractions = [(int(x[0]) + 1, int(x[1]) + 1) for x in contractions]
#                 swap_idx = int(fd[2])
#                 last_contr = contractions[-1]
#                 contractions = contractions[:swap_idx]
#                 contractions.append(last_contr)
#                 partition = get_partition(contractions)
#                 requested_partitions.add(partition)

print("Loaded cache requirements")

# Now start main scan
with Glucose4() as propagation_solver:
    with Cadical195() as overall_solver:
        oenc = encoding2.TwinWidthEncoding2(g, cubic=2, break_g_symmetry=False)
        oformula = oenc.encode(g, bound, conditional_steps=use_conditional)
        overall_solver.append_formula(oformula)
        propagation_solver.append_formula(oformula)
        last_contractions = []
        step_vars = []

        def are_twins(target_contractions):
            if not use_twins:
                return False

            # Check if twin
            if len(target_contractions) > 0:
                # gc = g.copy()
                # for x1, x2 in gc.edges:
                #     gc[x1][x2]['red'] = False
                #
                # for c1, c2 in target_contractions[:-1]:
                #     c1nb = set(gc.neighbors(c1))
                #     c2nb = set(gc.neighbors(c2))
                #
                #     for cx in c1nb:
                #         if cx != c2:
                #             gc.add_edge(c2, cx, red=(cx not in c2nb or gc[cx][c1]['red'] or gc[cx][c2]['red']))
                #     for cx in c2nb:
                #         if cx not in c1nb:
                #             gc[c2][cx]['red'] = True
                #     gc.remove_node(c1)
                #
                # c1, c2 = target_contractions[-1]
                # if len([x for x in gc.neighbors(c1) if gc[x][c1]['red']]) < len([x for x in gc.neighbors(c1) if gc[x][c1]['red']]):
                #     c1, c2 = c2, c1
                #
                # is_twin = True
                # for cx in gc.neighbors(c1):
                #     if not gc[cx][c1]['red'] and cx != c2:
                #         if not gc.has_edge(cx, c2) or gc[cx][c2]['red']:
                #             is_twin = False
                # for cx in gc.neighbors(c2):
                #     if cx != c1 and not gc.has_edge(cx, c1):
                #         is_twin = False
                #
                # if is_twin:
                #     print("Twin")

                cx, cy = target_contractions[-1]
                parts = [x for x in range(0, max(g.nodes)+1)]
                for ccontr in target_contractions[:-1]:
                    parts = [ci if ci != ccontr[0] else ccontr[1] for ci in parts]

                pos = [set(), set()]
                neg = [set(), set()]
                all_nodes = set(g.nodes)

                for z in [ci for ci, ct in enumerate(parts) if ct == cx]:
                    pos[0].update({parts[x] for x in g.neighbors(z)})
                    neg[0].update({parts[x] for x in (all_nodes - set(g.neighbors(z)))})
                for z in [ci for ci, ct in enumerate(parts) if ct == cy]:
                    pos[1].update({parts[x] for x in g.neighbors(z)})
                    neg[1].update({parts[x] for x in all_nodes - set(g.neighbors(z))})

                reds = [pos[0] & neg[0], pos[1] & neg[1]]
                pos[0] = {x for x in pos[0] if x == parts[x]}
                pos[1] = {x for x in pos[1] if x == parts[x]}
                pos[0] -= reds[0]
                pos[1] -= reds[1]

                for cl in [*reds, *pos]:
                    cl.difference_update(target_contractions[-1])

                new_reds = pos[0] ^ pos[1]
                new_reds |= reds[0] | reds[1]
                new_reds.discard(cx)
                new_reds.discard(cy)

                if reds[0].issuperset(new_reds) or reds[1].issuperset(new_reds):
                    return True
                return False

        def disregard_contractions(target_contractions, add_clause=True, negate_last=False):
            # Only add for cached entries!
            if add_clause and len(oenc.ord[len(target_contractions)+1 + twin_offset]) > 0:
                clause = []
                for cci, cc in enumerate(target_contractions):
                    if not negate_last or cci < len(target_contractions) - 1:
                        clause.extend([-oenc.ord[cci + 1 + twin_offset][cc[0]], -oenc.merge[cci + 1 + twin_offset][cc[1]]])
                if not negate_last:
                    overall_solver.add_clause(clause)
                else:
                    if use_conditional:
                        clause.append(-oenc.conditional_steps[len(target_contractions)])
                    overall_solver.add_clause([*clause, oenc.ord[len(target_contractions) + twin_offset][target_contractions[-1][0]]])
                    overall_solver.add_clause([*clause, oenc.merge[len(target_contractions) + twin_offset][target_contractions[-1][1]]])

            if len(requested_partitions) > 0:
                partition = get_partition(target_contractions)
                requested_partitions.discard(partition)

        def perform_backtracking(idx_to, target_contractions, line_nnumber):
            for cidx in range(len(target_contractions) - 2, idx_to - 1, -1):
                target_contractions = target_contractions[:cidx + 1]

                if len(oenc.ord[len(target_contractions)+1 + twin_offset]) > 0:
                    if are_twins(target_contractions):
                        disregard_contractions(target_contractions, True, True)
                        disregard_contractions(target_contractions, True, False)
                    else:
                        # Run incremental call
                        asspts = []

                        for ccidx, ccontr in enumerate(target_contractions):
                            asspts.extend([oenc.ord[ccidx + 1 + twin_offset][ccontr[0]], oenc.merge[ccidx + 1 + twin_offset][ccontr[1]]])

                        if not use_conditional:
                            if overall_solver.solve(asspts):
                                print(f"ERROR (Line: {line_nnumber+1}): Not exceeded in backtracking. {cl.strip()}")
                                exit(1)
                        else:
                            for cstep in range(len(target_contractions)+1+twin_offset, len(oenc.conditional_steps)):
                                if overall_solver.solve([*asspts, *oenc.conditional_steps[1:cstep], *[-x for x in oenc.conditional_steps[cstep:]]]):
                                    if cstep - len(target_contractions) - twin_offset > 2:
                                        add_ons = []
                                        st = set(overall_solver.get_model())
                                        for cx in range(len(target_contractions), cstep):
                                            merges = [(k, v) for k, v in oenc.pool.id2obj.items() if
                                                      v.startswith(f"merge{cx}_") and k in st]
                                            ords = [(k, v) for k, v in oenc.pool.id2obj.items() if
                                                    v.startswith(f"ord{cx}") and k in st]
                                            assert len(merges) == 1 and len(ords) == 1
                                            add_ons.append(f"{ords[0][1].split('_')[-1]} {merges[0][1].split('_')[-1]}")
                                        print(
                                            f"{target_contractions} {add_ons}")

                                    if cstep == len(oenc.conditional_steps) - 1:
                                        print(f"ERROR (Line: {line_nnumber+1}): Not exceeded in backtracking. {cl.strip()}")
                                        exit(1)
                                else:
                                    print(cstep - len(target_contractions) - twin_offset)
                                    break

                disregard_contractions(target_contractions, True)

        with open(inp_file) as proof_file:
            for line_no, cl in enumerate(proof_file):
                line_start = time.time()

                if cl.startswith("T "):
                    a, b = [int(x)+1 for x in cl[2:].strip().split(" ")]
                    if not g.has_node(a) or not g.has_node(b):
                        print(f"ERROR (Line: {line_no + 1}): Twin node does not exist. {cl.strip()}")
                        exit(1)
                    elif len((set(g.neighbors(a)) ^ set(g.neighbors(b)) - {a, b}) - twins) > 0:
                        print(f"ERROR (Line: {line_no + 1}): Not twins. {cl.strip()}")
                        exit(1)
                    twins.add(a)
                    twin_offset += 1
                    overall_solver.add_clause([oenc.ord[twin_offset][a]])
                    overall_solver.add_clause([oenc.merge[twin_offset][a]])
                    propagation_solver.add_clause([oenc.ord[twin_offset][a]])
                    propagation_solver.add_clause([oenc.merge[twin_offset][a]])
                else:
                    fd = cl.strip()[2:].split(":")
                    contractions = [x[1:-1].split(",") for x in fd[0].split(" ")]
                    contractions = [(int(x[0]) + 1, int(x[1]) + 1) for x in contractions]

                    if int(fd[-1]) < bound:
                        print(f"ERROR (Line: {line_no+1}): Bound too low. {cl.strip()}")
                        exit(1)

                    # Found a subgraph, verify that it has indeed exceeding twin-width
                    if cl.startswith("S "):
                        # Extract nodes and reds edges from proof
                        nodes = {int(x) + 1 for x in fd[1].split(" ")}
                        ireds = [[int(z) + 1 for z in x[1:-1].split(",")] for x in fd[2].split(" ")]
                        node_map = [None for _ in node_map]

                        # Map this into a new graph and construct
                        for cni, cn in enumerate(nodes):
                            node_map[cn] = cni + 1

                        reds = [(node_map[x[0]], node_map[x[1]]) for x in ireds]

                        sg = nx.Graph()
                        for i in range(1, len(nodes)+1):
                            sg.add_node(i)
                        for n1, n2 in g.edges:
                            if n1 in nodes and n2 in nodes:
                                sg.add_edge(node_map[n1], node_map[n2])

                        # enc = encoding2.TwinWidthEncoding2(sg, cubic=2, break_g_symmetry=False)
                        # with Cadical195() as slv:
                        #     slv.append_formula(enc.encode(sg, bound, reds=reds))
                        #
                        #     if slv.solve():
                        #         steps = len(sg.nodes) - bound
                        #         print(f"{enc.decode(slv.get_model(), sg, bound, steps, False)}")
                        #         print(f"ERROR (Line: {line_no+1}): Subgraph verification failed. {cl.strip()}")
                        #         exit(1)

                        # Run SAT encoding on subtrigraph
                        enc = encoding.TwinWidthEncoding(use_sb_static_full=False, use_sb_static=False, use_sb_red=False)
                        formula = enc.encode(sg, bound, None, None, reds=reds)
                        assert all(enc.node_map[x] == x for x in (cy for cx in reds for cy in cx))

                        any_solved = False
                        with Cadical195() as slv:
                            slv.append_formula(formula)
                            for c_bound in (bound, 0, -1):
                                for cv in enc.get_card_vars(c_bound):
                                    slv.add_clause([cv])
                                if not slv.solve():
                                    break
                                else:
                                    any_solved = True

                                print(enc.decode(slv.get_model(), sg, bound, True, reds=reds))
                            if any_solved:
                                print(f"ERROR (Line: {line_no+1}): Subgraph verification failed. {cl.strip()}")
                                exit(1)

                        sub_graphs.append(SubGraph(nodes, ireds))
                        target_idx = int(fd[-2])
                        while len(contractions) > target_idx:
                            disregard_contractions(contractions, True)
                            contractions.pop()
                    # Other option, we found a search tree leaf
                    elif cl.startswith("C "):
                        # Handle backtracking if necessary
                        idx = 0
                        if len(last_contractions) > 0:
                            # Find out where the last and the current contraction sequence diverge
                            while idx < len(last_contractions):
                                if idx == len(contractions):
                                    break
                                if last_contractions[idx][0] != contractions[idx][0] or last_contractions[idx][1] != contractions[idx][1]:
                                    break
                                idx += 1

                            # If they do not diverge, there is something wrong with the format
                            if idx == len(last_contractions) and idx == len(contractions):
                                continue

                            assert idx < len(last_contractions)

                            # Check if we backtrack by more than one
                            if idx < len(last_contractions) - 1:
                                perform_backtracking(idx, last_contractions, line_no)

                        add_clause = True
                        if are_twins(contractions):
                            disregard_contractions(contractions, True, True)

                        if fd[1] == "C":
                            if len(requested_partitions) > 0 and get_partition(contractions) in requested_partitions:
                                print(f"ERROR (Line: {line_no+1}): Cached entry not found. {cl.strip()}")
                                exit(1)
                        elif fd[1] == "O":
                            target_idx = int(fd[2])
                            # Adapt contraction sequence as used in the solver
                            cut_contraction = contractions[-1]
                            cut_contractions = contractions[target_idx:-1]
                            adapted_contractions = contractions[:target_idx]
                            adapted_contractions.append(cut_contraction)
                            adapted_partition = get_partition(adapted_contractions)

                            # First check that the cache entry exists
                            if adapted_partition in requested_partitions:
                                print(f"ERROR (Line: {line_no+1}): Order Cached entry not found. {cl.strip()}")
                                exit(1)

                            # Now ensure that the modified contraction sequence does not exceed the bound
                            assumptions = []
                            adapted_contractions.extend(cut_contractions)
                            for cci, cc in enumerate(adapted_contractions):
                                assumptions.extend([oenc.ord[cci + 1 + twin_offset][cc[0]], oenc.merge[cci + 1][cc[1]]])
                            if use_conditional:
                                assumptions.extend(oenc.conditional_steps[1:])
                            if not propagation_solver.propagate(assumptions):
                                print(f"ERROR (Line: {line_no}): Invalid Reordering. {cl.strip()}")
                                exit(1)

                        elif fd[1] == "E":
                            # Ensure that the contraction sequence indeed exceeds the bound.
                            if len(oenc.ord[len(contractions)+1 + twin_offset]) > 0:
                                assumptions = []
                                for cci, cc in enumerate(contractions):
                                    if cci+1 < len(oenc.ord):
                                        assumptions.extend([oenc.ord[cci+1 + twin_offset][cc[0]], oenc.merge[cci+1 + twin_offset][cc[1]]])
                                if use_conditional:
                                    assumptions.extend(oenc.conditional_steps[1:])
                                if overall_solver.solve(assumptions):
                                    print(f"ERROR (Line: {line_no+1}): Not exceeded. {cl.strip()}")
                                    exit(1)
                                add_clause = False
                        elif fd[1] == "G":
                            # Ensure that there is a known lower bounding subgraph
                            nodes = set(g.nodes)

                            list_partition = [{i} if i in g.nodes else set() for i in range(0, max(g.nodes)+1)]
                            for cc in contractions:
                                nodes.discard(cc[0])
                                list_partition[cc[1]].update(list_partition[cc[0]])
                                list_partition[cc[0]].clear()

                            found = False
                            target_subgraph = None
                            for csg in sub_graphs:
                                if all(len(x & csg.nodes) <= 1 for x in list_partition):
                                    cmap = {next(iter(list_partition[i] & csg.nodes)): i for i in range(0, len(list_partition)) if len(list_partition[i] & csg.nodes) > 0}
                                    found = True
                                    for x, y in csg.reds:
                                        x = cmap[x]
                                        y = cmap[y]

                                        neg_found = False
                                        pos_found = False
                                        for n1 in list_partition[x]:
                                            for n2 in list_partition[y]:
                                                if g.has_edge(n1, n2):
                                                    pos_found = True
                                                else:
                                                    neg_found = True
                                            if pos_found and neg_found:
                                                break
                                        if not pos_found or not neg_found:
                                            found = False
                                if found:
                                    target_subgraph = csg
                                    break
                            if not found:
                                print(f"ERROR (Line: {line_no+1}): Subgraph not found. {cl.strip()}")
                                exit(1)
                        else:
                            print(f"ERROR (Line: {line_no+1}): Unknown reason. {cl.strip()}")
                            exit(1)

                        disregard_contractions(contractions, True)
                        last_contractions.clear()
                        last_contractions.extend(contractions)
                    else:
                        print(f"ERROR (Line: {line_no+1}): Unknown line start. {cl.strip()}")
                        exit(1)
                print(f"Verified Line {line_no+1} ({round(time.time() - line_start,2)}s/{round(time.time() - overall_start,2)}s)")

        # Run full encoding with information we gathered
        perform_backtracking(-1, last_contractions, -1)
        print(f"Successful ({round(time.time() - overall_start,2)}s)")
