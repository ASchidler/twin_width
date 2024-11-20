import sys
from collections import defaultdict

import networkx as nx

import parser
import encoding
import encoding2
import time

from pysat.solvers import Glucose4, Cadical195

inp_file = sys.argv[1]
graph_file = sys.argv[2]
bound = int(sys.argv[3])


class SubGraph:
    """Represents a subgraph for subgraph lower bounds."""
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

twins = set()

type_times = defaultdict(int)
last_backtrack = 0

c_partition_tmp = [list(g.nodes) for x in range(0, max(g.nodes)+1)]
c_partition_tmp[0].clear()


def get_partition(l_contractions):
    """Converts a sequence of contractions to a partition representation that is hashable"""
    for cei, ce in enumerate(c_partition_tmp[1:]):
        ce.clear()
        ce.append(cei+1)

    for l_cc in l_contractions:
        c_partition_tmp[l_cc[1]].extend(c_partition_tmp[l_cc[0]])
        # Sorting here is faster, since Tim Sort performs a merge
        c_partition_tmp[l_cc[1]].sort(reverse=True)
        c_partition_tmp[l_cc[0]].clear()
    # > 1 makes the representation smaller and the whole algorithm faster
    return str([el for x in c_partition_tmp for el in x if len(x) > 1])


# First scan for requested partitions
with open(inp_file) as proof_file:
    for line_no, cl in enumerate(proof_file):
        if cl.startswith("C "):
            fd = cl.strip()[2:].split(":")
            if fd[1] == "C" or fd[1] == "O":
                contractions = [x[1:-1].split(",") for x in fd[0].split(" ")]
                contractions = [(int(x[0]) + 1, int(x[1]) + 1) for x in contractions]
                if fd[1] == "C":
                    # Simply add partition to set
                    requested_partitions.add(get_partition(contractions))
                if fd[1] == "O":
                    # Here, we have to rorder the contractions, i.e., move the last contraction forward
                    swap_idx = int(fd[2])
                    last_contr = contractions[-1]
                    contractions = contractions[:swap_idx]
                    contractions.append(last_contr)
                    requested_partitions.add(get_partition(contractions))

print("Loaded cache requirements")
type_times["Cache Init"] = time.time() - overall_start

# Now start main scan
with Glucose4() as propagation_solver:
    with Cadical195() as overall_solver:
        oenc = encoding2.TwinWidthEncoding2(g, cubic=2, break_g_symmetry=False)
        oformula = oenc.encode(g, bound)
        overall_solver.append_formula(oformula)
        propagation_solver.append_formula(oformula)
        last_contractions = []
        step_vars = []

        def disregard_contractions(target_contractions, target_partition=None):
            # Only add for cached entries!
            if len(oenc.ord[len(target_contractions)+1 + len(twins)]) > 0:
                clause = []
                for contraction_index, contraction in enumerate(target_contractions):
                    clause.extend([-oenc.ord[contraction_index + 1 + len(twins)][contraction[0]], -oenc.merge[contraction_index + 1 + len(twins)][contraction[1]]])
                overall_solver.add_clause(clause)

            # Mark the partition as seen
            if len(requested_partitions) > 0:
                requested_partitions.discard(target_partition if target_partition is not None else get_partition(target_contractions))

        def perform_backtracking(idx_to, target_contractions, line_nnumber):
            # Check if twins exist in any of the trigraphs we visit during backtracking
            backtrack_twins = set()
            if len(target_contractions) > 0:
                parts = [x for x in range(0, max(g.nodes) + 1)]

                for ccontr_id, ccontr in enumerate(target_contractions):
                    if ccontr_id >= idx_to:
                        cx, cy = ccontr
                        pos = [set(), set()]
                        neg = [set(), set()]
                        all_nodes = set(g.nodes)

                        for z in [ci for ci, ct in enumerate(parts) if ct == cx]:
                            pos[0].update({parts[x] for x in g.neighbors(z)})
                            neg[0].update({parts[x] for x in (all_nodes - set(g.neighbors(z)))})

                        for z in [ci for ci, ct in enumerate(parts) if ct == cy]:
                            pos[1].update({parts[x] for x in g.neighbors(z)})
                            neg[1].update({parts[x] for x in all_nodes - set(g.neighbors(z))})

                        creds = [pos[0] & neg[0], pos[1] & neg[1]]
                        pos[0] = {x for x in pos[0] if x == parts[x]}
                        pos[1] = {x for x in pos[1] if x == parts[x]}
                        pos[0] -= creds[0]
                        pos[1] -= creds[1]

                        # Remove the contracted and contraction vertex from sets
                        for centries in [*creds, *pos]:
                            centries.difference_update(target_contractions[ccontr_id])

                        new_reds = pos[0] ^ pos[1]
                        new_reds |= creds[0] | creds[1]
                        new_reds.discard(cx)
                        new_reds.discard(cy)

                        if creds[0].issuperset(new_reds) or creds[1].issuperset(new_reds):
                            backtrack_twins.add(ccontr_id)
                    parts = [ci if ci != ccontr[0] else ccontr[1] for ci in parts]

            type_times["Twin Count"] += len(backtrack_twins)

            for cidx in range(len(target_contractions) - 2, idx_to - 1, -1):
                target_contractions = target_contractions[:cidx + 1]
                if cidx + 1 not in backtrack_twins:
                    if len(oenc.ord[len(target_contractions)+1 + len(twins)]) > 0:
                        # Run incremental call
                        asspts = []

                        for ccidx, ccontr in enumerate(target_contractions):
                            asspts.extend([oenc.ord[ccidx + 1 + len(twins)][ccontr[0]], oenc.merge[ccidx + 1 + len(twins)][ccontr[1]]])

                        if overall_solver.solve(asspts):
                            print(f"ERROR (Line: {line_nnumber+1}): Not exceeded in backtracking.")
                            exit(1)

                disregard_contractions(target_contractions)

        with open(inp_file) as proof_file:
            for line_no, cl in enumerate(proof_file):
                line_start = time.time()

                if cl.startswith("T "):
                    # Check that the two vertices are really twins
                    a, b = [int(x)+1 for x in cl[2:].strip().split(" ")]
                    if not g.has_node(a) or not g.has_node(b):
                        print(f"ERROR (Line: {line_no + 1}): Twin node does not exist. {cl.strip()}")
                        exit(1)
                    elif len(((set(g.neighbors(a)) ^ set(g.neighbors(b))) - {a, b}) - twins) > 0:
                        print(f"ERROR (Line: {line_no + 1}): Not twins. {cl.strip()}")
                        exit(1)

                    twins.add(a)
                    overall_solver.add_clause([oenc.ord[len(twins)][a]])
                    overall_solver.add_clause([oenc.merge[len(twins)][a]])
                    propagation_solver.add_clause([oenc.ord[len(twins)][a]])
                    propagation_solver.add_clause([oenc.merge[len(twins)][a]])
                else:
                    fd = cl.strip()[2:].split(":")
                    contractions = [x[1:-1].split(",") for x in fd[0].split(" ")]
                    contractions = [(int(x[0]) + 1, int(x[1]) + 1) for x in contractions]
                    contraction_partition = None

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

                        # Run SAT encoding on subtrigraph
                        enc = encoding.TwinWidthEncoding(use_sb_static_full=False, use_sb_static=False, use_sb_red=False)
                        formula = enc.encode(sg, bound, None, None, reds=reds)
                        assert all(enc.node_map[x] == x for x in (cy for cx in reds for cy in cx))

                        with Cadical195() as slv:
                            slv.append_formula(formula)

                            for cv in enc.get_card_vars(bound):
                                slv.add_clause([cv])
                            if slv.solve():
                                print(f"ERROR (Line: {line_no+1}): Subgraph verification failed. {cl.strip()}")
                                exit(1)

                        sub_graphs.append(SubGraph(nodes, ireds))
                        target_idx = int(fd[-2])
                        while len(contractions) > target_idx:
                            disregard_contractions(contractions)
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
                                before = time.time()
                                perform_backtracking(idx, last_contractions, line_no)
                                last_backtrack = time.time() - before

                        if fd[1] == "C":
                            contraction_partition = get_partition(contractions)
                            # Here, we just have to check if the partition is still marked as unseen.
                            if len(requested_partitions) > 0 and contraction_partition in requested_partitions:
                                print(f"ERROR (Line: {line_no+1}): Cached entry not found. {cl.strip()}")
                                exit(1)
                        elif fd[1] == "O":
                            # First, we have to verify that the modified partition has been seen.
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
                                assumptions.extend([oenc.ord[cci + 1 + len(twins)][cc[0]], oenc.merge[cci + 1][cc[1]]])
                            if not propagation_solver.propagate(assumptions):
                                print(f"ERROR (Line: {line_no}): Invalid Reordering. {cl.strip()}")
                                exit(1)

                        elif fd[1] == "E":
                            # Ensure that the contraction sequence indeed exceeds the bound.
                            if len(oenc.ord[len(contractions)+1 + len(twins)]) > 0:
                                assumptions = []
                                for cci, cc in enumerate(contractions):
                                    if cci+1 < len(oenc.ord):
                                        assumptions.extend([oenc.ord[cci+1 + len(twins)][cc[0]], oenc.merge[cci+1 + len(twins)][cc[1]]])

                                if overall_solver.solve(assumptions):
                                    print(f"ERROR (Line: {line_no+1}): Not exceeded. {cl.strip()}")
                                    exit(1)
                        elif fd[1] == "G":
                            # Ensure that there is a known lower bounding subgraph
                            nodes = set(g.nodes)
                            removed = {x for x, _ in contractions}

                            list_partition = [{i} if i in g.nodes else set() for i in range(0, max(g.nodes)+1)]
                            for cc in contractions:
                                nodes.discard(cc[0])
                                list_partition[cc[1]].update(list_partition[cc[0]])
                                list_partition[cc[0]].clear()

                            found = False
                            target_subgraph = None
                            for csg in reversed(sub_graphs):
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

                        disregard_contractions(contractions, contraction_partition)
                        last_contractions.clear()
                        last_contractions.extend(contractions)
                    else:
                        print(f"ERROR (Line: {line_no+1}): Unknown line start. {cl.strip()}")
                        exit(1)

                print(f"Verified Line {line_no+1} ({round(time.time() - line_start,2)}s/{round(time.time() - overall_start,2)}s)")
                type_times[fd[1] if cl.startswith("C") else cl[0]] += time.time() - line_start - last_backtrack
                type_times["Backtrack"] += last_backtrack
                last_backtrack = 0

        # Run full encoding with information we gathered
        perform_backtracking(-1, last_contractions, -1)
        print(f"Successful ({round(time.time() - overall_start,2)}s)")
        print("Time Spent:")
        for ck in sorted(type_times.keys()):
            print(f"{ck}: {round(type_times[ck], 2)}")
