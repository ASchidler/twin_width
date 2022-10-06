import sys
import time

from networkx import Graph
from pysat.card import CardEnc, EncType, ITotalizer
from pysat.formula import CNF, IDPool
from threading import Timer
import tools


class TwinWidthEncoding2:
    def __init__(self, g, card_enc=EncType.mtotalizer):
        self.pool = None
        self.g = g
        self.card_enc = card_enc

    def remap_graph(self, g):
        self.node_map = {}
        cnt = 1
        gn = Graph()

        for u, v in sorted(g.edges()):
            if u not in self.node_map:
                self.node_map[u] = cnt
                cnt += 1
            if v not in self.node_map:
                self.node_map[v] = cnt
                cnt += 1

            gn.add_edge(self.node_map[u], self.node_map[v])

        return gn

    def splits(self, g1, g2):
        return self.pool.id(f"splits_{g1}_{g2}")

    def groups(self, g1, node):
        return self.pool.id("group_{}_{}".format(g1, node))

    def red_edge(self, g1, g2, time):
        return self.pool.id("red_edge_{}_{}_{}".format(min(g1, g2), max(g1, g2), time))

    def group_edge(self, g1, node, time):
        return self.pool.id("group_edge_{}_{}_{}". format(g1, node, time))

    def encode(self, g, d):
        groups = d + 2
        g = self.remap_graph(g)
        self.pool = IDPool()
        self.formula = CNF()
        n = len(g)

        self.encode_partition(g, d, groups)

        for cgroups in range(d+2+1, n+1):
            self.formula.append([self.groups(cgroups, i) for i in range(1, n + 1)])
            self.formula.append([self.splits(cgroups, cg) for cg in range(1, cgroups)])
            self.formula.extend(CardEnc.atmost([self.splits(cgroups, cg) for cg in range(1, cgroups)]))

            alo_clause = []
            for i in range(1, n + 1):
                for cg in range(1, cgroups):
                    # Only members of the split group can be split
                    clause = [-self.groups(cgroups, i), -self.splits(cgroups, cg), self.groups(cg, i)]
                    self.formula.append(clause)
                    for cg2 in range(cg+1, cgroups):
                        clause = list(clause)
                        clause.pop()
                        clause.append(-self.groups(cg2, i))
                        self.formula.append(clause)

                    self.formula.append([-self.splits(cgroups, cg),
                                         -self.pool.id(f"aux_ng_{cgroups}_{i}"),
                                         self.groups(cg, i)
                                         ])

                    self.formula.append([-self.pool.id(f"aux_ng_{cgroups}_{i}"),
                                         -self.groups(cgroups, i)])

                alo_clause.append(self.pool.id(f"aux_ng_{cgroups}_{i}"))
            self.formula.append(alo_clause)
            #
            #
            # Encode outgoing edges
            for gr in range(1, cgroups + 1):
                for i in range(1, n + 1):
                    if gr < cgroups:
                        self.formula.append([-self.group_edge(gr, i, cgroups-1), self.splits(cgroups, gr), self.group_edge(gr, i, cgroups)])

                    inb = set(g.neighbors(i))
                    for j in range(i + 1, n + 1):
                        jnb = set(g.neighbors(j))
                        jnb.discard(i)
                        diff = jnb ^ inb  # Symmetric difference
                        diff.discard(j)

                        for k in diff:
                            if gr == cgroups:
                                self.formula.append([-self.groups(cgroups, i), -self.groups(cgroups, j), self.group_edge(gr, k, cgroups)])
                            else:
                                clause = [-self.splits(cgroups, gr), -self.group_edge(gr, k, cgroups-1),
                                          -self.groups(gr, i), -self.groups(gr, j), self.group_edge(gr, k, cgroups)]
                                group_appendage1 = [self.groups(gr2, i) for gr2 in range(gr+1, cgroups+1)]
                                group_appendage2 = [self.groups(gr2, j) for gr2 in range(gr+1, cgroups+1)]
                                clause.extend(group_appendage1)
                                clause.extend(group_appendage2)
                                self.formula.append(clause)

                for gr2 in range(1, cgroups + 1):
                    if gr == gr2:
                        continue

                    if gr2 > gr and gr2 != cgroups:
                        self.formula.append([-self.red_edge(gr, gr2, cgroups-1), self.splits(cgroups, gr), self.splits(cgroups, gr2),
                                             self.red_edge(gr, gr2, cgroups)])

                    for i in range(1, n + 1):
                        if gr2 == cgroups:
                            self.formula.append([-self.group_edge(gr, i, cgroups), -self.groups(gr2, i),
                                                 self.red_edge(gr, gr2, cgroups)])
                        elif gr == cgroups:
                            clause = [-self.group_edge(gr, i, cgroups), -self.groups(gr2, i),
                                                 self.red_edge(gr, gr2, cgroups)]
                            group_appendage = [self.groups(gr3, i) for gr3 in range(gr2 + 1, cgroups + 1)]
                            clause.extend(group_appendage)
                            self.formula.append(clause)
                        else:
                            clause = [-self.splits(cgroups, gr), -self.group_edge(gr, i, cgroups), -self.groups(gr2, i),
                                                self.red_edge(gr, gr2, cgroups)]
                            group_appendage1 = [self.groups(gr3, i) for gr3 in range(gr2 + 1, cgroups + 1)]
                            clause.extend(group_appendage1)
                            self.formula.append(clause)
                            clause = [-self.splits(cgroups, gr2), -self.group_edge(gr, i, cgroups), -self.groups(gr2, i),
                                      self.red_edge(gr, gr2, cgroups)]
                            group_appendage1 = [self.groups(gr3, i) for gr3 in range(gr2 + 1, cgroups + 1)]
                            clause.extend(group_appendage1)
                            self.formula.append(clause)

            for gr in range(1, cgroups + 1):  # As last one is the root, no counter needed
                cvars = [self.red_edge(gr, gr2, cgroups) for gr2 in range(1, cgroups + 1) if gr != gr2]

                with ITotalizer(cvars, ubound=d, top_id=self.pool.id(f"totalizer{gr}_{cgroups}")) as tot:
                    self.formula.extend(tot.cnf)
                    self.pool.occupy(self.pool.top - 1, tot.top_id)
                    self.formula.append([-tot.rhs[d]])

        return self.formula

    def encode_partition(self, g, d, groups=None):
        n = len(g.nodes)

        # Each group has at least one member
        for gr in range(1, groups+1):
            self.formula.append([self.groups(gr, i) for i in range(1, n+1)])

        # Each vertex is member of exactly one group
        for i in range(1, n+1):
            self.formula.append([self.groups(gr, i) for gr in range(1, min(groups+1, i+1))])
            self.formula.extend(CardEnc.atmost([self.groups(gr, i) for gr in range(1, groups+1)], vpool=self.pool))

        # Encode outgoing edges
        for gr in range(1, groups+1):
            for i in range(1, n+1):
                inb = set(g.neighbors(i))
                for j in range(i+1, n+1):
                    jnb = set(g.neighbors(j))
                    jnb.discard(i)
                    diff = jnb ^ inb  # Symmetric difference
                    diff.discard(j)

                    for k in diff:
                        self.formula.append([-self.groups(gr, i),
                                             -self.groups(gr, j),
                                             self.group_edge(gr, k, groups)])

            for gr2 in range(1, groups+1):
                if gr == gr2:
                    continue

                for i in range(1, n+1):
                    self.formula.append([-self.group_edge(gr, i, groups),
                                         -self.groups(gr2, i),
                                         self.red_edge(min(gr, gr2), max(gr, gr2), groups)])


        for gr in range(1, groups+1):  # As last one is the root, no counter needed
            cvars = [self.red_edge(min(gr, gr2), max(gr, gr2), groups) for gr2 in range(1, groups+1) if gr != gr2]
            with ITotalizer(cvars, ubound=d, top_id=self.pool.id(f"totalizer{gr}_{groups}")) as tot:
                self.formula.extend(tot.cnf)
                self.pool.occupy(self.pool.top - 1, tot.top_id)
                self.formula.append([-tot.rhs[d]])

        return self.formula

    def run(self, g, solver, start_bound, verbose=True, check=True, lb=0, timeout=0, i_od=None, i_mg=None, steps_limit=None):
        start = time.time()

        with solver() as slv:
            formula = self.encode(g, start_bound)

            slv.append_formula(formula)

            if verbose:
                print(f"Created encoding in {time.time() - start} {len(formula.clauses)}/{formula.nv}")
                print(f"Solver clauses {slv.nof_clauses()}/{slv.nof_vars()}")

            if slv.solve() if timeout == 0 else slv.solve_limited():
                if verbose:
                    print(f"Found {start_bound}")
                self.decode(slv.get_model(), g, start_bound, True)
                # cb, od, mg = self.decode(slv.get_model(), g, i, steps, verbose)
            else:
                if verbose:
                    print(f"Failed {start_bound}")

            if verbose:
                print(f"Finished cycle in {time.time() - start}")

        if verbose:
            print(f"Finished in {time.time() - start}")

    def decode(self, model, g, d, verbose):
        g = g.copy()
        model = {abs(x): x > 0 for x in model}
        unmap = {}
        for u, v in self.node_map.items():
            unmap[v] = u

        cgroups = [set() for _ in range(0, len(g.nodes))]

        for gr in range(1, len(g.nodes)):
            for i in range(1, len(g.nodes) + 1):
                if model[self.groups(gr, i)]:
                    cgroups[gr].add(i)

        for gr in cgroups[1:]:
            if len(gr) == 0:
                print(f"Empty group")

        for i in range(1, len(g.nodes)+1):
            cnt = sum(1 if i in x else 0 for x in cgroups)
            if cnt == 0:
                print(f"Node {i} is in no group")

        splits = {}
        for x in range(d+2+1, len(g.nodes)):
            for cg in range(1, x):
                if model[self.splits(x, cg)]:
                    if x in splits:
                        print("Double split")
                    splits[x] = cg

        for i, ce in enumerate(cgroups):
            if i in splits:
                print(f"{i}/{splits[i]}: {ce}")
            else:
                print(f"{i}/x: {ce}")

        c_max = 0
        for groups in range(d+2, len(g.nodes)):
            cgr = [[]]
            for gr1 in range(1, groups + 1):
                ggr1 = set(cgroups[gr1])
                for ggr2 in cgroups[gr1 + 1:groups+1]:
                    ggr1 -= ggr2
                cgr.append(ggr1)

            for gr1 in range(1, groups+1):
                ggr1 = cgr[gr1]

                shared = set(range(1, len(g.nodes) + 1))
                together = set()
                red_cnt = 0
                for i in ggr1:
                    cnb = {self.node_map[x] for x in g.neighbors(unmap[i])}
                    shared &= cnb
                    together |= cnb

                non_shared = together - shared - ggr1

                for gr2 in range(1, groups+1):
                    if gr1 != gr2 and len(non_shared & cgr[gr2]) > 0:
                        red_cnt += 1
                        if not model[self.red_edge(min(gr1, gr2), max(gr1, gr2), groups)]:
                            print(f"{groups}: Missing red edge ({gr1}, {gr2})")
                            if all(not model[self.group_edge(gr1, k, groups)] for k in cgroups[gr2]):
                                print(f"{groups}: Missing group edge  ({gr1}, {gr2})")

                if red_cnt > d:
                    print(f"{groups}: Exceeded group {gr1} degree {red_cnt}")

                c_max = max(c_max, red_cnt)

        return c_max
