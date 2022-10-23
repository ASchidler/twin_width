import os
import sys
import time
from collections import defaultdict

from networkx import Graph
from pysat.card import CardEnc, EncType, ITotalizer
from pysat.formula import CNF, IDPool
from threading import Timer
import tools


class TwinWidthEncoding2:
    def __init__(self, g, card_enc=EncType.mtotalizer, cubic=0, sb_ord=False, twohop=False, sb_static=sys.maxsize, sb_static_full=False, sb_static_diff=False, is_grid=False):
        self.ord = None
        self.merge = None
        self.merged_edge = None
        self.merged_at = None
        self.node_map = None
        self.pool = None
        self.ord_vars = None
        self.formula = None
        self.totalizer = None
        self.cardvars = None
        self.g = g
        self.card_enc = card_enc
        self.cubic = cubic
        self.sb_ord = sb_ord
        self.twohop = twohop
        self.sb_static = sb_static
        self.sb_static_full = sb_static_full
        self.static_card = None
        self.sb_static_diff = sb_static_diff
        self.is_grid = is_grid
        self.real_merge = None

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

    def init_var(self, g, steps, d):
        self.red = [[{} for _ in range(0, len(g.nodes) + 1)] for _ in range(0, len(g.nodes) + 1)]
        self.ord = [{} for _ in range(0, len(g.nodes) + 1)]
        self.merge = [{} for _ in range(0, len(g.nodes) + 1)]
        self.real_merge = [{} for _ in range(0, len(g.nodes) + 1)]
        self.cardvars = [[] for _ in range(0, len(g.nodes) + 1)]
        self.static_card = [[{} for _ in range(0, len(g.nodes) + 1)] for _ in range(0, len(g.nodes) + 1)]
        self.counters = [[{} for _ in range(0, len(g.nodes) + 1)] for _ in range(0, len(g.nodes) + 1)]
        self.ord_vars = [{} for _ in range(0, len(g.nodes) + 1)]

        for t in range(1, steps):
            for j in range(1, len(g.nodes)+1):
                self.merge[t][j] = self.pool.id(f"merge{t}_{j}")
        for i in range(1, len(g.nodes) + 1):
            for j in range(i + 1, len(g.nodes) + 1):
                self.real_merge[i][j] = self.pool.id(f"real_merge{i}_{j}")

        if self.cubic > 0:
            self.merged_edge = [{} for _ in range(0, len(g.nodes) + 1)]
            for t in range(2, steps):
                for j in range(1, len(g.nodes)+1):
                    self.merged_edge[t][j] = self.pool.id(f"me{t}_{j}")

        for t in range(1, steps):
            for i in range(1, len(g.nodes) + 1):
                self.ord[t][i] = self.pool.id(f"ord{t}_{i}")

                for j in range(i + 1, len(g.nodes) + 1):
                    self.red[t][i][j] = self.pool.id(f"red{t}_{i}_{j}")

        for t in range(1, steps):
            for i in range(1, len(g.nodes) + 1):
                for x in range(1, d+1):
                    self.counters[t][i][x] = self.pool.id(f"counters_{t}_{i}_{x}")

    def encode_order(self, n, steps):
        # Assign one node to each time step
        for t in range(1, steps):
            if t < n:
                self.formula.append([-self.ord[t][n]])
            self.formula.extend(
                CardEnc.atmost([self.ord[t][i] for i in range(1, n+1)], bound=1, vpool=self.pool))
            self.formula.append([self.ord[t][i] for i in range(1, n + 1)])

        # Make sure each node is assigned only once...
        for i in range(1, n + 1):
            self.ord_vars[i] = [None]
            clauses, ovs = tools.amo_seq([self.ord[t][i] for t in range(1, steps)], f"ord_amo_{i}", self.pool)
            self.ord_vars[i].extend(ovs)
            self.formula.extend(clauses)

            # Make implication an equivalence, causes speedup
            for t in range(1, steps):
                self.formula.append([*[self.ord[t2][i] for t2 in range(1, t+1)], -self.ord_vars[i][t]])

    def encode_merge(self, n, steps):
        for t in range(1, steps):
            self.formula.extend(
                CardEnc.atmost([self.merge[t][j] for j in range(1, n + 1)], bound=1, vpool=self.pool))
            self.formula.append([self.merge[t][j] for j in range(1, n + 1)])

            for i in range(1, n+1):
                # Do not merge with yourself
                self.formula.append([-self.ord_vars[i][t], -self.merge[t][i]])
                # Lex ordering
                self.formula.append([-self.merge[t][i], *[self.ord[t][j] for j in range(1, i)]])

                for j in range(i+1, n+1):
                    self.formula.append([-self.ord[t][i], -self.merge[t][j], self.real_merge[i][j]])

            if t > 1:
                for i in range(1, n + 1):
                    for k in range(1, n+1):
                        if k == i:
                            continue
                        self.formula.append([-self.ord[t][i], -self.tred(t-1, i, k), self.merged_edge[t][k]])
                        self.formula.append([-self.ord[t][i], self.tred(t-1, i, k), -self.merged_edge[t][k]])

        for i in range(1, n):
            self.formula.append([self.ord_vars[i][steps-1], self.real_merge[i][n]])
            self.formula.extend(CardEnc.atmost([self.real_merge[i][j] for j in range(i + 1, n + 1)], vpool=self.pool))

    def encode_sb_order(self, n, d, steps, max_diff=sys.maxsize):
        """Enforce lex ordering whenever there is no node that reaches the bound at time t"""
        if d == 0:
            return
        
        ord_dec = [{} for _ in range(0, n + 1)]

        for t in range(2, steps):
            for i in range(1, n+1):
                ord_dec[t][i] = self.pool.id(f"ord_dec_{t}_{i}")

                clause = [-ord_dec[t][i]]
                for cd in range(1, d+1):
                    aux_dec_s = self.pool.id(f"ord_dec_{t}_{i}_{cd}")
                    self.formula.append([-self.counters[t-1][i][cd], self.counters[t][i][cd], aux_dec_s])
                    self.formula.append([-aux_dec_s, self.counters[t-1][i][cd]])
                    self.formula.append([-aux_dec_s, -self.counters[t][i][cd]])
                    self.formula.append([-aux_dec_s, ord_dec[t][i]])
                    clause.append(aux_dec_s)
                self.formula.append(clause)

        for t in range(1, steps-2):
            decs = [self.pool.id(f"ord_dec_{t}_{i}") for i in range(1, n+1)]
            for i in range(t+1, n+1):
                aux = self.pool.id(f"exceeded_{t}_{i}")
                self.formula.append([-self.ord[t][i], aux, *decs])
                if t > 1:
                    self.formula.append([-self.pool.id(f"exceeded_{t-1}_{i}"), self.pool.id(f"exceeded_{t}_{i}"), *decs])

            for i in range(1, n+1):
                for j in range(n+1, 1):
                    self.formula.append([-self.ord[t][i], -self.pool.id(f"exceeded_{t}_{j}")])

        # Full
        # ord_dec_t = [[{} for _ in range(0, n+1)] for _ in range(0, n + 1)]
        # exceeded = [{} for _ in range(0, n + 1)]
        #
        # for t in range(1, steps-1):
        #     for i in range(t+1, n+1):
        #         exceeded[t][i] = self.pool.id(f"exceeded_{t}_{i}")
        #         self.formula.append([-self.ord[t][i], exceeded[t][i]])
        #
        # for t in range(2, steps):
        #     for i in range(2, n+1):
        #         exceeded[t][i] = self.pool.id(f"exceeded_{t}_{i}")
        #         # self.formula.append([-exceeded[t][i], exceeded[t-1][i], self.ord[t][i]])
        #
        # for t in range(2, steps):
        #     for i in range(2, n+1):
        #         for j in range(1, n+1):
        #             if i != j:
        #                 ord_dec_t[t][i][j] = self.pool.id(f"ord_dec_t_{t}_{i}_{j}")
        #
        #                 if t == 2:
        #                     self.formula.append([-ord_dec_t[t][i][j], ord_dec[t][j]])
        #                 else:
        #                     self.formula.append([-ord_dec_t[t][i][j], ord_dec[t][j], ord_dec_t[t-1][i][j]])
        #                     self.formula.append([-ord_dec_t[t - 1][i][j], -exceeded[t][i], ord_dec_t[t][i][j]])
        #
        #                 self.formula.append([-ord_dec_t[t][i][j], exceeded[t][i]])
        #                 self.formula.append([-ord_dec[t][j], -exceeded[t][i], ord_dec_t[t][i][j]])
        #
        # for t in range(2, steps):
        #     for i in range(1, n+1):
        #         if t > 1:
        #             for j in range(2, n+1):
        #                 if i != j:
        #                     ok_aux = self.pool.id(f"is_ok_{t}_{j}_{i}")
        #                     self.formula.append([self.counters[t-1][i][d], -self.counters[t][i][d],  -ord_dec_t[t][j][i], ok_aux])
        #                     self.formula.append([-self.counters[t-1][i][d], -ok_aux])
        #                     self.formula.append([ord_dec_t[t][j][i], -ok_aux])
        #                     self.formula.append([self.counters[t][i][d], -ok_aux])
        #
        #             if i > 1 and t < steps - 1:
        #                 self.formula.append([-exceeded[t][i], *[self.pool.id(f"is_ok_{t}_{i}_{j}") for j in range(1, n+1) if i != j], exceeded[t+1][i]])
        #
        #             for j in range(i+1, n+1):
        #                 self.formula.append([-self.ord[t][i], -exceeded[t][j]])

    def sb_ord2(self, n, d, g, steps):
        """Enforce lex ordering whenever there is no node that reaches the bound at time t"""
        if d == 0:
            return

        ord_dec = [{} for _ in range(0, n + 1)]

        for t in range(3, steps):
            for i in range(1, n + 1):
                ord_dec[t][i] = self.pool.id(f"ord_dec_{t}_{i}")

                for cd in range(1, d+1):
                    aux = self.pool.id(f"ord_dec_{t}_{i}_{cd}")
                    if cd < d:
                        self.formula.append([-self.merge[t][i], -self.counters[t-2][i][cd], self.counters[t-1][i][cd], -self.counters[t][i][d], ord_dec[t][i]])
                    else:
                        self.formula.append([-self.counters[t - 2][i][cd], self.counters[t - 1][i][cd], -self.counters[t][i][d], ord_dec[t][i]])

                    self.formula.append([-self.counters[t - 2][i][cd], self.counters[t-1][i][cd], aux])
                    self.formula.append([-aux, self.counters[t-2][i][cd]])
                    self.formula.append([-aux, -self.counters[t-1][i][cd]])
                    self.formula.append([-ord_dec[t][i], -self.merge[t][i], -self.counters[t-2][i][cd], -self.counters[t-1][i][cd]])

                self.formula.append([-ord_dec[t][i], self.counters[t][i][d]])
                self.formula.append([-ord_dec[t][i], self.counters[t-2][i][1]])
                self.formula.append([-ord_dec[t][i], self.merge[t][i], self.counters[t - 2][i][d]])
                self.formula.append([-ord_dec[t][i], self.merge[t][i], -self.counters[t - 1][i][d]])

        for t in range(2, steps-1):
            for j in range(t+1, n+1):
                for i in range(1, j):
                    self.formula.append([-self.ord[t+1][i], -self.ord[t][j], *[ord_dec[t+1][k] for k in range(1, n+1)]])

    def encode_sb_static(self, n, d, g, steps):
        for n1 in range(1, n+1):
            n1nb = set(g.neighbors(n1))
            for n2 in range(n1+1, n + 1):
                n2nb = set(g.neighbors(n2))
                sd = n1nb ^ n2nb
                sd.discard(n1)
                sd.discard(n2)

                if len(sd) > d:
                    if self.sb_static_full and len(sd) > d + 1:
                        lits = [self.pool.id(f"static_st_{n1}_{cn}") for cn in sd]
                        with ITotalizer(lits, ubound=d, top_id=self.pool.id(f"static_full_{n1}_{n2}")) as tot:
                            self.pool.occupy(self.pool.top - 1, tot.top_id)
                            self.formula.extend(tot.cnf)
                            cards = list(tot.rhs)
                        # form, cards = tools.encode_cards_exact(self.pool, lits, d, f"static_full_{n1}_{n2}",
                        #                                        add_constraint=False)
                        # self.formula.extend(form)
                        self.static_card[n1][n2] = cards

                        for t in range(1, min(steps, self.sb_static)):
                            for cn in sd:
                                self.formula.append([-self.ord_vars[n1][t], self.ord_vars[cn][t], self.pool.id(f"static_st_{n1}_{cn}")])
                                # cl = [self.pool.id(f"static_st_{n1}_{cn}"), -self.ord[t][n1]]
                                # for t2 in range(1, t):
                                #     cl.append(self.ord[t2][cn])
                                #
                                # self.formula.append(cl)

                    for t in range(1, len(sd)-d):
                        self.formula.append([-self.ord[t][n1], -self.real_merge[n1][n2]])

                    if self.sb_static_full and len(sd) > d + 1:
                        self.formula.append(
                            [-self.real_merge[n1][n2], -self.static_card[n1][n2][d]])

                    for t in range(len(sd)-d, min(steps, self.sb_static)):
                        if len(sd) - d > 1 and self.sb_static_full:
                            # self.formula.append(
                            #     [-self.ord[t][n1], -self.merge[t][n2], -self.static_card[n1][n2][d]])
                            if self.sb_static_diff:
                                for cd in range(1, d+1):
                                    if cd > 0:
                                        # There might be a red edge between n1 and n2 and some other node that might be contracted away, so + 1
                                        self.formula.append([-self.ord[t][n1],
                                                        -self.real_merge[n1][n2],
                                                        -self.counters[t-1][n2][cd],
                                                        -self.static_card[n1][n2][d - cd + 1]])
                                    self.formula.append([-self.ord[t][n1],
                                                    -self.real_merge[n1][n2],
                                                    self.tred(t - 1, n1, n2), # Edge did not exist -> tighten bound
                                                    -self.counters[t-1][n2][cd],
                                                    -self.static_card[n1][n2][d - cd]])
                        else:
                            cl = [-self.ord[t][n1], -self.real_merge[n1][n2]]
                            for i in sd:
                                cl.append(self.ord_vars[i][t])
                            self.formula.append(cl)
    def tred(self, t, i, j):
        if i < j:
            return self.red[t][i][j]
        else:
            return self.red[t][j][i]

    def encode_red(self, n, g, steps, d):
        differences = {}
        edge_sources = defaultdict(list)

        for i in range(1, n + 1):
            inb = set(g.neighbors(i))
            for j in range(i+1, n+1):
                # Create red arcs
                jnb = set(g.neighbors(j))
                jnb.discard(i)
                diff = jnb ^ inb  # Symmetric difference
                diff.discard(j)

                differences[(i, j)] = diff
                for k in diff:
                    edge_sources[(min(i, k), max(i, k))].append((i, j))
                    edge_sources[(min(j, k), max(j, k))].append((i, j))

        # Create and maintain edges
        for i in range(1, n + 1):
            for t in range(1, steps):
                for j in range(i+1, n+1):
                    if i == j:
                        continue

                    # Create new edges
                    diff = differences[(i, j)]
                    for k in diff:
                        # start = [-self.ord[t][i], -self.real_merge[i][j], self.tred(t, j, k), *[self.ord[t2][k] for t2 in range(1, t)]]
                        if t > 1:
                            start = [-self.ord_vars[i][t], self.ord_vars[j][t], self.ord_vars[k][t],
                                     -self.real_merge[i][j], self.tred(t, j, k)]
                        else:
                            start = [-self.ord[t][i], -self.real_merge[i][j], self.tred(t, j, k)]

                        self.formula.append(start)


                    # Make sure every red edge is created for a reason
                    clause1 = [-self.tred(t, i, j), -self.merge[t][i]]
                    clause2 = [-self.tred(t, i, j), -self.merge[t][j]]
                    for cmu, cmv in edge_sources[(min(i, j), max(i, j))]:
                        if cmv == i:
                            clause1.append(self.ord[t][cmu]) # cmu < cmv, hence only cmu can be the merge source
                        elif cmv == j:
                            clause2.append(self.ord[t][cmu])

                    if t > 1:
                        self.formula.append(
                            [-self.tred(t, i, j), self.tred(t-1, i, j), self.merge[t][j], self.merge[t][i]])
                        clause1.extend([self.merged_edge[t][j], self.tred(t-1, i, j)])
                        clause2.extend([self.merged_edge[t][i], self.tred(t-1, i, j)])
                        self.formula.append(clause1)
                        self.formula.append(clause2)
                    else:
                        self.formula.append(
                            [-self.tred(t, i, j), self.merge[t][j], self.merge[t][i]])
                        self.formula.append(clause1)
                        self.formula.append(clause2)

                # Maintain all other red arcs
                if t > 1:
                    for j in range(1, n+1):
                        if i == j:
                            continue
                        if i < j:
                            # self.formula.append([self.ord_vars[i][t], self.ord_vars[j][t], -self.tred(t - 1, i, j), self.tred(t, i, j)])
                            self.formula.append(
                                [self.ord[t][i], self.ord[t][j], -self.tred(t - 1, i, j), self.tred(t, i, j)])

                        self.formula.append([-self.merge[t][i], -self.merged_edge[t][j], self.tred(t, i, j)])

    def encode_vertex_counters(self, n, d, steps):
        # Counting
        # First, move the merge targets red edges to red_count
        for t in range(1, steps):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if i != j:
                        self.formula.append([-self.merge[t][j], -self.tred(t, i, j), self.pool.id(f"red_count_{t}_{i}")])

        if d == 0:
            return

        # Next, use the cardinality constraint for counting
        for t in range(1, steps - 1):
            for i in range(1, n + 1):
                for x in range(1, d + 1):
                    self.formula.append([-self.merge[t][i], -self.cardvars[t][x - 1], self.counters[t][i][x]])

        # Next, handle all other counters
        for t in range(1, steps):
            for i in range(1, n + 1):
                aux_now = self.pool.id(f"red_count_{t}_{i}")

                if t == 1:
                    self.formula.append([self.merge[t][i], -aux_now, self.counters[t][i][1]])

                    if self.sb_ord:
                        self.formula.append([self.merge[t][i], aux_now, -self.counters[t][i][1]])
                        for x in range(2, d + 1):
                            self.formula.append([self.merge[t][i], -self.counters[t][i][d]])
                else:
                    aux_source = self.merged_edge[t][i]
                    aux_target = self.pool.id(f"mergeexists_{t}_{i}")
                    prefix = [self.merge[t][i], aux_source, aux_target, -aux_now, ]

                    # Increase counters
                    self.formula.append([*prefix, self.counters[t][i][1]])
                    for x in range(1, d):
                        self.formula.append([*prefix, -self.counters[t - 1][i][x], self.counters[t][i][x + 1]])
                    self.formula.append([*prefix, -self.counters[t - 1][i][d]])  # Exceeds

                    # Maintain
                    for x in range(1, d + 1):
                        self.formula.append([self.merge[t][i], -self.counters[t - 1][i][x], aux_source,
                                             self.counters[t][i][x]])
                        self.formula.append(
                            [self.merge[t][i], -self.counters[t - 1][i][x], aux_target, self.counters[t][i][x]])

                    # Ensure counter decreases at most by 1
                    for x in range(2, d + 1):
                        self.formula.append([-self.counters[t-1][i][x], self.counters[t][i][x-1]])

                    if self.sb_ord:
                        for x in range(1, d + 1):
                            self.formula.append([self.merge[t][i], -self.counters[t - 1][i][x], self.counters[t][i][x],
                                                 aux_source])
                            self.formula.append([self.merge[t][i], -self.counters[t - 1][i][x], self.counters[t][i][x],
                                                 aux_target])
                            if x > 1:
                                self.formula.append(
                                    [self.merge[t][i], -self.counters[t][i][x], self.counters[t - 1][i][x - 1]])
                            if x < d:
                                self.formula.append(
                                    [self.merge[t][i], -self.counters[t][i][x], self.counters[t - 1][i][x],
                                     -aux_source])
                                self.formula.append(
                                    [self.merge[t][i], -self.counters[t][i][x], self.counters[t - 1][i][x],
                                     -aux_target])

                for j in range(1, n + 1):
                    if i != j:
                        if t > 1:
                            self.formula.append([-self.merge[t][j], -self.tred(t - 1, i, j), aux_target])
                            self.formula.append([-self.merge[t][j], self.tred(t - 1, i, j), -aux_target])

    def encode_counters(self, g, d, steps):
        for t in range(1, steps):  # As last one is the root, no counter needed
            vars = [self.pool.id(f"red_count_{t}_{i}") for i in range(1, len(g.nodes)+1)]
            with ITotalizer(vars, ubound=d, top_id=self.pool.id(f"totalizer_{t}")) as tot:
                self.formula.extend(tot.cnf)
                self.pool.occupy(self.pool.top - 1, tot.top_id)
                self.formula.append([-tot.rhs[d]])
                self.cardvars[t] = list(tot.rhs)

    def encode(self, g, d, od=None, mg=None, steps=None):
        if steps is None:
            steps = len(g.nodes) - d - 1

        g = self.remap_graph(g)
        n = len(g.nodes)
        self.pool = IDPool()
        self.formula = CNF()
        self.init_var(g, steps, d)

        if od is not None:
            for i, u in enumerate(od):
                if i + 1 < steps:
                    self.formula.append([self.ord[i+1][u]])

        if mg is not None:
            if self.cubic == 2:
                for i, u in enumerate(od):
                    if i + 1 < steps:
                        self.formula.append([self.merge[i+1][mg[u]]])
            else:
                for k, v in mg.items():
                    self.formula.append([self.merge[k][v]])

        self.encode_order(n, steps)
        self.encode_merge(n, steps)
        self.encode_counters(g, d, steps)
        self.encode_red(n, g, steps, d)
        self.encode_vertex_counters(n, d, steps)

        if self.sb_ord:
            # self.sb_ord2(n, d, g, steps)
            self.encode_sb_order(n, d, steps)
        if self.twohop:
            self.sb_twohop(n, g, steps, True)

        if self.sb_static > 0:
            self.encode_sb_static(n, d, g, steps)

        if self.is_grid:
            clause = []
            dimension1 = max(x for x, y in self.node_map.keys()) + 1
            dimension2 = max(y for x, y in self.node_map.keys()) + 1

            for x1 in range(0, dimension1 // 2 + dimension1 % 2):
                for x2 in range(0, dimension2 // 2 + dimension2 % 2):
                    x1_coord = [x1, dimension1 - 1 - x1]
                    x2_coord = [x2, dimension2 - 1 - x2]

                    coords = [(x1_coord[0], x2_coord[0]), (x1_coord[1], x2_coord[0]),
                              (x1_coord[0], x2_coord[1]), (x1_coord[1], x2_coord[0])]
                    clause.append(self.ord[1][min(self.node_map[x] for x in coords)])

            self.formula.append(clause)

        return self.formula

    def sb_twohop(self, n, g, steps, full=True):
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if g.has_edge(i, j):
                    continue
                istc = False
                for k in range(1, n + 1):
                    if g.has_edge(i, k) and g.has_edge(j, k):
                        istc = True
                        break
                if istc:
                    continue

                if self.cubic < 2:
                    self.formula.append([-self.ord[1][i], -self.merge[i][j]])
                else:
                    self.formula.append([-self.ord[1][i], -self.merge[1][j]])

                if not full:
                    continue

                for t in range(2, steps):
                    for k in range(1, n + 1):
                        if k == i or k == j:
                            continue

                        if self.cubic < 2:
                            overall_clause = [-self.merge[i][j], -self.ord[t][i], self.tred(t-1, i, j)]
                        else:
                            overall_clause = [-self.merge[t][j], -self.ord[t][i], self.tred(t - 1, i, j)]

                        if g.has_edge(i, k):
                            overall_clause.append(self.tred(t-1, j, k))
                        elif g.has_edge(j, k):
                            overall_clause.append(self.tred(t-1, i, k))
                        else:
                            aux = self.pool.id(f"tc_{t}_{i}_{k}_{j}")
                            self.formula.append([-aux, self.tred(t-1, i, k)])
                            self.formula.append([-aux, self.tred(t - 1, j, k)])
                            self.formula.append([aux, -self.tred(t - 1, i, k), -self.tred(t - 1, j, k)])
                            overall_clause.append(aux)

                        self.formula.append(overall_clause)

    def run(self, g, solver, start_bound, verbose=True, check=True, lb = 0, timeout=0, i_od=None, i_mg=None, steps_limit=None, write=False):
        if len(g.nodes) < 4:
            return 0, None, None

        start = time.time()
        cb = start_bound
        od = None
        mg = None

        done = []
        c_slv = None
        def interrupt():
            if c_slv is not None:
                c_slv.interrupt()
            done.append(True)

        timer = None
        if timeout > 0:
            timer = Timer(timeout, interrupt)
            timer.start()

        i = start_bound
        while i >= lb:
            if done:
                break
            with solver() as slv:
                c_slv = slv
                if steps_limit is None:
                    steps = len(g.nodes) - i - 1
                else:
                    steps = min(len(g.nodes) - i - 1, steps_limit)

                if steps <= 1:
                    cb = i
                    i = cb - 1
                    continue

                formula = self.encode(g, i, i_od, i_mg, steps)
                if write:
                    formula.to_file("test3.cnf")
                # if os.path.exists("symmetries.txt"):
                #     with open("symmetries.txt") as syminp:
                #         for cl in syminp:
                #             cl = cl.strip()
                #             cl = [int(x) for x in cl.split(" ") if x != "0"]
                #
                #             cl2 = [("-" if x < 0 else "") + (self.pool.obj(abs(x)) if abs(x) <= self.pool.top else f"aux{x}") for x in cl]
                #             print(cl2)
                #             slv.add_clause(cl)

                slv.append_formula(formula)

                if verbose:
                    print(f"Created encoding in {time.time() - start} {len(formula.clauses)}/{formula.nv}")
                    print(f"Solver clauses {slv.nof_clauses()}/{slv.nof_vars()}")

                if done:
                    break

                if slv.solve() if timeout == 0 else slv.solve_limited():
                    if verbose:
                        print(f"Found {i}")
                    cb, od, mg = self.decode(slv.get_model(), g, i, steps, verbose)
                    i = cb - 1
                else:
                    if verbose:
                        print(f"Failed {i}")
                    break

                if verbose:
                    print(f"Finished cycle in {time.time() - start}")
        if timer is not None:
            timer.cancel()
        if verbose:
            print(f"Finished in {time.time() - start}")
        if od is None:
            return cb
        else:
            return cb, od, mg

    def decode(self, model, g, d, steps, verbose):
        g = g.copy()
        model = {abs(x): x > 0 for x in model}
        unmap = {}
        for u, v in self.node_map.items():
            unmap[v] = u

        # Find merge targets and elimination order
        mg = {}
        od = []
        unordered = set(range(1, len(g.nodes)+1))

        for t in range(1, steps):
            for j in range(1, len(g.nodes) + 1):
                if model[self.ord[t][j]]:
                    if len(od) >= t:
                        print("Double order")
                    od.append(j)
                    unordered.remove(j)
            if len(od) < t:
                print("Order missing")

        if len(set(od)) < len(od):
            print("Node twice in order")

        if self.cubic < 2:
            for i in range(1, len(g.nodes)):
                for j in range(i+1, len(g.nodes) + 1):
                    if model[self.merge[i][j]]:
                        if i in mg:
                            print("Error, double merge!")
                        mg[i] = j
        else:
            for i in range(1, steps):
                t = od[i-1]
                for j in range(1, len(g.nodes) + 1):
                    if model[self.merge[i][j]]:
                        if t in mg:
                            print("Error, double merge!")
                        mg[t] = j

        # Perform contractions, last node needs not be contracted...
        for u, v in g.edges:
            g[u][v]['red'] = False

        c_max = 0
        step = 1
        watches = set()
        decreases = [[]]
        red_counts = [[]]

        for i, n in enumerate(od):
            if verbose:
                print(f"{n} => {mg[n]}")

            red_counts.append({})
            decreases.append(set())

            t = unmap[mg[n]]
            n = unmap[n]

            tn = set(g.neighbors(t))
            tn.discard(n)
            nn = set(g.neighbors(n))

            for v in nn:
                if v != t:
                    if g[n][v]['red'] and v in tn and g[t][v]['red']:
                        decreases[-1].add(v)

                    # Red remains, should edge exist
                    if v in tn and g[n][v]['red']:
                        g[t][v]['red'] = True
                    # Add non-existing edges
                    if v not in tn:
                        g.add_edge(t, v, red=True)
            for v in tn:
                if v not in nn:
                    g[t][v]['red'] = True
            g.remove_node(n)

            # Count reds...
            for u in g.nodes:
                cc = 0
                for v in g.neighbors(u):
                    if g[u][v]['red']:
                        cc += 1
                        u2, v2 = self.node_map[u], self.node_map[v]
                        u2, v2 = min(u2, v2), max(u2, v2)

                        if not model[self.red[step][u2][v2]]:
                            print(f"Missing red edge in step {step}")
                red_counts[-1][u] = cc
                if step > 1 and red_counts[-2][u] < red_counts[-1][u]:
                    decreases[-1].add(u)

                if cc > d:
                    climit = 0
                    for x in range(1, d+1):
                        if model[self.counters[step][self.node_map[u]][x]]:
                            climit = x
                    print(f"Exceeded bound in step {step}, node {u}, {cc}/{climit}")
                c_max = max(c_max, cc)

            for cn in g.nodes:
                if red_counts[-1][cn] == d and (step == 1 or red_counts[-2][cn] != d):
                    for ci, cd in enumerate(decreases):
                        if cn in cd:
                            cdel = [cw for cw in watches if cw[0] <= ci]
                            for cde in cdel:
                                watches.discard(cde)

            # if any(od[i] < x[1] for x in watches):
            #     print("SB Violation")
            #
            # for i in range(1, len(g.nodes) + 1):
            #     for j in range(i+1, len(g.nodes) + 1):
            #         if model[self.tred(step, i, j)]:
            #             u2, v2 = unmap[i], unmap[j]
            #             if not g.has_edge(u2, v2) or not g[u2][v2]["red"]:
            #                 print(f"Excess red edge {u2} {v2}")

            step += 1

            if od[i] > i+1:
                watches.add((i, od[i]))

        return c_max, od, mg
