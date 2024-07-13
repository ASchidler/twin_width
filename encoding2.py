import sys
import time

from networkx import Graph
from pysat.card import CardEnc, EncType, ITotalizer
from pysat.formula import CNF, IDPool, WCNF
from threading import Timer
import tools
import os


class TwinWidthEncoding2:
    def __init__(self, g, card_enc=EncType.mtotalizer, cubic=0, sb_ord=False, twohop=False, sb_static=sys.maxsize, sb_static_full=False, sb_static_diff=False, break_g_symmetry=True):
        self.ord = None
        self.merge = None
        self.merged_edge = None
        self.merged_at = None
        self.node_map = None
        self.pool = None
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
        self.ord_vars = None
        self.real_merge = None
        self.break_g_symmetry = break_g_symmetry

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

    def init_var(self, g, steps):
        self.red = [[{} for _ in range(0, len(g.nodes) + 1)] for _ in range(0, len(g.nodes) + 1)]
        self.ord = [{} for _ in range(0, len(g.nodes) + 1)]
        self.merge = [{} for _ in range(0, len(g.nodes) + 1)]
        self.ord_vars = [{} for _ in range(0, len(g.nodes) + 1)]
        self.cardvars = [[] for _ in range(0, len(g.nodes) + 1)]
        self.static_card = [{} for _ in range(0, len(g.nodes) + 1)]
        self.real_merge = [{} for _ in range(0, len(g.nodes) + 1)]

        if self.cubic < 2:
            for i in range(1, len(g.nodes)+1):
                for j in range(i+1, len(g.nodes)+1):
                    self.merge[i][j] = self.pool.id(f"merge{i}_{j}")
        else:
            for t in range(1, steps):
                for j in range(1, len(g.nodes)+1):
                    self.merge[t][j] = self.pool.id(f"merge{t}_{j}")

            for i in range(1, len(g.nodes)+1):
                for j in range(i+1, len(g.nodes)+1):
                    self.real_merge[i][j] = self.pool.id(f"rmerge{i}_{j}")

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
        if self.cubic < 2:
            for i in range(1, n):
                self.formula.extend(
                    CardEnc.atmost([self.merge[i][j] for j in range(i + 1, n + 1)], bound=1, vpool=self.pool))
                self.formula.append([self.merge[i][j] for j in range(i + 1, n + 1)])

            # Ensure that nodes are never merged into an already merged node
            for t in range(1, steps):
                for i in range(1, n + 1):
                    for j in range(i+1, n + 1):
                        for t2 in range(1, t):
                            self.formula.append([-self.ord[t][i], -self.merge[i][j], -self.ord[t2][j]])
        else:
            for t in range(1, steps):
                self.formula.extend(
                    CardEnc.atmost([self.merge[t][j] for j in range(1, n + 1)], bound=1, vpool=self.pool))
                self.formula.append([self.merge[t][j] for j in range(1, n + 1)])

                for i in range(1, n+1):
                    # Do not merge with yourself or if already eliminated
                    self.formula.append([-self.ord_vars[i][t], -self.merge[t][i]])
                    # Lex ordering
                    self.formula.append([-self.merge[t][i], *[self.ord[t][j] for j in range(1, i)]])

                    for j in range(i + 1, n + 1):
                        self.formula.append([-self.ord[t][i], -self.merge[t][j], self.real_merge[i][j]])

        if self.cubic > 0:
            for t in range(2, steps):
                for i in range(1, n + 1):
                    for k in range(1, n+1):
                        if k == i:
                            continue
                        self.formula.append([-self.ord[t][i], -self.tred(t-1, i, k), self.merged_edge[t][k]])

            for i in range(1, n):
                self.formula.extend(CardEnc.atmost([self.real_merge[i][j] for j in range(i + 1, n + 1)], vpool=self.pool))

    def sb_ord2(self, n, d, g, steps):
        """Enforce lex ordering whenever there is no node that reaches the bound at time t"""
        if d == 0:
            return

        ord_dec = [{} for _ in range(0, n + 1)]

        for t in range(3, steps):
            for i in range(1, n + 1):
                ord_dec[t][i] = self.pool.id(f"ord_dec_{t}_{i}")
            #
                for cd in range(1, d):
                    if cd < d-1:
                        # pass
                        self.formula.append([-self.merge[t][i], -self.cardvars[t-2][i-1][cd], self.cardvars[t-1][i-1][cd], -self.cardvars[t][i-1][d], self.ord_vars[i][t], ord_dec[t][i]])
                    else:
                        self.formula.append([-self.cardvars[t - 2][i-1][d-1], self.cardvars[t - 1][i-1][d-1], -self.cardvars[t][i-1][d-1], self.ord_vars[i][t], ord_dec[t][i]])

                    self.formula.append([-ord_dec[t][i], -self.merge[t][i], -self.cardvars[t-2][i-1][cd], self.ord_vars[i][t], -self.cardvars[t-1][i-1][cd]])

                self.formula.append([-ord_dec[t][i], self.cardvars[t][i-1][d-1]])
                self.formula.append([-ord_dec[t][i], -self.ord_vars[i][t]])
                self.formula.append([-ord_dec[t][i], self.merge[t][i], self.cardvars[t - 2][i-1][d-1]])
                self.formula.append([-ord_dec[t][i], self.merge[t][i], -self.cardvars[t - 1][i-1][d-1]])

        for t in range(2, steps-1):
            for j in range(1, n+1):
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
                    if self.sb_static_full:
                        lits = [self.pool.id(f"static_st_{n1}_{cn}") for cn in sd]
                        with ITotalizer(lits, ubound=d, top_id=self.pool.id(f"static_full_{n1}_{n2}")) as tot:
                            self.pool.occupy(self.pool.top - 1, tot.top_id)
                            self.formula.extend(tot.cnf)
                            cards = list(tot.rhs)
                        # form, cards = tools.encode_cards_exact(self.pool, lits, d, f"static_full_{n1}_{n2}",
                        #                                        add_constraint=False)
                        # self.formula.extend(form)
                        self.static_card[n1][n2] = cards

                        if self.cubic < 2 and d > 0:
                            self.formula.append([-self.merge[n1][n2], -self.static_card[n1][n2][d]])

                        for t in range(1, min(steps, self.sb_static)):
                            for cn in sd:
                                cl = [self.pool.id(f"static_st_{n1}_{cn}"), -self.ord[t][n1]]
                                for t2 in range(1, t):
                                    cl.append(self.ord[t2][cn])

                                self.formula.append(cl)

                    for t in range(1, len(sd)-d):
                        self.formula.append([-self.ord[t][n1], -self.merge[t if self.cubic == 2 else n1][n2]])

                    for t in range(len(sd)-d, min(steps, self.sb_static)):
                        if len(sd) - d > 1 and self.sb_static_full:
                            if self.cubic == 2:
                                self.formula.append(
                                    [-self.ord[t][n1], -self.merge[t][n2], -self.static_card[n1][n2][d]])

                            if self.sb_static_diff:
                                for cd in range(0, d):
                                    if cd > 0:
                                        # There might be a red edge between i and j and some other node that might be contracted away, so + 1
                                        self.formula.append([-self.ord[t][n1],
                                                        -self.merge[t if self.cubic == 2 else n1][n2],
                                                        -self.cardvars[t - 1][n2-1][cd],
                                                        -self.static_card[n1][n2][d - cd]])
                                    self.formula.append([-self.ord[t][n1],
                                                    -self.merge[t if self.cubic == 2 else n1][n2],
                                                    self.tred(t - 1, n1, n2),
                                                    -self.cardvars[t - 1][n2-1][cd],
                                                    -self.static_card[n1][n2][d - cd - 1]])  # Substract 1 as cardvars[d] expresses that d is exceeded!
                        else:
                            cl = [-self.ord[t][n1], -self.merge[t if self.cubic == 2 else n1][n2]]

                            for t2 in range(1, t):
                                for i in sd:
                                    cl.append(self.ord[t2][i])
                            self.formula.append(cl)
    def tred(self, t, i, j):
        if i < j:
            return self.red[t][i][j]
        else:
            return self.red[t][j][i]

    def encode_red(self, n, g, steps):
        for i in range(1, n + 1):
            inb = set(g.neighbors(i))
            for t in range(1, steps):
                for j in range(i+1, n+1):
                    if i == j:
                        continue

                    # Create red arcs
                    jnb = set(g.neighbors(j))
                    jnb.discard(i)
                    diff = jnb ^ inb  # Symmetric difference
                    diff.discard(j)

                    for k in diff:
                        if self.cubic == 0:
                            start = [-self.ord[t][i], -self.merge[i][j], self.tred(t, j, k)]
                            for t2 in range(1, t):
                                start.append(self.ord[t2][k])
                        else:
                            # start = [-self.ord[t][i], -self.real_merge[i][j], self.tred(t, j, k), *[self.ord[t2][k] for t2 in range(1, t)]]
                            if t > 1:
                                start = [-self.ord_vars[i][t], self.ord_vars[j][t], self.ord_vars[k][t],
                                         -self.real_merge[i][j], self.tred(t, j, k)]
                            else:
                                start = [-self.ord[t][i], -self.real_merge[i][j], self.tred(t, j, k)]

                            # start = [-self.ord[t][i], -self.merge[t][j], self.tred(t, j, k)]
                        self.formula.append(start)

                    # if self.sb_ord and i < j and self.cubic == 2:
                    #     for k in set(range(1, n+1)) - diff - {i, j}:
                    #         if t > 1:
                    #             self.formula.append([-self.merge[t][j], -self.ord[t][i], self.tred(t-1, j, k), self.tred(t-1, i, k), -self.tred(t, j, k)])
                    #         else:
                    #             self.formula.append([-self.merge[t][j], -self.ord[t][i], -self.tred(t, j, k)])

                    # Transfer from merge source to merge target
                    if self.cubic == 0 and t > 1:
                        for k in range(1, n + 1):
                            if i == k or j == k:
                                continue
                            self.formula.append([-self.ord[t][i], -self.merge[i][j], -self.tred(t-1, i, k), self.tred(t, j, k)])

                    if self.sb_ord and i < j and self.cubic == 2:
                        if t > 1:
                            self.formula.append(
                                [-self.tred(t, i, j), self.tred(t-1, i, j), self.merge[t][j], self.merge[t][i]])
                        else:
                            self.formula.append(
                                [-self.tred(t, i, j), self.merge[t][j], self.merge[t][i]])

                # Maintain all other red arcs
                if t > 1:
                    for j in range(1, n+1):
                        if i == j:
                            continue
                        if i < j:
                            self.formula.append([self.ord[t][i], self.ord[t][j], -self.tred(t - 1, i, j), self.tred(t, i, j)])
                        
                        if self.cubic == 2:
                            self.formula.append([-self.merge[t][i], -self.merged_edge[t][j], self.tred(t, i, j)])

    def encode_counters(self, g, d, steps):
        if self.sb_ord:
            self.cardvars.append([])  # Start indexing from 0
        for t in range(1, steps):  # As last one is the root, no counter needed
            # if self.cubic > 0 and t > 1:
            #     vars = [self.merged_edge[t][j] for j in range(1, len(g.nodes) + 1)]
            #     self.formula.extend(CardEnc.atmost(vars, bound=d, vpool=self.pool, encoding=self.card_enc))

            for i in range(1, len(g.nodes) + 1):
                vars = [self.tred(t, i, j) for j in range(1, len(g.nodes)+1) if i != j]
                if self.sb_ord:
                    form, cvars = tools.encode_cards_exact(self.pool, vars, d, f"cardvars_{t}_{i}")
                    self.formula.extend(form)
                    self.cardvars[t].append(cvars)
                else:
                    with ITotalizer(vars, ubound=d, top_id=self.pool.id(f"totalizer{t}_{i}")) as tot:
                        self.formula.extend(tot.cnf)
                        self.pool.occupy(self.pool.top - 1, tot.top_id)
                        self.formula.append([-tot.rhs[d]])
                        self.cardvars[t].append(list(tot.rhs))

                    # self.formula.extend(CardEnc.atmost(vars, bound=d, vpool=self.pool, encoding=self.card_enc))

    def encode(self, g, d, od=None, mg=None, steps=None, skip_cards=False):
        if steps is None:
            steps = len(g.nodes) - d

        g = self.remap_graph(g)
        n = len(g.nodes)
        self.pool = IDPool()
        self.formula = CNF()
        self.init_var(g, steps)

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
        self.encode_red(n, g, steps)

        if not skip_cards:
            self.encode_counters(g, d, steps)

        if self.sb_ord:
            self.sb_ord2(n, d, g, steps)

        if self.sb_static > 0:
            self.encode_sb_static(n, d, g, steps)

        if self.break_g_symmetry:
            import pynauty

            gn = pynauty.Graph(len(g.nodes))
            for cn in g.nodes:
                gn.connect_vertex(cn - 1, [cb - 1 for cb in g.neighbors(cn)])
            grp = pynauty.autgrp(gn)
            orbits = grp[3]
            seen = set()
            clause = []
            nodes = []
            orbit_groups = [[] for _ in range(0, len(g.nodes))]

            # The first eliminated vertex must be the smallest from each group
            for cni, cn in enumerate(range(1, len(g.nodes)+1)):
                orbit_groups[orbits[cni]].append(cn)
                if orbits[cni] not in seen:
                    clause.append(self.ord[1][cn])
                    nodes.append(cn)
                    seen.add(orbits[cni])
            self.formula.append(clause)

        return self.formula

    def wcnf_export(self, g, start_bound, filename, export_cards):
        formula = self.encode(g, start_bound, None, None, steps=len(g.nodes), skip_cards=export_cards)
        wcnf = WCNF() if not export_cards else CNF()
        wcnf.extend(formula)

        if not export_cards:
            softs = [self.pool.id(f"softs_{i}") for i in range(1, start_bound+1)]
            for cs in softs:
                wcnf.append([-cs], 1)

            for ct in self.cardvars:
                for cn in ct:
                    for cd in range(0, start_bound):
                        wcnf.append([-cn[cd], softs[cd]])

        wcnf.to_file(filename)
        if export_cards:
            with open(filename+".cards", "w") as outp:
                for t in range(1, len(g.nodes)-1):  # As last one is the root, no counter needed
                    # if self.cubic > 0 and t > 1:
                    #     vars = [self.merged_edge[t][j] for j in range(1, len(g.nodes) + 1)]
                    #     self.formula.extend(CardEnc.atmost(vars, bound=d, vpool=self.pool, encoding=self.card_enc))

                    for i in range(1, len(g.nodes) + 1):
                        vars = [self.tred(t, i, j) for j in range(1, len(g.nodes) + 1) if i != j]
                        outp.write(" ".join(str(x) for x in vars))
                        outp.write(f" <= d" + os.linesep)

    def run(self, g, solver, start_bound, verbose=True, check=True, lb=0, timeout=0, i_od=None, i_mg=None, steps_limit=None, write=False):
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
                    formula.to_file("test2.cnf")
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
        for n in od:
            t = unmap[mg[n]]
            n = unmap[n]
            if verbose:
                print(f"{n} => {t}")
            tn = set(g.neighbors(t))
            tn.discard(n)
            nn = set(g.neighbors(n))

            for v in nn:
                if v != t:
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

                if cc > d:
                    print(f"Exceeded bound in step {step}")
                c_max = max(c_max, cc)

            step += 1
        return c_max, od, mg
