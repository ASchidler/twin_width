import time

from networkx import Graph
from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool
from threading import Timer
import tools


class TwinWidthEncoding2:
    def __init__(self, g, card_enc=EncType.totalizer, cubic=0, sb_ord=False, twohop=False):
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

    def remap_graph(self, g):
        self.node_map = {}
        cnt = 1
        gn = Graph()

        for u, v in g.edges():
            if u not in self.node_map:
                self.node_map[u] = cnt
                cnt += 1
            if v not in self.node_map:
                self.node_map[v] = cnt
                cnt += 1

            gn.add_edge(self.node_map[u], self.node_map[v])

        return gn

    def init_var(self, g, d):
        self.red = [[{} for _ in range(0, len(g.nodes) + 1)] for _ in range(0, len(g.nodes) + 1)]
        self.ord = [{} for _ in range(0, len(g.nodes) + 1)]
        self.merge = [{} for _ in range(0, len(g.nodes) + 1)]
        self.cardvars = [[] for _ in range(0, len(g.nodes) + 1)]

        if self.cubic < 2:
            for i in range(1, len(g.nodes)+1):
                for j in range(i+1, len(g.nodes)+1):
                    self.merge[i][j] = self.pool.id(f"merge{i}_{j}")
            if self.cubic == 1:
                self.merged_at = [{} for _ in range(0, len(g.nodes) + 1)]
                for t in range(2, len(g.nodes) - d):
                    for j in range(1, len(g.nodes) + 1):
                        self.merged_at[t][j] = self.pool.id(f"ma{t}_{j}")
        else:
            for t in range(1, len(g.nodes) - d):
                for j in range(1, len(g.nodes)+1):
                    self.merge[t][j] = self.pool.id(f"merge{t}_{j}")

        if self.cubic > 0:
            self.merged_edge = [{} for _ in range(0, len(g.nodes) + 1)]
            for t in range(2, len(g.nodes) - d):
                for j in range(1, len(g.nodes)+1):
                    self.merged_edge[t][j] = self.pool.id(f"me{t}_{j}")

        for t in range(1, len(g.nodes) - d):
            for i in range(1, len(g.nodes) + 1):
                self.ord[t][i] = self.pool.id(f"ord{t}_{i}")

                for j in range(i + 1, len(g.nodes) + 1):
                    self.red[t][i][j] = self.pool.id(f"red{t}_{i}_{j}")

    def encode_order(self, n, d):
        # Assign one node to each time step
        for t in range(1, n-d):
            self.formula.append([-self.ord[t][n]])
            self.formula.append([-self.ord[t][n - 1]])
            self.formula.extend(
                CardEnc.atmost([self.ord[t][i] for i in range(1, n+1)], bound=1, vpool=self.pool))
            self.formula.append([self.ord[t][i] for i in range(1, n + 1)])

        # Make sure each node is assigned only once...
        for i in range(1, n + 1):
            self.formula.extend(
                CardEnc.atmost([self.ord[t][i] for t in range(1, n - d)], bound=1, vpool=self.pool))

    def encode_merge(self, n, d):
        if self.cubic < 2:
            for i in range(1, n):
                self.formula.extend(
                    CardEnc.atmost([self.merge[i][j] for j in range(i + 1, n + 1)], bound=1, vpool=self.pool))
                self.formula.append([self.merge[i][j] for j in range(i + 1, n + 1)])

            # Ensure that nodes are never merged into an already merged node
            for t in range(1, n - d):
                for i in range(1, n + 1):
                    for j in range(i+1, n + 1):
                        if t > 1 and self.cubic == 1:
                            self.formula.append([-self.ord[t][i], -self.merge[i][j], self.merged_at[t][j]])
                        for t2 in range(1, t):
                            self.formula.append([-self.ord[t][i], -self.merge[i][j], -self.ord[t2][j]])
        else:
            for t in range(1, n-d):
                self.formula.extend(
                    CardEnc.atmost([self.merge[t][j] for j in range(1, n + 1)], bound=1, vpool=self.pool))
                self.formula.append([self.merge[t][j] for j in range(1, n + 1)])

                for i in range(1, n+1):
                    # Do not merge with yourself
                    self.formula.append([-self.ord[t][i], -self.merge[t][i]])
                    # Do not merge with merged nodes
                    for t2 in range(1, t):
                        self.formula.append([-self.ord[t2][i], -self.merge[t][i]])

        if self.cubic > 0:
            for t in range(2, n-d):
                for i in range(1, n + 1):
                    for k in range(1, n+1):
                        if k == i:
                            continue
                        self.formula.append([-self.ord[t][i], -self.tred(t-1, i, k), self.merged_edge[t][k]])

    def encode_sb_order(self, n, d):
        """Enforce lex ordering whenever there is no node that reaches the bound at time t"""
        if d == 0:
            return

        for t in range(2, n - d):
            c_step_almosts = [self.cardvars[t][x][-2] for x in range(0, n)]
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):
                    clause = list(c_step_almosts)
                    clause.extend([-self.ord[t-1][j], -self.ord[t][i]])
                    self.formula.append(clause)

    def tred(self, t, i, j):
        if i < j:
            return self.red[t][i][j]
        else:
            return self.red[t][j][i]

    def encode_red(self, n, d, g):
        for i in range(1, n + 1):
            inb = set(g.neighbors(i))
            for t in range(1, n - d):
                for j in range(i+1 if self.cubic < 2 else 1, n+1):
                    if i == j:
                        continue

                    # Create red arcs
                    jnb = set(g.neighbors(j))
                    jnb.discard(i)
                    diff = jnb ^ inb  # Symmetric difference
                    diff.discard(j)

                    for k in diff:
                        if self.cubic < 2:
                            start = [-self.ord[t][i], -self.merge[i][j], self.tred(t, j, k)]
                        else:
                            start = [-self.ord[t][i], -self.merge[t][j], self.tred(t, j, k)]
                        for t2 in range(1, t-1):
                            start.append(self.ord[t2][k])
                        self.formula.append(start)

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
                        self.formula.append([self.ord[t][i], self.ord[t][j], -self.tred(t - 1, i, j), self.tred(t, i, j)])
                        if self.cubic == 1:
                            self.formula.append([-self.merged_at[t][i], -self.merged_edge[t][j], self.tred(t, i, j)])
                        elif self.cubic == 2:
                            self.formula.append([-self.merge[t][i], -self.merged_edge[t][j], self.tred(t, i, j)])

    def encode_counters(self, g, d):
        if self.sb_ord:
            self.cardvars.append([])  # Start indexing from 0
        for t in range(1, len(g.nodes)-d):  # As last one is the root, no counter needed
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
                    self.formula.extend(CardEnc.atmost(vars, bound=d, vpool=self.pool, encoding=self.card_enc))

    def encode(self, g, d, od=None, mg=None):
        g = self.remap_graph(g)
        n = len(g.nodes)
        self.pool = IDPool()
        self.formula = CNF()
        self.init_var(g, d)

        if od is not None:
            for i, u in enumerate(od):
                self.formula.append([-self.ord[i+1][u]])

        if mg is not None:
            for k, v in mg.items():
                self.formula.append([-self.merge[k][v]])

        self.encode_order(n, d)
        self.encode_merge(n, d)
        self.encode_red(n, d, g)
        self.encode_counters(g, d)

        if self.sb_ord:
            self.encode_sb_order(n, d)
        if self.twohop:
            self.sb_twohop(n, d, g, True)

        return self.formula

    def sb_twohop(self, n, d, g, full=True):
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

                for t in range(2, n-d):
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
    def run(self, g, solver, start_bound, verbose=True, check=True, lb = 0, timeout=0, od=None, mg=None):
        start = time.time()
        cb = start_bound

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
                formula = self.encode(g, i, od, mg)

                slv.append_formula(formula)

                if verbose:
                    print(f"Created encoding in {time.time() - start} {slv.nof_clauses()}/{slv.nof_vars()}")

                if done:
                    break

                if slv.solve() if timeout == 0 else slv.solve_limited():
                    if verbose:
                        print(f"Found {i}")
                    cb = self.decode(slv.get_model(), g, i)
                    i = cb - 1
                else:
                    if verbose:
                        print(f"Failed {i}")
                    break

                if verbose:
                    print(f"Finished cycle in {time.time() - start}")
        if timer is not None:
            timer.cancel()
        print(f"Finished in {time.time() - start}")
        return cb

    def decode(self, model, g, d):
        g = g.copy()
        model = {abs(x): x > 0 for x in model}
        unmap = {}
        for u, v in self.node_map.items():
            unmap[v] = u

        # Find merge targets and elimination order
        mg = {}
        od = []
        unordered = set(range(1, len(g.nodes)+1))

        for t in range(1, len(g.nodes) - d):
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
            for i in range(1, len(g.nodes)-d):
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
        print(f"Done {c_max}/{d}")
        return c_max
