import sys
import time

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
        self.formula = None
        self.totalizer = None
        self.cardvars = None
        self.g = g
        self.card_enc = card_enc
        self.sb_ord = sb_ord
        self.twohop = twohop
        self.sb_static = sb_static
        self.sb_static_full = sb_static_full
        self.static_card = None
        self.sb_static_diff = sb_static_diff
        self.is_grid = is_grid

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
        self.cardvars = [[] for _ in range(0, len(g.nodes) + 1)]
        self.static_card = [{} for _ in range(0, len(g.nodes) + 1)]

        for i in range(1, len(g.nodes)+1):
            for j in range(i+1, len(g.nodes)+1):
                self.merge[i][j] = self.pool.id(f"merge{i}_{j}")

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
            self.formula.extend(
                CardEnc.atmost([self.ord[t][i] for t in range(1, steps)], bound=1, vpool=self.pool))

    def encode_merge(self, n, steps):
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
                        start = [-self.ord[t][i], -self.merge[i][j], self.tred(t, j, k)]

                        for t2 in range(1, t-1):
                            start.append(self.ord[t2][k])
                        self.formula.append(start)

                    # Transfer from merge source to merge target
                    if t > 1:
                        for k in range(1, n + 1):
                            if i == k or j == k:
                                continue
                            self.formula.append([-self.ord[t][i], -self.merge[i][j], -self.tred(t-1, i, k), self.tred(t, j, k)])

                # Maintain all other red arcs
                if t > 1:
                    for j in range(1, n+1):
                        if i == j:
                            continue
                        if i < j:
                            self.formula.append([self.ord[t][i], self.ord[t][j], -self.tred(t - 1, i, j), self.tred(t, i, j)])

    def encode_counters(self, g, d, steps):
        if self.sb_ord:
            self.cardvars.append([])  # Start indexing from 0
        for t in range(1, steps):  # As last one is the root, no counter needed
            # if self.cubic > 0 and t > 1:
            #     vars = [self.merged_edge[t][j] for j in range(1, len(g.nodes) + 1)]
            #     self.formula.extend(CardEnc.atmost(vars, bound=d, vpool=self.pool, encoding=self.card_enc))

            for i in range(1, len(g.nodes) + 1):
                vars = [self.tred(t, i, j) for j in range(1, len(g.nodes)+1) if i != j]
                with ITotalizer(vars, ubound=d, top_id=self.pool.id(f"totalizer{t}_{i}")) as tot:
                    self.formula.extend(tot.cnf)
                    self.pool.occupy(self.pool.top - 1, tot.top_id)
                    self.formula.append([-tot.rhs[d]])
                    self.cardvars[t].append(list(tot.rhs))

                # self.formula.extend(CardEnc.atmost(vars, bound=d, vpool=self.pool, encoding=self.card_enc))

    def encode(self, g, d, od=None, mg=None, steps=None):
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
            for k, v in mg.items():
                self.formula.append([self.merge[k][v]])

        self.encode_order(n, steps)
        self.encode_merge(n, steps)
        self.encode_red(n, g, steps)
        self.encode_counters(g, d, steps)

        return self.formula

    def run(self, g, solver, start_bound, verbose=True, check=True, lb = 0, timeout=0, i_od=None, i_mg=None, steps_limit=None):
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
                    steps = len(g.nodes) - i
                else:
                    steps = min(len(g.nodes) - i, steps_limit)

                formula = self.encode(g, i, i_od, i_mg, steps)

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
