import time

from networkx import Graph
from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool
from threading import Timer
from multiprocessing import Process, Manager
import tools
import encoding5, encoding
import tools

class TwinWidthEncoding2:
    def __init__(self, g, card_enc=EncType.totalizer):
        self.ord = None
        self.merge = None
        self.node_map = None
        self.pool = None
        self.formula = None
        self.totalizer = None
        self.g = g
        self.g_mapped = None
        self.cstep = 0
        self.card_enc = card_enc
        self.last_step = []
        self.merged_with = None
        self.merged_edge = None

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

    def add_step(self, n, g, d, slv, use_ord_sb=True, use_red_rev=True):
        self.cstep += 1

        # Encode ordering
        for i in range(1, n + 1):
            self.ord[self.cstep][i] = self.pool.id(f"ord_{self.cstep}_{i}")

        if self.cstep < n:
            slv.add_clause([-self.ord[self.cstep][n]])
        if self.cstep < n-1:
            slv.add_clause([-self.ord[self.cstep][n-1]])

        slv.append_formula(
            CardEnc.atmost([self.ord[self.cstep][i] for i in range(1, n+1)], bound=1, vpool=self.pool))
        slv.append_formula(CardEnc.atleast([self.ord[self.cstep][i] for i in range(1, n+1)], bound=1, vpool=self.pool))

        for t in range(1, self.cstep):
            for i in range(1, n + 1):
                slv.add_clause([-self.ord[t][i], -self.ord[self.cstep][i]])

        # Encode merge
        for i in range(1, n + 1):
            self.merge[self.cstep][i] = self.pool.id(f"merge_{self.cstep}_{i}")
        slv.append_formula(
            CardEnc.atmost([self.merge[self.cstep][j] for j in range(1, n + 1)], bound=1, vpool=self.pool))
        slv.append_formula(CardEnc.atleast([self.merge[self.cstep][j] for j in range(1, n + 1)], bound=1, vpool=self.pool))

        for t in range(1, self.cstep):
            for i in range(1, n + 1):
                slv.add_clause([-self.ord[t][i], -self.merge[self.cstep][i]])

        for i in range(1, n + 1):
            slv.add_clause([-self.ord[self.cstep][i], -self.merge[self.cstep][i]])

        # Arcs
        for i in range(1, n + 1):
            for j in range(i+1, n+1):
                self.red[self.cstep][i][j] = self.pool.id(f"red{self.cstep}_{i}_{j}")

        if self.cstep > 1:
            for i in range(1, n + 1):
                self.merged_edge[self.cstep][i] = self.pool.id(f"me{self.cstep}_{i}")

            for i in range(1, n + 1):
                for k in range(1, n+1):
                    if k == i:
                        continue
                    slv.add_clause([-self.ord[self.cstep][i], -self.tred(self.cstep-1, i, k), self.merged_edge[self.cstep][k]])

        for i in range(1, n + 1):
            inb = set(g.neighbors(i))
            for j in range(1, n+1):
                if i == j:
                    continue
                # Create red arcs
                jnb = set(g.neighbors(j))
                jnb.discard(i)
                diff = jnb ^ inb  # Symmetric difference
                diff.discard(j)

                for k in diff:
                    start = [-self.ord[self.cstep][i], -self.merge[self.cstep][j]]
                    for t in range(1, self.cstep):
                        start.append(self.ord[t][k])
                    start.append(self.tred(self.cstep, j, k))
                    slv.add_clause(start)

            # Maintain all other red arcs
            if self.cstep > 1:
                for j in range(i+1, n+1):
                    if i == j:
                        continue
                    slv.add_clause([self.ord[self.cstep][i], self.ord[self.cstep][j], -self.tred(self.cstep - 1, i, j), self.tred(self.cstep, i, j)])
                    slv.add_clause([-self.merge[self.cstep][i], -self.merged_edge[self.cstep][j], self.tred(self.cstep, i, j)])
                    if use_red_rev and i < j:
                        slv.add_clause(
                            [-self.tred(self.cstep, i, j), self.tred(self.cstep-1, i, j), self.merge[self.cstep][j], self.merge[self.cstep][i]])
            elif use_red_rev:
                for j in range(i+1, n+1):
                    if i == j:
                        continue
                    slv.add_clause(
                        [-self.tred(self.cstep, i, j), self.merge[self.cstep][j], self.merge[self.cstep][i]])

        c_step_almosts = []
        for i in range(1, len(g.nodes) + 1):
            vars = [self.tred(self.cstep, i, j) for j in range(1, len(g.nodes) + 1) if i != j]
            formula, cardvars = tools.encode_cards_exact(self.pool, vars, d, f"cardvars_{self.cstep}_{i}")
            slv.append_formula(formula)
            if use_ord_sb:
                if d > 0:
                    c_step_almosts.append(cardvars[-2])
            # else:
            #     slv.append_formula(CardEnc.atmost(vars, bound=d, vpool=self.pool, encoding=self.card_enc))

        if use_ord_sb:
            if self.cstep > 1 and d > 0:
                for i in range(1, len(g.nodes) + 1):
                    for j in range(i + 1, len(g.nodes) + 1):
                        clause = list(c_step_almosts)
                        clause.extend([-self.ord[self.cstep-1][j], -self.ord[self.cstep][i]])
                        slv.add_clause(clause)

    def init_var(self, g):
        self.cstep = 0
        self.red = [[{} for _ in range(0, len(g.nodes) + 1)] for _ in range(0, len(g.nodes) + 1)]
        self.ord = [{} for _ in range(0, len(g.nodes) + 1)]
        self.merge = [{} for _ in range(0, len(g.nodes) + 1)]
        self.merged_with = [{} for _ in range(0, len(g.nodes) + 1)]
        self.merged_edge = [{} for _ in range(0, len(g.nodes) + 1)]

        # for i in range(1, len(g.nodes)+1):
        #     for j in range(1, len(g.nodes)+1):
        #         self.merge[i][j] = self.pool.id(f"merge{i}_{j}")

    def tred(self, t, i, j):
        if i < j:
            return self.red[t][i][j]
        else:
            return self.red[t][j][i]

    def encode(self, g):
        g = self.remap_graph(g)
        self.g_mapped = g
        n = len(g.nodes)
        self.pool = IDPool()
        self.formula = CNF()
        self.init_var(g)

        return self.formula

    def run(self, g, solver, start_bound, verbose=True, check=True, timeout=0):
        use_ord_sb = True
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

        lb = 0
        success = False
        while not success:
            success = True
            result = False

            with solver() as slv:
                c_slv = slv
                formula = self.encode(g)
                slv.append_formula(formula)
                self.last_step.clear()
                while self.cstep < len(g.nodes) - lb:
                    if len(self.last_step) > 0:
                        for cv in self.last_step:
                            slv.add_clause([-cv])
                        self.last_step.clear()

                    self.add_step(len(g.nodes), self.g_mapped, lb, slv, use_ord_sb)
                    # self.sb_twohop(len(g.nodes), self.g_mapped, slv, True)
                    if verbose:
                        print(f"Bound: {lb}, Step: {self.cstep} {slv.nof_clauses()}/{slv.nof_vars()}")

                    result = (slv.solve() if timeout == 0 and len(self.last_step) == 0 else slv.solve_limited(self.last_step))

                    if verbose:
                        print(f"Finished cycle in {time.time() - start}")

                    if not result:
                        if verbose:
                            print(f"Unsat, increasing bound to {lb + 1}")
                        success = False
                        lb += 1
                        break

                if result and success:
                    cb = self.decode(slv.get_model(), g, lb)
                    break

        if timer is not None:
            timer.cancel()

        return cb

    def sb_twohop(self, n, g, slv, full=True):
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

                if self.cstep == 1:
                    self.formula.append([-self.ord[1][i], -self.merge[1][j]])
                if not full or self.cstep == 1:
                    continue

                for k in range(1, n + 1):
                    if k == i or k == j:
                        continue

                    overall_clause = [-self.merge[self.cstep][j], -self.ord[self.cstep][i], self.tred(self.cstep-1, i, j)]

                    if g.has_edge(i, k):
                        overall_clause.append(self.tred(self.cstep - 1, j, k))
                    elif g.has_edge(j, k):
                        overall_clause.append(self.tred(self.cstep - 1, i, k))
                    else:
                        aux = self.pool.id(f"tc_{self.cstep}_{i}_{k}_{j}")
                        slv.add_clause([-aux, self.tred(self.cstep-1, i, k)])
                        slv.add_clause([-aux, self.tred(self.cstep - 1, j, k)])
                        overall_clause.append(aux)

                    slv.add_clause(overall_clause)

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

        for i in range(1, self.cstep + 1):
            for j in range(1, len(g.nodes) + 1):
                if model[self.ord[i][j]]:
                    if len(od) >= i:
                        print("Double order")
                    od.append(j)
                    unordered.remove(j)
            if len(od) < i:
                print("Order missing")

        if len(set(od)) < len(od):
            print("Node twice in order")

        for i in range(1, self.cstep + 1):
            for j in range(1, len(g.nodes) + 1):
                if model[self.merge[i][j]]:
                    if i in mg:
                        print(f"Error, double merge! {i} ({mg[i]}/{j})")
                    mg[i] = j

        # Perform contractions, last node needs not be contracted...
        for u, v in g.edges:
            g[u][v]['red'] = False

        c_max = 0
        step = 1
        for i, n in enumerate(od):
            t = unmap[mg[i+1]]
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
        return c_max, od, mg
