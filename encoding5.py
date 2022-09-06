import time

from networkx import Graph
from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool
from threading import Timer
import tools


class TwinWidthEncoding2:
    def __init__(self, g, card_enc=EncType.totalizer):
        self.ord = None
        self.merge = None
        self.merged = None
        self.node_map = None
        self.pool = None
        self.formula = None
        self.totalizer = None
        self.g = g
        self.card_enc = card_enc

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
        self.merged = [{} for  _ in range(0, len(g.nodes) + 1)]

        for i in range(1, len(g.nodes)+1):
            for j in range(i+1, len(g.nodes)+1):
                self.merge[i][j] = self.pool.id(f"merge{i}_{j}")

        for t in range(1, len(g.nodes) + 1):
            for i in range(1, len(g.nodes) + 1):
                self.ord[i][t] = self.pool.id(f"ord{i}_{t}")

                if t <= len(g.nodes) - d:
                    self.merged[i][t] = self.pool.id(f"merged{i}_{t}")

                    for j in range(i + 1, len(g.nodes) + 1):
                        self.red[t][i][j] = self.pool.id(f"red{t}_{i}_{j}")

    def encode_order(self, n, d):
        self.formula.append([self.ord[n][n]])
        self.formula.append([self.ord[n-1][n-1]])

        # Assign one node to each time step
        for t in range(1, n-1):
            self.formula.extend(tools.amo_commander([self.ord[t][i] for i in range(1, n+1)], self.pool))
            self.formula.extend(CardEnc.atleast([self.ord[t][i] for i in range(1, n+1)], bound=1, vpool=self.pool))

        # Make sure each node is assigned only once...
        for i in range(1, n + 1):
            self.formula.extend(tools.amo_commander([self.ord[t][i] for t in range(1, n + 1)], vpool=self.pool))

        # Mark nodes as merged
        for i in range(1, n+1):
            for t in range(1, n-d):
                if t > 1:
                    self.formula.append([-self.merged[i][t-1], self.merged[i][t]])
                    self.formula.append([-self.ord[t][i], -self.merged[i][t-1]])
                self.formula.append([-self.ord[t][i], self.merged[i][t]])

            for t in range(n-d, n+1):
                self.formula.append([-self.ord[t][i], -self.merged[i][n - d - 1]])

    def encode_merge(self, n, d):
        for i in range(1, n):
            self.formula.extend(tools.amo_commander([self.merge[i][j] for j in range(i + 1, n + 1)], self.pool))
            self.formula.extend(CardEnc.atleast([self.merge[i][j] for j in range(i + 1, n + 1)], bound=1, vpool=self.pool))

        # Ensure that nodes are never merged into an already merged node
        for t in range(1, n - d):
            for i in range(1, n + 1):
                for j in range(i+1, n + 1):
                    self.formula.append([-self.ord[t][i], -self.merge[i][j], -self.merged[j][t]])

    def tred(self, t, i, j):
        if i < j:
            return self.red[t][i][j]
        else:
            return self.red[t][j][i]

    # TODO: Could explicitly remove all red arcs after merge

    def encode_red(self, n, d, g):
        for i in range(1, n + 1):
            inb = set(g.neighbors(i))
            for t in range(2, n - d + 1):
                for j in range(i+1, n+1):
                    # Create red arcs
                    jnb = set(g.neighbors(j))
                    jnb.discard(i)
                    diff = jnb ^ inb  # Symmetric difference
                    diff.discard(j)

                    for k in diff:
                        self.formula.append(
                            [-self.ord[t - 1][i], -self.merge[i][j], self.merged[k][t-1], self.tred(t-1, j, k)])

                    # Transfer from merge source to merge target
                    for k in range(1, n + 1):
                        if i == k or j == k:
                            continue
                        self.formula.append([-self.ord[t][i], -self.merge[i][j], -self.tred(t-1, i, k), self.tred(t, j, k)])

                # Maintain all other red arcs
                for j in range(1, n+1):
                    if i == j:
                        continue
                    self.formula.append([self.ord[t][i], self.ord[t][j], -self.tred(t - 1, i, j), self.tred(t, i, j)])

    def sb(self, n, d):
        for t in range(1, n-d):
            for i in range(1, n+1):
                for j in range(i+1, n+1):
                    self.formula.append([-self.merged[i][t], -self.red[t][i][j]])
                    self.formula.append([-self.merged[j][t], -self.red[t][i][j]])

    def encode_counters(self, g, d):
        for t in range(1, len(g.nodes)-d):  # As last one is the root, no counter needed
            for i in range(1, len(g.nodes) + 1):
                vars = [self.tred(t, i, j) for j in range(1, len(g.nodes)+1) if i != j]
                self.formula.extend(CardEnc.atmost(vars, bound=d, vpool=self.pool, encoding=self.card_enc))

    def encode(self, g, d):
        g = self.remap_graph(g)
        n = len(g.nodes)
        self.pool = IDPool()
        self.formula = CNF()
        self.init_var(g, d)
        self.encode_order(n, d)
        self.encode_merge(n, d)
        self.encode_red(n, d, g)
        self.encode_counters(g, d)
        self.sb(n, d)

        print(f"{len(self.formula.clauses)} / {self.formula.nv}")
        return self.formula

    def run(self, g, solver, start_bound, verbose=True, check=True, timeout=0):
        start = time.time()
        cb = start_bound

        if verbose:
            print(f"Created encoding in {time.time() - start}")

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
        while i >= 0:
            if done:
                break
            with solver() as slv:
                c_slv = slv
                formula = self.encode(g, i)
                slv.append_formula(formula)

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

        for i in range(1, len(g.nodes) + 1):
            for j in range(1, len(g.nodes) + 1):
                if model[self.ord[i][j]]:
                    if len(od) >= i:
                        print("Double order")
                    od.append(j)
                    unordered.remove(j)
            if len(od) < i:
                print("Order missing")
            if i < len(g.nodes) - d:
                for c_node in od:
                    if not model[self.merged[c_node][i]]:
                        print(f"Not merged, node {c_node} step {i}")
                for c_node in unordered:
                    if model[self.merged[c_node][i]]:
                        print(f"Merged, node {c_node} step {i}")
        if len(set(od)) < len(od):
            print("Node twice in order")

        for i in range(1, len(g.nodes)):
            for j in range(i+1, len(g.nodes) + 1):
                if model[self.merge[i][j]]:
                    if i in mg:
                        print("Error, double merge!")
                    mg[i] = j

        # Perform contractions, last node needs not be contracted...
        for u, v in g.edges:
            g[u][v]['red'] = False

        c_max = 0
        step = 1
        for n in od[:-d-1]:
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
