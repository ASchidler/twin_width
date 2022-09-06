import time

from networkx import Graph
from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool
from threading import Timer
import tools


class TwinWidthEncoding2:
    def __init__(self, g, card_enc=EncType.totalizer):
        self.edge = None
        self.ord = None
        self.merge = None
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
        self.edge = [{} for _ in range(0, len(g.nodes) + 1)]
        self.red = [[{} for _ in range(0, len(g.nodes) + 1)] for _ in range(0, len(g.nodes) + 1)]
        self.ord = [{} for _ in range(0, len(g.nodes) + 1)]
        self.merge = [{} for _ in range(0, len(g.nodes) + 1)]

        for i in range(1, len(g.nodes) + 1):
            for j in range(1, len(g.nodes) + 1):
                self.ord[i][j] = self.pool.id(f"ord{i}_{j}")

            for j in range(i + 1, len(g.nodes) + 1):
                self.edge[i][j] = self.pool.id(f"edge{i}_{j}")
                if i <= len(g.nodes) - d:
                    self.merge[i][j] = self.pool.id(f"merge{i}_{j}")

                    for k in range(j+1, len(g.nodes) + 1):
                        self.red[i][j][k] = self.pool.id(f"red{i}_{j}_{k}")

    def encode_order(self, n):
        self.formula.append([self.ord[n][n]])
        self.formula.append([self.ord[n-1][n-1]])
        # Assign one node to each time step
        for i in range(1, n):
            self.formula.extend(tools.amo_commander([self.ord[i][j] for j in range(1, n+1)], self.pool))
            self.formula.extend(CardEnc.atleast([self.ord[i][j] for j in range(1, n+1)], bound=1, vpool=self.pool))

        # Make sure each node is assigned only once...
        for i in range(1, n+1):
            self.formula.extend(tools.amo_commander([self.ord[j][i] for j in range(1, n + 1)], vpool=self.pool))

    def encode_edges(self, g):
        n = len(g.nodes)

        for i in range(1, n+1):
            for j in range(i+1, n+1):
                # Enumerate possible edges
                for k in range(1, n+1):
                    for m in range(k+1, n+1):
                        if g.has_edge(k, m):
                            self.formula.append([-self.ord[i][k], -self.ord[j][m], self.edge[i][j]])
                            self.formula.append([-self.ord[i][m], -self.ord[j][k], self.edge[i][j]])
                        else:
                            self.formula.append([-self.ord[i][k], -self.ord[j][m], -self.edge[i][j]])
                            self.formula.append([-self.ord[i][m], -self.ord[j][k], -self.edge[i][j]])

    def encode_edges2(self, g):
        n = len(g.nodes)
        ep = [{} for _ in range(0, n+1)]
        for i in range(1, n+1):
            for j in range(1, n + 1):
                ep[i][j] = self.pool.id(f"edges_ep{i}_{j}")

        for i in range(1, n+1):
            for j in range(1, n+1):
                nb = set(g.neighbors(j))
                for k in range(1, n+1):
                    if j == k:
                        continue
                    if k in nb:
                        self.formula.append([-self.ord[i][j], ep[i][k]])
                    else:
                        self.formula.append([-self.ord[i][j], -ep[i][k]])

                nb_vars = [self.ord[i][x] for x in nb]
                self.formula.append([-ep[i][j], *nb_vars])

        for i in range(1, n+1):
            for j in range(1, n+1):
                for k in range(i+1, n+1):
                    self.formula.append([-self.ord[i][j], -ep[k][j], self.edge[i][k]])
                    self.formula.append([-self.ord[i][j], ep[k][j], -self.edge[i][k]])

    def break_symmetry(self, n, d):
        ep = [{} for _ in range(0, n + 1)]
        for i in range(1, n + 1 - d):
            for k in range(1, n + 1):
                caux = self.pool.id(f"symmetry_ep{i}_{k}")
                ep[i][k] = caux
                for j in range(i + 1, n + 1):
                    self.formula.append([-self.merge[i][j], -self.ord[j][k], caux])

        # Merges must always occur in lexicographic order
        for i in range(1, n + 1 - d):
            for j in range(1, n+1):
                for k in range(j+1, n + 1):
                    self.formula.append([-ep[i][j], -self.ord[i][k]])

    def skip_doublehops(self, n, d):
        for i in range(1, n-d + 1):
            for j in range(i+1, n+1):
                vars = []
                for k in range(i+1, n+1):
                    if j == k:
                        continue

                    caux = self.pool.id(f"doublehop{i}_{j}_{k}")
                    if i > 1:
                        self.formula.append([-caux, self.edge[i][k], self.red[i-1][i][k]])
                        if j < k:
                            self.formula.append([-caux, self.edge[j][k], self.red[i-1][j][k]])
                        else:
                            self.formula.append([-caux, self.edge[k][j], self.red[i - 1][k][j]])
                    else:
                        self.formula.append([-caux, self.edge[i][k]])
                        if j < k:
                            self.formula.append([-caux, self.edge[j][k]])
                        else:
                            self.formula.append([-caux, self.edge[k][j]])
                    vars.append(caux)

                if i > 1:
                    self.formula.append([-self.merge[i][j], self.edge[i][j], self.red[i-1][i][j], *vars])
                else:
                    self.formula.append([-self.merge[i][j], self.edge[i][j], *vars])

    def encode_merge(self, n, d):
        # Exclude root
        for i in range(1, n-d + 1):
            self.formula.extend(tools.amo_commander([self.merge[i][j] for j in range(i+1, n + 1)], vpool=self.pool))
            self.formula.extend(CardEnc.atleast([self.merge[i][j] for j in range(i + 1, n + 1)], vpool=self.pool, bound=1))

    def encode_red(self, n, d):
        for i in range(1, n - d + 1):
            for j in range(i+1, n+1):
                for k in range(j+1, n+1):
                    if i > 1:
                        # Transfer red arcs from i to merge target
                        self.formula.append([-self.merge[i][j], -self.red[i-1][i][k], self.red[i][j][k]])
                        self.formula.append([-self.merge[i][k], -self.red[i-1][i][j], self.red[i][j][k]])
                        # Transfer red arcs from other nodes
                        self.formula.append([-self.red[i-1][j][k], self.red[i][j][k]])
                    # Create red arcs
                    self.formula.append([-self.merge[i][j], -self.edge[i][k], self.edge[j][k], self.red[i][j][k]])
                    self.formula.append([-self.merge[i][j], -self.edge[j][k], self.edge[i][k], self.red[i][j][k]])
                    self.formula.append([-self.merge[i][k], -self.edge[i][j], self.edge[j][k], self.red[i][j][k]])
                    self.formula.append([-self.merge[i][k], -self.edge[j][k], self.edge[i][j], self.red[i][j][k]])

    def encode_counters(self, g, d):
        def tred(u, x, y):
            if x < y:
                return self.red[u][x][y]
            else:
                return self.red[u][y][x]

        for i in range(1, len(g.nodes)-d+1):  # As last one is the root, no counter needed
            for x in range(i+1, len(g.nodes) + 1):
                vars = [tred(i, x, y) for y in range(i+1, len(g.nodes)+1) if x != y]
                self.formula.extend(CardEnc.atmost(vars, bound=d, vpool=self.pool, encoding=self.card_enc))

    def encode(self, g, d):
        g = self.remap_graph(g)
        n = len(g.nodes)
        self.pool = IDPool()
        self.formula = CNF()
        self.init_var(g, d)
        self.encode_edges2(g)
        self.encode_order(n)
        self.encode_merge(n, d)
        self.encode_red(n, d)
        self.encode_counters(g, d)
        #self.skip_doublehops(n, d)
        #print(f"{len(self.formula.clauses)}")

        #self.break_symmetry(n, d)
        #self.encode_perf2(g, d)
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

        for i in range(1, len(g.nodes) + 1):
            for j in range(1, len(g.nodes) + 1):
                if model[self.ord[i][j]]:
                    if len(od) >= i:
                        print("Double order")
                    od.append(j)
            if len(od) < i:
                print("Order missing")
        if len(set(od)) < len(od):
            print("Node twice in order")

        for i in range(1, len(g.nodes) + 1 - d):
            for j in range(i+1, len(g.nodes) + 1):
                if model[self.merge[i][j]]:
                    if od[i-1] in mg:
                        print("Error, double merge!")
                    mg[od[i-1]] = od[j-1]

        # Check edges relation...
        for i in range(0, len(g.nodes)-d):
            for j in range(i+1, len(g.nodes)-d):
                if model[self.edge[i+1][j+1]] ^ g.has_edge(unmap[od[i]], unmap[od[j]]):
                    if model[self.edge[i+1][j+1]]:
                        print(f"Edge error: Unknown edge in model {i+1}, {j+1} = {od[i], od[j]}")
                    else:
                        print("Edge error: Edge not in model")

        # Perform contractions, last node needs not be contracted...
        for u, v in g.edges:
            g[u][v]['red'] = False

        c_max = 0
        step = 1
        for n in od[:-d]:
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
                        u2, v2 = od.index(self.node_map[u]) + 1, od.index(self.node_map[v]) + 1
                        u2, v2 = min(u2, v2), max(u2, v2)
                        if not model[self.red[step][u2][v2]]:
                            print(f"Missing red edge in step {step}")
                if cc > d:
                    print(f"Exceeded bound in step {step}")
                c_max = max(c_max, cc)

            step += 1
        print(f"Done {c_max}/{d}")

        return c_max
