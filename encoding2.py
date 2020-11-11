import base_encoding
from networkx import Graph
from functools import cmp_to_key


class TwinWidthEncoding2(base_encoding.BaseEncoding):
    def __init__(self, stream):
        super().__init__(stream)

        self.edge = None
        self.ord = None
        self.merge = None
        self.node_map = None

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
                self.ord[i][j] = self.add_var()

            for j in range(i + 1, len(g.nodes) + 1):
                self.edge[i][j] = self.add_var()
                if i <= len(g.nodes) - d:
                    self.merge[i][j] = self.add_var()

                    for k in range(j+1, len(g.nodes) + 1):
                        self.red[i][j][k] = self.add_var()

    def encode_order(self, n):
        # Assign one node to each time step
        for i in range(1, n + 1):
            clause = []
            for j in range(1, n + 1):
                clause.append(self.ord[i][j])
                for k in range(j + 1, n+1):
                    self.add_clause(-self.ord[i][j], -self.ord[i][k])
            self.add_clause(*clause)

        # Make sure each node is assigned only once...
        for i in range(1, n+1):
            for j in range(1, n+1):
                for k in range(j+1, n+1):
                    self.add_clause(-self.ord[j][i], -self.ord[k][i])

    def encode_edges(self, g):
        n = len(g.nodes)

        for i in range(1, n+1):
            for j in range(i+1, n+1):
                # Enumerate possible edges
                for k in range(1, n+1):
                    for m in range(k+1, n+1):
                        if g.has_edge(k, m):
                            self.add_clause(-self.ord[i][k], -self.ord[j][m], self.edge[i][j])
                            self.add_clause(-self.ord[i][m], -self.ord[j][k], self.edge[i][j])
                        else:
                            self.add_clause(-self.ord[i][k], -self.ord[j][m], -self.edge[i][j])
                            self.add_clause(-self.ord[i][m], -self.ord[j][k], -self.edge[i][j])

    def encode_edges2(self, g):
        n = len(g.nodes)
        ep = [{} for _ in range(0, n+1)]
        for i in range(1, n+1):
            for j in range(1, n + 1):
                ep[i][j] = self.add_var()

        for i in range(1, n+1):
            for j in range(1, n+1):
                nb = set(g.neighbors(j))
                for k in range(1, n+1):
                    if j == k:
                        continue
                    if k in nb:
                        self.add_clause(-self.ord[i][j], ep[i][k])
                    else:
                        self.add_clause(-self.ord[i][j], -ep[i][k])

        for i in range(1, n+1):
            for j in range(1, n+1):
                for k in range(i+1, n+1):
                    self.add_clause(-self.ord[i][j], -ep[k][j], self.edge[i][k])
                    self.add_clause(-self.ord[i][j], ep[k][j], -self.edge[i][k])

    def break_symmetry(self, n, d):
        ep = [{} for _ in range(0, n + 1)]
        for i in range(1, n + 1 - d):
            for k in range(1, n + 1):
                caux = self.add_var()
                ep[i][k] = caux
                for j in range(i + 1, n + 1):
                    self.add_clause(-self.merge[i][j], -self.ord[j][k], caux)

        # Merges must always occur in lexicographic order
        for i in range(1, n + 1 - d):
            for j in range(1, n+1):
                for k in range(j+1, n + 1):
                    self.add_clause(-ep[i][j], -self.ord[i][k])

    def encode_merge(self, n, d):
        # Exclude root
        for i in range(1, n-d + 1):
            clause = []
            for j in range(i+1, n+1):
                clause.append(self.merge[i][j])
                for k in range(j+1, n+1):
                    self.add_clause(-self.merge[i][j], -self.merge[i][k])
            self.add_clause(*clause)

    def encode_red(self, n, d):
        for i in range(1, n - d + 1):
            for j in range(i+1, n+1):
                for k in range(j+1, n+1):
                    if i > 1:
                        # Transfer red arcs from i to merge target
                        self.add_clause(-self.merge[i][j], -self.red[i-1][i][k], self.red[i][j][k])
                        self.add_clause(-self.merge[i][k], -self.red[i-1][i][j], self.red[i][j][k])
                        # Transfer red arcs from other nodes
                        self.add_clause(-self.red[i-1][j][k], self.red[i][j][k])
                    # Create red arcs
                    self.add_clause(-self.merge[i][j], -self.edge[i][k], self.edge[j][k], self.red[i][j][k])
                    self.add_clause(-self.merge[i][j], -self.edge[j][k], self.edge[i][k], self.red[i][j][k])
                    self.add_clause(-self.merge[i][k], -self.edge[i][j], self.edge[j][k], self.red[i][j][k])
                    self.add_clause(-self.merge[i][k], -self.edge[j][k], self.edge[i][j], self.red[i][j][k])

    def encode_perf1(self, g, d):
        # This is a heuristic, not correct
        n = len(g.nodes)

        for i in range(1, n + 1 - d):
            for j in range(i + 1, n + 1):
                self.add_clause(self.edge[i][j], -self.merge[i][j])

    def encode_perf2(self, g, d):
        # This is a heuristic, not correct
        n = len(g.nodes)

        for i in range(1, n + 1 - d):
            for j in range(i + 1, n + 1):
                if i == 1:
                    self.add_clause(self.edge[i][j], -self.merge[i][j])
                else:
                    self.add_clause(self.edge[i][j], self.red[i-1][i][j], -self.merge[i][j])

    def encode_counters(self, g, d):
        def tred(u, x, y):
            if x < y:
                return self.red[u][x][y]
            else:
                return self.red[u][y][x]

        for i in range(1, len(g.nodes)-d+1):  # As last one is the root, no counter needed
            # Map vars to full adjacency matrix
            vars = [[tred(i, x, y) for x in range(i+1, len(g.nodes) + 1) if x != y] for y in range(i + 1, len(g.nodes) + 1)]
            self.encode_cardinality_sat(d, vars)

    def encode_counters2(self, g, d):
        # This is incorrect
        def tred(u, x, y):
            if x < y:
                return self.red[u][x][y]
            else:
                return self.red[u][y][x]

        vars = []
        for i in range(2, len(g.nodes)+1):
            cvars = []
            for j in range(2, len(g.nodes) + 1):
                if i == j:
                    continue

                clause = []
                for k in range(1, min(min(i, j), len(g.nodes) + 1 - d)):
                    if j + k > len(g.nodes):
                        break
                    if j + k - 1 < i:
                        clause.append(tred(k, i, j + k - 1))
                    else:
                        clause.append(tred(k, i, j + k))

                cvars.append(clause)
            vars.append(cvars)

        self.encode_cardinality_sat(d, vars)

    def encode(self, g, d):
        g = self.remap_graph(g)
        n = len(g.nodes)
        self.init_var(g, d)
        self.encode_edges2(g)
        print(f"{self.clauses}")
        self.encode_order(n)
        print(f"{self.clauses}")
        self.encode_merge(n, d)
        print(f"{self.clauses}")
        self.encode_red(n, d)
        print(f"{self.clauses}")
        self.encode_counters(g, d)
        print(f"{self.clauses}")
        self.break_symmetry(n, d)
        #self.encode_perf2(g, d)
        print(f"{self.clauses} / {self.vars}")
        self.write_header()

    def decode(self, model, g, d):
        g = g.copy()
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
