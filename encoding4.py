import base_encoding
from networkx import Graph
from functools import cmp_to_key
from collections import defaultdict


class TwinWidthEncoding(base_encoding.BaseEncoding):
    def __init__(self, stream):
        super().__init__(stream)

        self.edge = None
        self.ord = None
        self.merge = None
        self.node_map = None
        self.d = 0

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
        self.ord = [{} for _ in range(0, len(g.nodes) + 1)]
        self.merge = [{} for _ in range(0, len(g.nodes) + 1)]
        self.d = d

        for i in range(1, len(g.nodes) + 1):
            for j in range(i + 1, len(g.nodes) + 1):
                self.ord[i][j] = self.add_var()
                self.merge[i][j] = self.add_var()

        for i in range(1, len(g.nodes)+1):
            self.edge[i] = {}
            for j in range(1, len(g.nodes)+1):
                if i == j:
                    continue
                self.edge[i][j] = {}
                for k in range(1, d+1):

                    self.edge[i][j][k] = {}
                    for m in range(1, len(g.nodes)+1):
                        if i == m:
                            continue

                        if j != m:
                            self.edge[i][j][k][m] = self.add_var()

    def tord(self, i, j):
        if i < j:
            return self.ord[i][j]

        return -self.ord[j][i]

    def tedge(self, n, i, j):
        if i < j:
            return self.edge[n][i][j]

        return self.edge[n][j][i]

    # def tmerge(self, i, j):
    #     if i < j:
    #         return self.merge[i][j]
    # 
    #     return self.merge[j][i]

    def encode_edge(self, n, d):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i == j:
                    continue
                # Only one bit set per entry
                for k in range(1, d+1):
                    self.amo_commander([self.edge[i][j][k][x] for x in range(1, n+1) if j != x and i != x], alo=False)
                # Edges may only occur once (symmetry breaking)
                for k in range(1, n+1):
                    if j != k and i != k:
                        self.amo_pair([self.edge[i][j][x][k] for x in range(1, d + 1)], alo=False)
                # # Sort ascending (symmetry breaking)
                # for k in range(d, 0, -1):
                #     for k2 in range(k-1, 0, -1):
                #         for x in range(1, n+1):
                #             if j == x or i == x:
                #                 continue
                #             for x2 in range(x+1, n+1):
                #                 if j == x2 or i == x2:
                #                     continue
                #
                #                 self.add_clause(-self.edge[i][j][k][x], -self.edge[i][j][k2][x2])

    def edge_clause(self, i, j, k):
        clause = []
        for m in range(1, self.d+1):
            clause.append(self.edge[i][j][m][k])
        return clause

    def encode(self, g, d):
        g = self.remap_graph(g)
        n = len(g.nodes)
        self.init_var(g, d)

        # Encode relationships
        # Transitivity
        for i in range(1, len(g.nodes) + 1):
            for j in range(1, len(g.nodes) + 1):
                if i == j:
                    continue
                for k in range(1, len(g.nodes) + 1):
                    if i == k or j == k:
                        continue

                    self.add_clause(-self.tord(i, j), -self.tord(j, k), self.tord(i, k))
        
        # Merge/ord relationship
        for i in range(1, len(g.nodes) + 1):            
            for j in range(i+1, len(g.nodes) + 1):
                self.add_clause(-self.merge[i][j], self.tord(i, j))
                
        # single merge target
        for i in range(1, len(g.nodes) + 1):
            # self.amo_pair([self.merge[i][j] for j in range(i+1, len(g.nodes)+1)], elo=True)
            self.amo_commander([self.merge[i][j] for j in range(i + 1, len(g.nodes) + 1)], alo=True)

        # Create red arcs
        for i in range(1, len(g.nodes) + 1):
            inb = set(g.neighbors(i))
            for j in range(i+1, len(g.nodes) + 1):
                jnb = set(g.neighbors(j))
                jnb.discard(i)
                diff = jnb ^ inb  # Symmetric difference
                diff.discard(j)

                for k in diff:
                    # TODO: On dense graphs one could use the complementary graph...
                    self.add_clause(-self.merge[i][j], -self.tord(i, k), *self.edge_clause(i, j, k))
                    self.add_clause(-self.merge[i][j], -self.tord(i, k), *self.edge_clause(i, k, j))

        self.encode_edge(n, d)
        self.encode_reds2(g)
        #self.perf(n)

        print(f"{self.clauses} / {self.vars}")
        self.write_header()

    def encode_reds1(self, g):
        # Maintain red arcs
        for i in range(1, len(g.nodes) + 1):
            for j in range(1, len(g.nodes) + 1):
                if i == j:
                    continue

                for k in range(1, len(g.nodes) + 1):
                    if j == k or i == k:
                        continue

                    for cd in range(1, self.d+1):
                        for m in range(1, len(g.nodes) + 1):
                            if i == m or j == m or m == k:
                                continue

                            self.add_clause(-self.tord(i, j), -self.tord(j, k), -self.tord(j, m), -self.edge[i][k][cd][m],
                                            *self.edge_clause(j, m, k))

        # Transfer red arcs
        for i in range(1, len(g.nodes) + 1):
            for j in range(i + 1, len(g.nodes) + 1):
                if i == j:
                    continue

                for k in range(1, len(g.nodes) + 1):
                    if k == i or k == j:
                        continue
                    for cd in range(1, self.d + 1):
                        for m in range(1, len(g.nodes) + 1):
                            if m == i or m == k or m == j:
                                continue

                            self.add_clause(-self.merge[i][j], -self.tord(m, i), -self.tord(i, k), -self.edge[m][i][cd][k],
                                            *self.edge_clause(i, j, k))
                            self.add_clause(-self.merge[i][j], -self.tord(m, i), -self.tord(i, k),
                                            -self.edge[m][i][cd][k],
                                            *self.edge_clause(i, k, j))

    def encode_reds2(self, g):
        auxes = {}
        for i in range(1, len(g.nodes) + 1):
            auxes[i] = {}
            for j in range(i+1, len(g.nodes) + 1):
                if j == i:
                    continue
                c_aux = self.add_var()
                auxes[i][j] = c_aux
                for cd in range(1, self.d + 1):
                    for k in range(1, len(g.nodes) + 1):
                        if k == j or i == k:
                            continue

                        self.add_clause(-self.tord(k, i), -self.tord(k, j), -self.edge[k][i][cd][j], c_aux)

        # Maintain red arcs
        for i in range(1, len(g.nodes) + 1):
            for j in range(1, len(g.nodes) + 1):
                if i == j:
                    continue

                for k in range(1, len(g.nodes) + 1):
                    if j == k or i == k:
                        continue

                    for cd in range(1, self.d+1):
                        for m in range(k + 1, len(g.nodes) + 1):
                            if i == m or j == m:
                                continue

                            self.add_clause(-self.tord(i, j), -self.tord(j, k), -self.tord(j, m),
                                            -self.edge[i][k][cd][m],
                                            *self.edge_clause(j, k, m))
                            self.add_clause(-self.tord(i, j), -self.tord(j, k), -self.tord(j, m),
                                            -self.edge[i][k][cd][m],
                                            *self.edge_clause(j, m, k))

        # Transfer red arcs
        for i in range(1, len(g.nodes) + 1):
            for j in range(i + 1, len(g.nodes) + 1):
                if i == j:
                    continue

                for k in range(1, len(g.nodes) + 1):
                    if k == i or k == j:
                        continue

                    if i < k:
                        # We can make this ternary by doubling the aux vars and implying i < k that way
                        self.add_clause(-self.merge[i][j], -self.tord(i, k), -auxes[i][k], *self.edge_clause(i, j, k))
                        self.add_clause(-self.merge[i][j], -self.tord(i, k), -auxes[i][k], *self.edge_clause(i, k, j))
                    else:
                        self.add_clause(-self.merge[i][j], -self.tord(i, k), -auxes[k][i], *self.edge_clause(i, j, k))
                        self.add_clause(-self.merge[i][j], -self.tord(i, k), -auxes[k][i], *self.edge_clause(i, k, j))

    def decode(self, model, g, d):
        g = g.copy()
        unmap = {}
        for u, v in self.node_map.items():
            unmap[v] = u

        # Find merge targets and elimination order
        mg = {}
        od = list(range(1, len(g.nodes) + 1))

        def find_ord(x, y):
            if x < y:
                return -1 if model[self.ord[x][y]] else 1
            else:
                return 1 if model[self.ord[y][x]] else -1

        for i in range(1, len(g.nodes) + 1):
            for j in range(i+1, len(g.nodes) + 1):
                if model[self.merge[i][j]]:
                    if i in mg:
                        print("Error, double merge!")
                    mg[i] = j

        od.sort(key=cmp_to_key(find_ord))

        # Perform contractions, last node needs not be contracted...
        for u, v in g.edges:
            g[u][v]['red'] = False

        c_max = 0
        step = 1
        for n in od[:-1]:
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

            # Get red edges
            reds = defaultdict(set)
            for i in range(1, len(od)+1):
                x = self.node_map[n]
                if i == x:
                    continue
                for j in range(1, len(od)+1):
                    if j == i or j == x:
                        continue

                    for cd in range(1, d+1):
                        if model[self.edge[x][i][cd][j]]:
                            reds[unmap[i]].add(unmap[j])

            # Verify reds
            # for i, cl in reds.items():
            #     for j in cl:
            #         if not g.has_edge(i, j) or not g[i][j]['red']:
            #             print(f"One red too many {step} ({self.node_map[n]}) ({self.node_map[i]}, {self.node_map[j]})")
            for u, v in g.edges:
                if g[u][v]['red'] and not (u in reds and v in reds[u]):
                    print(f"One red too few {step} ({self.node_map[n]}) ({self.node_map[u]}, {self.node_map[v]})")

            # Count reds...
            for u in g.nodes:
                cc = 0
                for v in g.neighbors(u):
                    if g[u][v]['red']:
                        cc += 1
                if cc > d:
                    print(f"Exceeded bound in step {step}")
                c_max = max(c_max, cc)
            step += 1
        print(f"Done {c_max}/{d}")
