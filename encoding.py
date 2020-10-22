import base_encoding
from networkx import Graph
from functools import cmp_to_key


class TwinWidthEncoding(base_encoding.BaseEncoding):
    def __init__(self, stream):
        super().__init__(stream)

        self.edge = None
        self.ord = None
        self.merge = None
        self.root = None
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

    def init_var(self, g):
        self.edge = [[{} for _ in range(0, len(g.nodes) + 1)]  for _ in range(0, len(g.nodes) + 1)]
        self.ord = [{} for _ in range(0, len(g.nodes) + 1)]
        self.merge = [{} for _ in range(0, len(g.nodes) + 1)]
        self.root = [self.add_var() if i > 0 else None for i in range(0, len(g.nodes) + 1)]

        for i in range(1, len(g.nodes) + 1):
            for j in range(i + 1, len(g.nodes) + 1):
                for k in range(1, len(g.nodes) + 1):
                    self.edge[k][i][j] = self.add_var()
                self.ord[i][j] = self.add_var()
            for j in range(1, len(g.nodes) + 1):
                if i != j:
                    self.merge[i][j] = self.add_var()

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

    def encode(self, g, d):
        g = self.remap_graph(g)
        self.init_var(g)

        # Root
        root_clause = []
        for i in range(1, len(g.nodes) + 1):
            root_clause.append(self.root[i])
            for j in range(i+1, len(g.nodes) + 1):
                self.add_clause(-self.root[i], -self.root[j])
        self.add_clause(*root_clause)

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
            for j in range(1, len(g.nodes) + 1):
                if i == j:
                    continue
                self.add_clause(self.tord(i, j), -self.merge[i][j])
                
        # single merge target
        for i in range(1, len(g.nodes) + 1):
            clause = [self.root[i]]
            for j in range(1, len(g.nodes) + 1):
                if i == j:
                    continue
                clause.append(self.merge[i][j])
                for k in range(j+1, len(g.nodes) + 1):
                    if k == i or k == j:
                        continue
                    self.add_clause(-self.merge[i][j], -self.merge[i][k])
                self.add_clause(*clause)

        # Create red arcs
        for i in range(1, len(g.nodes) + 1):
            inb = set(g.neighbors(i))
            for j in range(1, len(g.nodes) + 1):
                jnb = set(g.neighbors(j))
                jnb.discard(i)
                diff = jnb ^ inb  # Symmetric difference
                diff.discard(j)

                for k in diff:
                    self.add_clause(-self.merge[i][j], -self.tord(i, k), self.tedge(i, j, k))

        # Maintain red arcs
        for i in range(1, len(g.nodes) + 1):
            for j in range(1, len(g.nodes) + 1):
                if i == j:
                    continue

                for k in range(1, len(g.nodes) + 1):
                    if j == k or i == k:
                        continue

                    for m in range(k+1, len(g.nodes) + 1):
                        if i == m or k == m or j == m:
                            continue

                        self.add_clause(-self.tord(i, j), -self.tord(j, k), -self.tord(j, m), -self.tedge(i, k, m),
                            self.tedge(j, k, m))

        # Transfer red arcs
        for i in range(1, len(g.nodes) + 1):
            for j in range(1, len(g.nodes) + 1):
                if i == j:
                    continue

                for k in range(1, len(g.nodes) + 1):
                    if k == i or k == j:
                        continue

                    # TODO: Is this correct
                    self.add_clause(-self.merge[i][j], -self.tedge(i, i, k), self.tedge(i, j, k))

        # Encode counters
        for i in range(1, len(g.nodes) + 1):
            # Map vars to full adjacency matrix
            vars = [[self.tedge(i, x, y) for x in range(1, len(g.nodes) + 1) if x != y] for y in range(1, len(g.nodes) + 1)]
            self.encode_cardinality_sat(d, vars)

        self.write_header()

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
            for j in range(1, len(g.nodes) + 1):
                if i == j:
                    continue

                if model[self.merge[i][j]]:
                    if i in mg:
                        print("Error, double merge!")
                    mg[i] = j

        od.sort(key=cmp_to_key(find_ord))

        # Perform contractions, last node needs not be contracted...
        for u, v in g.edges:
            g[u][v]['red'] = False

        c_max = 0
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

            # Count reds...
            for u in g.nodes:
                cc = 0
                for v in g.neighbors(u):
                    if g[u][v]['red']:
                        cc += 1
                c_max = max(c_max, cc)
        print(f"Done {c_max}/{d}")
