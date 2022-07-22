import sys

import networkx
import networkx as nx
from networkx import DiGraph
from functools import cmp_to_key
from pysat.formula import CNF, IDPool
from pysat.card import ITotalizer, CardEnc, EncType
import tools
import subprocess
import time
import networkx.algorithms.bipartite as bp

# TODO: Symmetry breaking: If two consecutive contractions have to node with red edges in common -> lex order
class TwinWidthEncoding:
    def __init__(self):
        self.edge = None
        self.ord = None
        self.merge = None
        self.node_map = None
        self.pool = IDPool()
        self.totalizer = None
        self.partition1 = None
        self.partition2 = None
        self.partition_sep = 0

    def remap_graph(self, g):
        try:
            self.partition1, self.partition2 = bp.sets(g)
        except networkx.AmbiguousSolution:
            self.partition1 = set(x for x in g.nodes if x.startwith("v"))
            self.partition2 = set(x for x in g.nodes if x.startwith("c"))

        self.node_map = {}
        cnt = 1
        gn = DiGraph()

        self.node_map = {n: cnt+1 for (cnt, n) in enumerate(self.partition1)}
        self.partition_sep = max(self.node_map.values())
        self.node_map = {n: cnt+1+len(self.partition1) for (cnt, n) in enumerate(self.partition2)}

        for u, v in g.edges():
            if u not in self.node_map:
                self.node_map[u] = cnt
                cnt += 1
            if v not in self.node_map:
                self.node_map[v] = cnt
                cnt += 1

            gn.add_edge(self.node_map[u], self.node_map[v], **g[u][v])

        return gn

    def init_var(self, g):
        if self.partition1 is None:
            try:
                self.partition1, self.partition2 = bp.sets(g)
            except networkx.AmbiguousSolution:
                self.partition1 = set(x for x in g.nodes if x.startwith("v"))
                self.partition2 = set(x for x in g.nodes if x.startwith("c"))

        self.edge = [[{} for _ in range(0, len(g.nodes) + 1)]  for _ in range(0, len(g.nodes) + 1)]
        self.ord = [{} for _ in range(0, len(g.nodes) + 1)]
        self.merge = [{} for _ in range(0, len(g.nodes) + 1)]

        for i in range(1, len(g.nodes) + 1):
            for j in range(i + 1, len(g.nodes) + 1):
                self.ord[i][j] = self.pool.id(f"ord{i}_{j}")
                self.ord[j][i] = -self.ord[i][j]
                if (i <= self.partition_sep and j <= self.partition_sep) or (i > self.partition_sep and j > self.partition_sep):
                    self.merge[i][j] = self.pool.id(f"merge{i}_{j}")

                if i <= self.partition_sep and j > self.partition_sep:
                    for k in range(1, len(g.nodes) + 1):
                        self.edge[k][i][j] = self.pool.id(f"edge{k}_{i}_{j}")
                        self.edge[k][j][i] = self.edge[k][i][j]
        print("x")
    # def tmerge(self, i, j):
    #     if i < j:
    #         return self.merge[i][j]
    # 
    #     return self.merge[j][i]

    def encode(self, g, d):
        g = self.remap_graph(g)
        n = len(g.nodes)
        self.init_var(g)
        formula = CNF()

        # Encode relationships
        # Transitivity
        for i in range(1, len(g.nodes) + 1):
            for j in range(1, len(g.nodes) + 1):
                if i == j:
                    continue
                for k in range(1, len(g.nodes) + 1):
                    if i == k or j == k:
                        continue

                    formula.append([-self.ord[i][j], -self.ord[j][k], self.ord[i][k]])
        
        # Merge/ord relationship
        for i in range(1, len(g.nodes) + 1):            
            for j in range(i+1, len(g.nodes) + 1):
                if j in self.merge[i]:
                    formula.append([-self.merge[i][j], self.ord[i][j]])
                
        # single merge target
        for i in range(1, len(g.nodes)):
            if len(self.merge[i]) > 0:
                formula.extend(CardEnc.atleast([self.merge[i][j] for j in range(i + 1, len(g.nodes) + 1) if j in self.merge[i]], bound=1, vpool=self.pool))
            formula.extend(tools.amo_commander([self.merge[i][j] for j in range(i+ 1, len(g.nodes) + 1) if j in self.merge[i]], self.pool))

        # Create red arcs
        for i in range(1, len(g.nodes) + 1):
            suc = set(g.successors(i))
            pred = set(g.predecessors(i))

            for j in range(i+1, len(g.nodes) + 1):
                jsuc = set(g.successors(j))
                jpred = set(g.predecessors(j))
                diff = (jpred ^ suc) | (jsuc ^ pred)
                diff.discard(i)
                diff.discard(j)

                for k in diff:
                    # TODO: On dense graphs one could use the complementary graph...
                    if j in self.merge[i] and k in self.edge[i][j]:
                        formula.append([-self.merge[i][j], -self.ord[i][k], self.edge[i][j][k]])

        self.encode_reds2(g, formula)
        #self.perf(n, formula)

        # Encode counters
        self.totalizer = {}
        for i in range(1, len(g.nodes)): # As last one is the root, no counter needed
            self.totalizer[i] = {}
            for x in range(1, len(g.nodes) + 1):
                vars = [self.edge[i][x][y] for y in range(1, len(g.nodes)+1) if y in self.edge[i][x]]
                self.totalizer[i][x] = ITotalizer(vars, ubound=d, top_id=self.pool.id(f"totalizer{i}_{x}"))
                formula.extend(self.totalizer[i][x].cnf)
                self.pool.occupy(self.pool.top-1, self.totalizer[i][x].top_id)

        #self.sb_grid(g, n, formula)
        self.sb_ord(n, formula)
        #self.sb_sth(g, formula, d)
        #print(f"{len(formula.clauses)} / {formula.nv}")
        return formula

    def run(self, g, solver, start_bound, verbose=True, check=True, lb=0):
        start = time.time()
        formula = self.encode(g, start_bound)
        cb = start_bound

        if verbose:
            print(f"Created encoding in {time.time() - start}")
        od = None
        mg = None
        with solver() as slv:
            slv.append_formula(formula)
            for i in range(start_bound, lb-1, -1):
                if verbose:
                    print(f"{slv.nof_clauses()}/{slv.nof_vars()}")
                slv.append_formula([[x] for x in self.get_card_vars(i)])

                if slv.solve():
                    cb = i
                    if verbose:
                        print(f"Found {i}")
                        sys.stdout.flush()
                    if check:
                        mx, od, mg = self.decode(slv.get_model(), g, i)
                        print(f"{mx}")
                else:
                    if verbose:
                        print(f"Failed {i}")
                        print(f"Finished cycle in {time.time() - start}")
                        sys.stdout.flush()
                    break

                if verbose:
                    print(f"Finished cycle in {time.time() - start}")
                sys.stdout.flush()

        for v in self.totalizer.values():
            for t in v.values():
                t.delete()

        if od is None:
            return cb
        else:
            return cb, od, mg

    def get_card_vars(self, d):
        vars = []
        for v in self.totalizer.values():
            vars.extend([-x.rhs[d] for x in v.values() if d < len(x.rhs)])

        return vars

    def encode_reds2(self, g, formula):
        auxes = {}
        for i in range(1, len(g.nodes) + 1):
            auxes[i] = {}
            for j in range(i+1, len(g.nodes) + 1):
                if j == i:
                    continue
                c_aux = self.pool.id(f"a_red{i}_{j}")
                auxes[i][j] = c_aux
                for k in range(1, len(g.nodes) + 1):
                    if k == j or i == k:
                        continue

                    if j in self.edge[k][i]:
                        formula.append([-self.ord[k][i], -self.ord[k][j], -self.edge[k][i][j], c_aux])

        # Maintain red arcs
        for i in range(1, len(g.nodes) + 1):
            for j in range(1, len(g.nodes) + 1):
                if i == j:
                    continue

                for k in range(1, len(g.nodes) + 1):
                    if j == k or i == k:
                        continue

                    for m in range(k + 1, len(g.nodes) + 1):
                        if i == m or j == m:
                            continue

                        if m in self.edge[j][k]:
                            formula.append([-self.ord[i][j], -self.ord[j][k], -self.ord[j][m], -self.edge[i][k][m],
                                             self.edge[j][k][m]])

        # Transfer red arcs
        for i in range(1, len(g.nodes) + 1):
            for j in range(i + 1, len(g.nodes) + 1):
                if i == j:
                    continue

                for k in range(1, len(g.nodes) + 1):
                    if k == i or k == j:
                        continue

                    if j in self.merge[i] and k in self.edge[i][j]:
                        if i < k:
                            # We can make this ternary by doubling the aux vars and implying i < k that way
                            formula.append([-self.merge[i][j], -self.ord[i][k], -auxes[i][k], self.edge[i][j][k]])
                        else:
                            formula.append([-self.merge[i][j], -self.ord[i][k], -auxes[k][i], self.edge[i][j][k]])

    def sb_ord(self, n, formula):
        for i in range(1, n):
            formula.append([self.ord[i][n]])
            # TODO: Can we do the same for the second to last?

    def sb_grid(self, g, n, formula):
        smallest = {x: self.pool.id(f"smallest{x}") for x in range(1, n+1)}
        formula.extend(CardEnc.atmost(smallest.values()))

        for i in range(1, n+1):
            for j in range(i+1, n+1):
                formula.append([-self.ord[i][j], -smallest[j]])
                formula.append([-self.ord[j][i], -smallest[i]])

        import math
        width = math.sqrt(n) // 2

        quadrant = [z for (x, y), z in self.node_map.items() if y <= width and z <= width]
        formula.append([smallest[i] for i in quadrant])

    def decode(self, model, g, d):
        g = g.copy()
        model = {abs(x): x > 0 for x in model}
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
                if j in self.merge[i]:
                    if model[self.merge[i][j]]:
                        if i in mg:
                            print("Error, double merge!")
                        mg[i] = j
        mg[self.partition_sep] = len(g.nodes)
        # for c_i in range(self.partition_sep, len(g.nodes)):
        #     mg[c_i] = len(g.nodes)
        od.sort(key=cmp_to_key(find_ord))

        # Perform contractions, last node needs not be contracted...
        for u, v in g.edges:
            g[u][v]['red'] = False

        c_max = 0
        cnt = 0
        for n in od[:-1]:
            t = unmap[mg[n]]
            n = unmap[n]
            print(f"{n} => {t}")
            # graph_export, line_export = tools.dot_export(g, t, n)
            # with open(f"progress_{cnt}.dot", "w") as f:
            #     f.write(graph_export)
            # with open(f"progress_{cnt}.png", "w") as f:
            #     subprocess.run(["dot", "-Kfdp", "-Tpng", f"progress_{cnt}.dot"], stdout=f)

            # with open(f"line_{cnt}.dot", "w") as f:
            #     f.write(line_export)
            # with open(f"line_{cnt}.png", "w") as f:
            #     subprocess.run(["dot", "-Tpng", f"line_{cnt}.dot"], stdout=f)

            tns = set(g.successors(t))
            tnp = set(g.predecessors(t))
            nns = set(g.successors(n))
            nnp = set(g.predecessors(n))

            nn = nns | nnp
            tn = tns | tnp

            for v in nn:
                if v != t:
                    # Red remains, should edge exist
                    if (v in g[n] and g[n][v]['red']) or v not in tn or (v in nns and v not in tns) or (v in nnp and v not in tnp):
                        if g.has_edge(t, v):
                            g[t][v]['red'] = True
                        else:
                            g.add_edge(t, v, red=True)

                        if g.has_edge(v, t):
                            g[v][t]['red'] = True
                        else:
                            g.add_edge(v, t, red=True)

            for v in tn:
                if v not in nn:
                    if g.has_edge(t, v):
                        g[t][v]['red'] = True
                    else:
                        g.add_edge(t, v, red=True)

                    if g.has_edge(v, t):
                        g[v][t]['red'] = True
                    else:
                        g.add_edge(v, t, red=True)

            g.remove_node(n)

            # Count reds...
            for u in g.nodes:
                cc = 0
                for v in g.neighbors(u):
                    if g[u][v]['red']:
                        cc += 1
                c_max = max(c_max, cc)
            cnt += 1
        #print(f"Done {c_max}/{d}")
        od = [unmap[x] for x in od]
        mg = {unmap[x]: unmap[y] for x, y in mg.items()}
        return c_max, od, mg

    def sb_sth(self, g, formula, ub):
        n = len(g.nodes)

        for i in range(1, len(g.nodes) + 1):
            inb = set(g.neighbors(i))
            for j in range(i+1, len(g.nodes) + 1):
                jnb = set(g.neighbors(j))
                jnb.discard(i)
                diff = jnb ^ inb  # Symmetric difference
                diff.discard(j)

                if len(diff) > ub:
                    lits = [self.ord[x][i] for x in diff]
                    formula.append([-self.merge[i][j], *lits])
