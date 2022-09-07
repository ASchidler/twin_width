from networkx import Graph
from functools import cmp_to_key
from pysat.formula import CNF, IDPool
from pysat.card import ITotalizer, CardEnc, EncType
import tools
import subprocess
import time


# TODO: Symmetry breaking: If two consecutive contractions have to node with red edges in common -> lex order
class TwinWidthEncoding:
    def __init__(self):
        self.edge = None
        self.ord = None
        self.merge = None
        self.node_map = None
        self.pool = IDPool()
        self.totalizer = None
        self.merged = None

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
        self.merged = [{} for _ in range(0, len(g.nodes) + 1)]

        for i in range(1, len(g.nodes) + 1):
            for j in range(i + 1, len(g.nodes) + 1):
                for k in range(1, len(g.nodes) + 1):
                    self.edge[k][i][j] = self.pool.id(f"edge{k}_{i}_{j}")
                self.ord[i][j] = self.pool.id(f"ord{i}_{j}")
                self.merge[i][j] = self.pool.id(f"merge{i}_{j}")

        for t in range(1, len(g.nodes) + 1):
            for i in range(1, len(g.nodes) + 1):
                self.merged[i][t] = self.pool.id(f"merged{i}_{t}")

    def encode_merged(self, n, formula):
        for i in range(1, n + 1):
            formula.append([self.merged[i][n]])

        for t in range(1, n + 1):
            for i in range(1, n + 1):
                helper_clause = [self.merged[i][t]]
                for j in range(1, n + 1):
                    if i == j:
                        continue
                    if t > 1:
                        formula.append([-self.merged[i][t], self.tord(i, j), self.merged[j][t-1]])
                        formula.append([-self.merged[j][t - 1], self.pool.id(f"mh_{i}_{j}_{t}")])
                        formula.append([-self.tord(i, j), self.pool.id(f"mh_{i}_{j}_{t}")])
                        helper_clause.append(-self.pool.id(f"mh_{i}_{j}_{t}"))
                    else:
                        helper_clause.append(-self.tord(i, j))
                        formula.append([-self.merged[i][1], self.tord(i, j)])
                    formula.append([-self.tord(i, j), -self.merged[j][t], self.merged[i][t]])

                formula.append(helper_clause)

    def tord(self, i, j):
        if i < j:
            return self.ord[i][j]

        return -self.ord[j][i]

    def tedge(self, n, i, j):
        if i < j:
            return self.edge[n][i][j]

        return self.edge[n][j][i]

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

                    formula.append([-self.tord(i, j), -self.tord(j, k), self.tord(i, k)])
        
        # Merge/ord relationship
        for i in range(1, len(g.nodes) + 1):            
            for j in range(i+1, len(g.nodes) + 1):
                formula.append([-self.merge[i][j], self.tord(i, j)])
                
        # single merge target
        for i in range(1, len(g.nodes)):
            formula.extend(CardEnc.atleast([self.merge[i][j] for j in range(i + 1, len(g.nodes) + 1)], bound=1, vpool=self.pool))
            formula.extend(tools.amo_commander([self.merge[i][j] for j in range(i + 1, len(g.nodes) + 1)], self.pool))

        # Create red arcs
        for i in range(1, len(g.nodes) + 1):
            inb = set(g.neighbors(i))
            for j in range(i+1, len(g.nodes) + 1):
                jnb = set(g.neighbors(j))
                jnb.discard(i)
                diff = jnb ^ inb  # Symmetric difference
                diff.discard(j)

                for t in range(1, len(g.nodes) + 1):
                    for k in diff:
                        # TODO: On dense graphs one could use the complementary graph...
                        formula.append([-self.merge[i][j], -self.merged[i][t], self.merged[k][t], self.merged[j][t], self.tedge(t, j, k)])

        self.encode_reds2(g, formula)
        self.encode_merged(len(g.nodes), formula)

        # Encode counters
        self.totalizer = {}
        for i in range(1, len(g.nodes)): # As last one is the root, no counter needed
            self.totalizer[i] = {}
            for x in range(1, len(g.nodes) + 1):
                vars = [self.tedge(i, x, y) for y in range(1, len(g.nodes)+1) if x != y]
                self.totalizer[i][x] = ITotalizer(vars, ubound=d, top_id=self.pool.id(f"totalizer{i}_{x}"))
                formula.extend(self.totalizer[i][x].cnf)
                self.pool.occupy(self.pool.top-1, self.totalizer[i][x].top_id)

        self.sb_ord(n, formula)
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
                if slv.solve(assumptions=self.get_card_vars(i)):
                    cb = i
                    if verbose:
                        print(f"Found {i}")
                    if check:
                        mx, od, mg = self.decode(slv.get_model(), g, i, verbose)
                else:
                    if verbose:
                        print(f"Failed {i}")
                        print(f"Finished cycle in {time.time() - start}")
                    break

                if verbose:
                    print(f"Finished cycle in {time.time() - start}")

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
            vars.extend([-x.rhs[d] for x in v.values()])

        return vars

    def encode_reds2(self, g, formula):
        for i in range(1, len(g.nodes) + 1):
            for j in range(i+1, len(g.nodes) + 1):
                formula.append([-self.merge[i][j], self.pool.id(f"is_target{j}")])

        for i in range(1, len(g.nodes) + 1):
            helpers = [-self.pool.id(f"is_target{i}")]
            for j in range(1, i):
                helpers.append(self.merged[j][i])
            formula.append(helpers)

        # Maintain red arcs
        for t in range(1, len(g.nodes) + 1):
            for i in range(1, len(g.nodes) + 1):
                for j in range(1, len(g.nodes) + 1):
                    if i == j:
                        continue
                    if t < len(g.nodes):
                        formula.append([-self.tedge(t, i, j), self.merged[i][t+1], self.merged[j][t+1], self.tedge(t+1, i, j)])

                    if i < j and t > 1:
                        for k in range(1, len(g.nodes) + 1):
                            if k == i or k == j:
                                continue
                            formula.append([-self.pool.id(f"is_target{j}"), -self.merge[i][j], -self.merged[i][t], self.merged[j][t], self.merged[k][t], -self.tedge(t-1, i, k), self.tedge(t, j, k)])

    def sb_ord(self, n, formula):
        for i in range(1, n):
            formula.append([self.ord[i][n]])
            # TODO: Can we do the same for the second to last?

    def sb_grid(self, g, n, formula):
        smallest = {x: self.pool.id(f"smallest{x}") for x in range(1, n+1)}
        formula.extend(CardEnc.atmost(smallest.values()))

        for i in range(1, n+1):
            for j in range(i+1, n+1):
                formula.append([-self.tord(i, j), -smallest[j]])
                formula.append([-self.tord(j, i), -smallest[i]])

        import math
        width = math.sqrt(n) // 2

        quadrant = [z for (x, y), z in self.node_map.items() if y <= width and z <= width]
        formula.append([smallest[i] for i in quadrant])

    def decode(self, model, g, d, verbose=False):
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
                if model[self.merge[i][j]]:
                    if i in mg:
                        print("Error, double merge!")
                    mg[i] = j

        od.sort(key=cmp_to_key(find_ord))

        # Perform contractions, last node needs not be contracted...
        for u, v in g.edges:
            g[u][v]['red'] = False

        c_max = 0
        cnt = 0
        for i, n in enumerate(od[:-1]):
            if not model[self.merged[n][i+1]]:
                print("Merge mismatch")
            for n2 in od[i+1:]:
                if model[self.merged[n2][i+1]]:
                    print("Unmerged mismatch")
            t = unmap[mg[n]]
            n = unmap[n]
            if verbose:
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
                    lits = [self.tord(x, i) for x in diff]
                    formula.append([-self.merge[i][j], *lits])
