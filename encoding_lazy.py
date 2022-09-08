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
        self.permanent_edges = None
        self.permanent_tot = None

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
        self.permanent_edges = [{} for _ in range(0, len(g.nodes) + 1)]
        self.totalizer = {}

        for i in range(1, len(g.nodes) + 1):
            for j in range(i + 1, len(g.nodes) + 1):
                self.ord[i][j] = self.pool.id(f"ord{i}_{j}")
                self.merge[i][j] = self.pool.id(f"merge{i}_{j}")

        for i in range(1, len(g.nodes) + 1):
            for j in range(1, len(g.nodes) + 1):
                if i != j:
                    self.permanent_edges[i][j] = self.pool.id(f"pe{i}_{j}")

    def tord(self, i, j):
        if i < j:
            return self.ord[i][j]

        return -self.ord[j][i]

    def tedge(self, n, i, j):
        i, j = min(i, j), max(i, j)
        if j not in self.edge[n][i]:
            self.edge[n][i][j] = self.pool.id(f"edge{n}_{i}_{j}")

        return self.edge[n][i][j]

    def encode(self, g, d):
        g = self.remap_graph(g)
        self.remapped_g = g
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

                for k in diff:
                    # TODO: On dense graphs one could use the complementary graph...
                    formula.append([-self.merge[i][j], -self.tord(i, k), -self.tord(j, k), self.permanent_edges[j][k]])
                    formula.append([-self.merge[i][j], -self.tord(i, k), -self.tord(k, j), self.permanent_edges[k][j]])
                    formula.append([-self.merge[i][j], -self.tord(i, k), self.tedge(i, j, k)])

                for k in range(1, len(g.nodes) + 1):
                    if k == i or k == j:
                        continue

                    formula.append(
                        [-self.merge[i][j], -self.tord(i, k), -self.permanent_edges[i][k], -self.tord(j, k),
                         self.permanent_edges[j][k]])
                    formula.append(
                        [-self.merge[i][j], -self.tord(i, k), -self.permanent_edges[i][k], -self.tord(k, j),
                         self.permanent_edges[k][j]])

        self.permanent_tot = {}
        for i in range(1, len(g.nodes)): # As last one is the root, no counter needed
            vars = [self.permanent_edges[i][j] for j in range(1, len(g.nodes)+1) if i != j]
            self.permanent_tot[i] = ITotalizer(vars, ubound=d, top_id=self.pool.id(f"perm_totalizer{i}"))
            formula.extend(self.permanent_tot[i].cnf)
            self.pool.occupy(self.pool.top-1, self.permanent_tot[i].top_id)

        self.sb_ord(n, formula)
        self.sb_reds(n, formula)
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

            i = start_bound
            while i >= lb:
                if verbose:
                    print(f"{slv.nof_clauses()}/{slv.nof_vars()}")
                if slv.solve(assumptions=self.get_card_vars(i)):
                    cb = i

                    ret = self.decode(slv.get_model(), g, i, slv, verbose)
                    if ret != False:
                        if verbose:
                            print(f"Found {i}")
                        i -= 1
                        mx, od, mg = ret

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
        for v in self.permanent_tot.values():
            v.delete()

        if od is None:
            return cb
        else:
            return cb, od, mg

    def get_card_vars(self, d):
        vars = []
        for v in self.totalizer.values():
            vars.extend([-x.rhs[d] for x in v.values()])

        for v in self.permanent_tot.values():
            vars.append(-v.rhs[d])

        return vars

    def encode_propagation(self, v, g, d, solver):
        for j in range(1, v):
            # Transfer red arcs
            for k in range(1, len(g.nodes) + 1):
                if k == v or k == j:
                    continue

                solver.add_clause(
                    [-self.merge[j][v], -self.tord(j, k), -self.permanent_edges[j][k], self.tedge(j, v, k)])

        for j in range(1, len(g.nodes) + 1):
            if j == v:
                continue

            # Transfer red arcs
            for k in range(j+1, len(g.nodes) + 1):
                if k == v or k == j:
                    continue

                if k not in self.totalizer:
                    solver.add_clause([-self.merge[j][k], -self.tord(j, v), -self.permanent_edges[j][v], self.tedge(j, v, k)])

        # Maintain red arcs
        for j in range(1, len(g.nodes) + 1):
            if v == j:
                continue

            for k in range(1, len(g.nodes) + 1):
                if j == k or v == k:
                    continue

                for m in range(1, len(g.nodes) + 1):
                    if v == m or j == m or k == m:
                        continue

                    if m not in self.totalizer:
                        solver.add_clause(
                            [-self.tord(k, v), -self.tord(j, k), -self.tord(k, m), -self.tedge(j, v, m),
                             self.tedge(k, v, m)])

        # Encode counters
        self.totalizer[v] = {}
        for x in range(1, len(g.nodes) + 1):
            if x == v:
                continue
            vars = [self.tedge(x, v, y) for y in range(1, len(g.nodes)+1) if v != y]
            self.totalizer[v][x] = ITotalizer(vars, ubound=d, top_id=self.pool.id(f"totalizer{v}_{x}"))
            solver.append_formula(self.totalizer[v][x].cnf)
            solver.add_clause([-self.totalizer[v][x].rhs[d]])
            self.pool.occupy(self.pool.top-1, self.totalizer[v][x].top_id)

    def sb_reds(self, n, formula):
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if i == j:
                    continue

                for k in range(1, n + 1):
                    if k == i or k == j:
                        continue

                    formula.append([-self.tord(j, i), -self.tedge(i, j, k)])

    def sb_ord(self, n, formula):
        for i in range(1, n):
            formula.append([self.ord[i][n]])

    def decode(self, model, g, d, solver, verbose=False):
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
        for n in od[:-1]:
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
                        # mn, mu, mv = self.node_map[n], self.node_map[u], self.node_map[v]
                        # mu, mv = min(mu, mv), max(mu, mv)
                        # if mv in self.edge[mn][mu]:
                        #     if not model[self.tedge(mn, mu, mv)]:
                        #         print("Missing red edge")
                if cc > d:
                    if verbose:
                        print(f"Bound exceeded {cc}/{d}")
                    self.encode_propagation(self.node_map[u], self.remapped_g, d, solver)
                    return False
                c_max = max(c_max, cc)
            cnt += 1
        #print(f"Done {c_max}/{d}")
        od = [unmap[x] for x in od]
        mg = {unmap[x]: unmap[y] for x, y in mg.items()}
        return c_max, od, mg
