import os

from networkx import Graph
from functools import cmp_to_key
from pysat.formula import CNF, IDPool, WCNF
from pysat.card import ITotalizer, CardEnc, EncType
import tools
import subprocess
import time


# TODO: Symmetry breaking: If two consecutive contractions have to node with red edges in common -> lex order
class TwinWidthEncoding:
    def     __init__(self, use_sb_static=True, use_sb_static_full=True, use_sb_red=True):
        self.edge = None
        self.ord = None
        self.merge = None
        self.node_map = None
        self.pool = IDPool()
        self.totalizer = None
        self.static_cards = None
        self.initial = None
        self.use_sb_static = use_sb_static
        self.use_sb_static_full = use_sb_static_full
        self.use_sb_red = use_sb_red

    def remap_graph(self, g):
        self.node_map = {}
        cnt = 1
        gn = Graph()

        for u in sorted(g.nodes):
            self.node_map[u] = cnt
            gn.add_node(cnt)
            cnt += 1

        for u, v in sorted(g.edges()):
            if u not in self.node_map:
                self.node_map[u] = cnt
                cnt += 1
            if v not in self.node_map:
                self.node_map[v] = cnt
                cnt += 1

            gn.add_edge(self.node_map[u], self.node_map[v])

        return gn

    def init_var(self, g, parts):
        self.edge = [[{} for _ in range(0, len(g.nodes) + 1)]  for _ in range(0, len(g.nodes) + 1)]
        self.ord = [{} for _ in range(0, len(g.nodes) + 1)]
        self.merge = [{} for _ in range(0, len(g.nodes) + 1)]
        self.static_cards = [{} for _ in range(0, len(g.nodes) + 1)]
        self.initial = [{} for _ in range(0, len(g.nodes) + 1)]
        self.static_cards = [{} for _ in range(0, len(g.nodes) + 1)]

        for i in range(1, len(g.nodes) + 1):
            for j in range(i + 1, len(g.nodes) + 1):
                for k in range(1, len(g.nodes) + 1):
                    self.edge[k][i][j] = self.pool.id(f"edge{k}_{i}_{j}")
                self.ord[i][j] = self.pool.id(f"ord{i}_{j}")
                if parts is None or j == len(g.nodes) + 1 or any(i in x and j in x for x in parts) or \
                        (any(i == max(x) for x in parts) and any(j == max(x) for x in parts)):
                    self.merge[i][j] = self.pool.id(f"merge{i}_{j}")

        if self.use_sb_static:
            for i in range(1, len(g.nodes) + 1):
                n1nb = set(g.neighbors(i))
                for j in range(i + 1, len(g.nodes) + 1):
                    n2nb = set(g.neighbors(j))
                    sd = n1nb ^ n2nb
                    sd.discard(i)
                    sd.discard(j)
                    self.initial[i][j] = sd

    def tord(self, i, j):
        if i < j:
            return self.ord[i][j]

        return -self.ord[j][i]

    def tedge(self, n, i, j):
        if i < j:
            return self.edge[n][i][j]

        return self.edge[n][j][i]

    def encode(self, g, d, od, mg, skip_cards=False, parts=None, reds=None):
        g = self.remap_graph(g)
        n = len(g.nodes)
        self.init_var(g, parts)
        formula = CNF()

        if parts is not None:
            maxes = {max(x) for x in parts}
            for cn in range(1, len(g.nodes) + 1):
                if cn not in maxes:
                    for cm in maxes:
                        formula.append([self.tord(cn, cm)])

        if od is not None:
            for i, cn in enumerate(od):
                for cn2 in od[i+1:]:
                    formula.append([self.tord(cn, cn2)])

            # done = set()
            # for cn in od:
            #     done.add(cn)
            #     for i in range(1, len(g.nodes) + 1):
            #         if i not in done and i != cn:
            #             formula.append([self.tord(cn, i)])

        if mg is not None:
            for k, v in mg.items():
                formula.append([self.merge[k][v]])

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
                if j in self.merge[i]:
                    formula.append([-self.merge[i][j], self.tord(i, j)])
                
        # single merge target
        for i in range(1, len(g.nodes)):
            formula.extend(
                CardEnc.atleast([self.merge[i][j] for j in range(i + 1, len(g.nodes) + 1) if j in self.merge[i]],
                                bound=1, vpool=self.pool))
            if len(self.merge[i]) > 1:
                formula.extend(CardEnc.atmost([self.merge[i][j] for j in range(i + 1, len(g.nodes) + 1) if j in self.merge[i]], bound=1, vpool=self.pool))

        # Create red arcs
        for i in range(1, len(g.nodes) + 1):
            inb = set(g.neighbors(i))
            for j in range(i+1, len(g.nodes) + 1):
                if j not in self.merge[i]:
                    continue

                jnb = set(g.neighbors(j))
                jnb.discard(i)
                diff = jnb ^ inb  # Symmetric difference
                diff.discard(j)

                for k in diff:
                    formula.append([-self.merge[i][j], -self.tord(i, k), self.tedge(i, j, k)])

        self.encode_reds2(g, formula, d)

        if reds is not None:
            for i in range(1, len(g.nodes) + 1):
                for j, k in reds:
                    j = self.node_map[j]
                    k = self.node_map[k]

                    if i != j and i != k:
                        formula.append([-self.tord(i, j), -self.tord(i, k), self.tedge(i, j, k)])
            for j, k in reds:
                j = self.node_map[j]
                k = self.node_map[k]
                assert f"a_red{j}_{k}" in self.pool.obj2id and f"a_red{k}_{j}" in self.pool.obj2id
                formula.append([-self.tord(j, k), self.pool.id(f"a_red{j}_{k}")])
                formula.append([-self.tord(k, j), self.pool.id(f"a_red{k}_{j}")])

        # Encode counters
        if not skip_cards:
            self.totalizer = {}
            for i in range(1, len(g.nodes)): # At last one is the root, no counter needed
                self.totalizer[i] = {}
                for x in range(1, len(g.nodes) + 1):
                    if i == x:
                        continue
                    vars = [self.tedge(i, x, y) for y in range(1, len(g.nodes)+1) if x != y]

                    self.totalizer[i][x] = ITotalizer(vars, ubound=d, top_id=self.pool.id(f"totalizer{i}_{x}"))
                    formula.extend(self.totalizer[i][x].cnf)
                    self.pool.occupy(self.pool.top-1, self.totalizer[i][x].top_id)

        self.sb_ord(n, formula)
        if self.sb_static(d, formula):
            self.sb_static(d, formula)
        if self.use_sb_red:
            self.sb_reds(n, formula)

        return formula

    def wcnf_export(self, g, start_bound, filename, export_cards):
        formula = self.encode(g, start_bound, None, None)
        wcnf = WCNF() if not export_cards else CNF()
        wcnf.extend(formula)

        if not export_cards:
            softs = [self.pool.id(f"softs_{i}") for i in range(1, start_bound+1)]
            for cs in softs:
                wcnf.append([-cs], 1)

            for ctots in self.totalizer.values():
                for ctot in ctots.values():
                    for cd in range(0, start_bound):
                        wcnf.append([-ctot.rhs[cd], softs[cd]])

        wcnf.to_file(filename)
        if export_cards:
            with open(filename+".cards", "w") as outp:
                for i in range(1, len(g.nodes)): # At last one is the root, no counter needed
                    for x in range(1, len(g.nodes) + 1):
                        if i == x:
                            continue
                        vars = [self.tedge(i, x, y) for y in range(1, len(g.nodes)+1) if x != y]
                        outp.write(" ".join(str(x) for x in vars))
                        outp.write(f" <= d"+ os.linesep)

    def run(self, g, solver, start_bound, verbose=True, check=True, lb=0, i_od=None, i_mg=None, write=False, parts=None, reds=None, steps_limit=None):
        start = time.time()
        formula = self.encode(g, start_bound, i_od, i_mg, parts=parts, reds=reds)
        cb = start_bound

        if verbose:
            print(f"Created encoding in {time.time() - start}")

        if write:
            for cv in self.get_card_vars(start_bound):
                formula.append([cv])
            formula.to_file("test.cnf")
        od = None
        mg = None
        with solver() as slv:
            slv.append_formula(formula)
            for i in range(start_bound, lb-1, -1):
                if verbose:
                    print(f"{slv.nof_clauses()}/{slv.nof_vars()}")
                for cv in self.get_card_vars(i):
                    slv.add_clause([cv])

                if self.use_sb_static and self.use_sb_static_full:
                    for n1, vl in enumerate(self.static_cards):
                        for n2, cards in vl.items():
                            if n2 in self.merge[n1]:
                                slv.add_clause([-self.merge[n1][n2], -cards[i]])

                if slv.solve():
                    if verbose:
                        print(f"Found {i}")
                    if check:
                        mx, od, mg = self.decode(slv.get_model(), g, i, verbose)
                    if self.use_sb_static and self.use_sb_static_full:
                        self.sb_static(i, None, cb, slv)
                    cb = i
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

    def encode_reds2(self, g, formula, d):
        auxes = {}
        for i in range(1, len(g.nodes) + 1):
            auxes[i] = {}
            for j in range(1, len(g.nodes) + 1):
                if j == i:
                    continue
                c_aux = self.pool.id(f"a_red{i}_{j}")
                auxes[i][j] = c_aux
                for k in range(1, len(g.nodes) + 1):
                    if k == j or i == k:
                        continue

                    # formula.append([-self.tord(k, i), -self.tord(k, j), -self.tedge(k, i, j), c_aux])
                    # formula.append([-self.tedge(k, i, j), c_aux])
                    formula.append([-self.tord(i, j), -self.tedge(k, i, j), c_aux])
            vars = [auxes[i][j] for j in range(1, len(g.nodes) + 1) if i != j]
            c_tot = ITotalizer(vars, ubound=d, top_id=self.pool.id(f"totalizer_aux_{i}"))
            formula.extend(c_tot.cnf)
            self.pool.occupy(self.pool.top - 1, c_tot.top_id)

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

                        formula.append([-self.tord(i, j), -self.tord(j, k), -self.tord(j, m), -self.tedge(i, k, m),
                                         self.tedge(j, k, m)])

                        # formula.append([-self.merge[k][m], -self.tord(k, i), -self.tedge(j, k, i), self.tedge(k, m, i)])

        # Transfer red arcs
        for i in range(1, len(g.nodes) + 1):
            for j in range(i + 1, len(g.nodes) + 1):
                if i == j or j not in self.merge[i]:
                    continue

                for k in range(1, len(g.nodes) + 1):
                    if k == i or k == j:
                        continue

                    formula.append([-self.merge[i][j], -self.tord(i, k), -auxes[i][k], self.tedge(i, j, k)])

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

    def sb_static(self, d, form, old_d=None, slv=None):
        for i, vl in enumerate(self.initial):
            for j, sd in vl.items():
                if len(sd) > d:
                    if (len(sd) - d > 1 and (old_d is None or len(sd) - old_d <= 1)) and self.use_sb_static_full:
                        lits = [self.tord(i, x) for x in sd]
                        with ITotalizer(lits, ubound=d, top_id=self.pool.id(f"static{i}_{j}")) as tot:
                            self.pool.occupy(self.pool.top - 1, tot.top_id)
                            self.static_cards[i][j] = list(tot.rhs)
                            if slv is None:
                                form.extend(tot.cnf)
                            else:
                                slv.append_formula(tot.cnf)
                        # cform, cards = tools.encode_cards_exact(self.pool, lits, d, f"static_{i}_{j}", rev=True, add_constraint=False)
                        # if slv is None:
                        #     form.extend(cform)
                        # else:
                        #     slv.append_formula(cform)
                        #
                        # self.static_cards[i][j] = cards
                    elif old_d is None or len(sd) <= old_d:
                        if j in self.merge[i]:
                            cl = [-self.merge[i][j]]
                            for x in sd:
                                cl.append(self.tord(x, i))
                            if slv is None:
                                form.append(cl)
                            else:
                                slv.add_clause(cl)

    def decode(self, model, g, d, verbose=False, reds=None):
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
                if j in self.merge[i] and model[self.merge[i][j]]:
                    if i in mg:
                        print("Error, double merge!")
                    mg[i] = j

        od.sort(key=cmp_to_key(find_ord))

        # Perform contractions, last node needs not be contracted...
        for u, v in g.edges:
            g[u][v]['red'] = False

        if reds is not None:
            for u, v in reds:
                g.add_edge(u, v, red=True)

        c_max = 0
        cnt = 0
        for c_step, n in enumerate(od[:-1]):
            t = unmap[mg[n]]
            n = unmap[n]
            if verbose:
                print(f"{n} => {t}")

            tn = set(g.neighbors(t))
            tn.discard(n)
            nn = set(g.neighbors(n))

            for v in nn:
                if v != t:
                    g.add_edge(t, v, red=v not in tn)

            for v in tn:
                if v not in nn:
                    g[t][v]['red'] = True
            g.remove_node(n)

            # Count reds...
            for u in g.nodes:
                cc = 0
                for v in g.neighbors(u):
                    if g[u][v]['red']:
                        assert model[self.tedge(od[c_step], u, v)]
                        cc += 1
                c_max = max(c_max, cc)
                if c_max > d:
                    print(f"Error: Bound exceeded {c_max}/{d}, Step {c_step+1}, Node {u}")

            cnt += 1

        return c_max, od, mg
