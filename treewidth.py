import sys

from pysat.formula import IDPool, CNF
import networkx.algorithms.clique as cq1
import networkx.algorithms.approximation.clique as cq2
from networkx import Graph, is_forest
from pysat.card import ITotalizer
import threading

def _ord(i, j, pool):
    if i < j:
        return pool.id(f"ord_{i}_{j}")
    else:
        return -pool.id(f"ord_{j}_{i}")


def _encode(g, pool):
    formula = CNF()

    # Encode edges
    for u, v in g.edges:
        formula.append([-_ord(u, v, pool), pool.id(f"arc_{u}_{v}")])
        formula.append([-_ord(v, u, pool), pool.id(f"arc_{v}_{u}")])

    # Rest
    for u in g.nodes:
        for v in g.nodes:
            if u == v:
                continue

            if u < v:
                # These clauses are just for speedup
                # Arcs cannot go in both directions
                formula.append([-pool.id(f"arc_{u}_{v}"), -pool.id(f"arc_{v}_{u}")])
                # Enforce arc direction from smaller to bigger ordered vertex
                formula.append([-_ord(u, v, pool), -pool.id(f"arc_{v}_{u}")])
                formula.append([-_ord(v, u, pool), -pool.id(f"arc_{u}_{v}")])

            for w in g.nodes:
                if u == w or v == w:
                    continue

                # Transitivity of ordering
                formula.append([-_ord(u, v, pool), -_ord(v, w, pool), _ord(u, w, pool)])

                # Additional edges due to linear ordering
                if v < w:
                    formula.append([-pool.id(f"arc_{u}_{v}"), -pool.id(f"arc_{u}_{w}"), pool.id(f"arc_{v}_{w}"), pool.id(f"arc_{w}_{v}")])

    return formula


def _encode_cliques(g, formula, pool):
    """Enforces lexicographical ordering of the cliques"""

    # This is probably slow at some point... Change to approx based on some heuristic?
    _, clique = max((len(c), c) for c in cq1.find_cliques(g))
    # clique = cq2.max_clique(g)

    # Put clique at the end of the ordering
    for u in g.nodes:
        if u in clique:
            continue

        for v in clique:
            formula.append([-_ord(u, v, pool)])

    # order clique lexicographically
    for u in clique:
        for v in clique:
            if u < v:
                formula.append([_ord(u, v, pool)])
                formula.append([pool.id(f"arc_{u}_{v}")])


def _decode(g, model, pool):
    model = {abs(x): x > 0 for x in model}
    # Establish ordering
    ordering = []

    for i in g.nodes:
        pos = 0
        for j in ordering:
            if (i < j and model[_ord(i, j, pool)]) or (i > j and not model[_ord(j, i, pool)]):
                break
            pos += 1

        ordering.insert(pos, i)

    bags = {n: {n} for n in ordering}
    tree = Graph()
    ps = {x: ordering.index(x) for x in ordering}

    # Add edges to bags
    for u, v in g.edges:
        if ps[v] < ps[u]:
            u, v = v, u
        bags[u].add(v)

    for n in ordering:
        tree.add_node(n)
        A = set(bags[n])
        if len(A) > 1:
            A.remove(n)
            _, nxt = min((ps[x], x) for x in A)

            bags[nxt].update(A)
            tree.add_edge(nxt, n)

    tw = max(len(x) for x in bags.values()) - 1
    return (tree, bags), tw


def _check(g, td):
    if not is_forest(td[0]):
        print("Not a tree")
        return False

    # Assuming a connected graph, it suffices to check every edge
    for u, v in g.edges():
        found = False
        for _, b in td[1].items():
            if u in b and v in b:
                found = True
                break

        # Edge not covered
        if not found:
            print(f"Edge {u}, {v} not covered")
            return False

    # Check connectedness
    for u in g.nodes():
        ns = set(k for k, v in td[1].items() if u in v)

        # For some reason weak connectedness does not work in networkx...
        found = set()
        q = [next(ns.__iter__())]

        while q:
            n = q.pop()
            found.add(n)
            nbs = set(td[0].adj[n])
            q.extend((nbs & ns) - found)

        if found != ns:
            print(f"{u} does not induce a connected component")
            return False

    return True


def solve(g, ub, slv, verbose=False, timeout=0):
    if len(g) == 1:
        n = next(iter(g.nodes))
        td = Graph()
        td.add_node(n)
        return (td, {n: {n}}), 0

    pool = IDPool()
    formula = _encode(g, pool)
    #_encode_cliques(g, formula, pool)

    cards = {}
    ub = min(ub, len(g.nodes)-1)
    ctop = pool.top + 1
    for n in g.nodes:
        tot = ITotalizer(lits=[pool.id(f"arc_{n}_{v}") for v in g.nodes if n != v], ubound=ub, top_id=ctop)
        ctop = tot.top_id + 1
        cards[n] = tot

    def interrupt(s):
        s.interrupt()

    best_model = None
    with slv() as solver:
        timer = None
        if timeout > 0:
            timer = threading.Timer(timeout, interrupt, [slv])
            timer.start()

        solver.append_formula(formula)
        for tot in cards.values():
            solver.append_formula(tot.cnf)

        for cb in range(ub, 0, -1):
            if verbose:
                print(f"Searching for {cb}")
                sys.stdout.flush()

            if cb < ub:
                for tot in cards.values():
                    solver.add_clause([-tot.rhs[cb]])

            if solver.solve():
                best_model = solver.get_model()
                if verbose:
                    print("Found solution")
                    sys.stdout.flush()
            else:
                break
    if timer is not None:
        timer.cancel()

    return _decode(g, best_model, pool)
