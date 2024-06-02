from collections import defaultdict

from pysat.solvers import Cadical153, Glucose3
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType

# spacing = 0
spacing = 23
k = 20
use_solutions = True
use_options = False

solutions = [
0,
0,
1,
3,
6,
11,
17,
25,
34,
44,
55,
72,
85,
106,
127,
151,
177,
199,
216,
246,
283,
333,
356,
372,
425,
480,
492,
553,
585,
]

n = solutions[k]

pool = IDPool()
formula = CNF()

positions = [[pool.id(f"p_{i}_{j}") for j in range(0, spacing+1)] for i in range(0, k)]
last_pos = [[pool.id(f"r_{i}_{j}") for j in range(0, n + 1)] for i in range(0, k)]
ruler = [pool.id(f"x_{i}") for i in range(0, n+1)]

for i in range(0, k):
    formula.append([positions[i][j] for j in range(0, len(positions[i]))])
    formula.extend(CardEnc.atmost([positions[i][j] for j in range(0, len(positions[i]))], 1, vpool=pool))

for j in range(0, spacing+1):
    formula.extend(CardEnc.atmost([positions[i][j] for i in range(0, k)], 1, vpool=pool))

# Place last tick at the end
for j in range(n-spacing, n):
    formula.append([-last_pos[k-2][j], positions[k-1][n-j-1]])

# Break symmetry with first space
for i in range(1, spacing):
    cl = [*[positions[0][x] for x in range(0, i)], -positions[-2][-i-1]]
    formula.append(cl)

# Check the absolute position of the ticks
for i in range(0, k):
    for j in range(0, spacing+1):
        if i == 0:
            formula.append([-positions[i][j], last_pos[i][j+1]])
            formula.append([positions[i][j], -last_pos[i][j+1]])
            formula.append([-positions[i][j], ruler[j+1]])
        else:
            for x in range(0, n+1):
                if x + j >= n:
                    formula.append([-positions[i][j], -last_pos[i - 1][x]])
                else:
                    formula.append([-positions[i][j], -last_pos[i - 1][x], last_pos[i][x + j + 1]])
                    formula.append([-positions[i][j], last_pos[i - 1][x], -last_pos[i][x + j + 1]])
                    formula.append([-last_pos[i][x+j+1], ruler[x+j + 1]])

formula.append([ruler[0]])
formula.append([ruler[-1]])

for i in range(0, k):
    for j in range(1, spacing + 1):
        for x in range(0, n+1):
            if j + x < n:
                if i > 0:
                    formula.append([-positions[i][j], -ruler[x], -ruler[x + j + 1], last_pos[i-1][x]])
                elif x > 0:
                    formula.append([-positions[i][j], -ruler[x], -ruler[x + j + 1]])



if use_solutions:
    for i in range(1, k):
        formula.append([last_pos[i][x] for x in range(0, n-solutions[k-i]+1)])
        # formula.append([pool.id(f"c_{n-solutions[k-i]+1}_{i+1}")])
#         # formula.extend(CardEnc().atleast(ruler[0:n-solutions[k-i]+1], i+1, vpool=pool))


formula.to_file("outpout.enc")
print(f"{formula.nv} Vars, {len(formula.clauses)} Clauses")

with Cadical153() as slv:
    slv.append_formula(formula)

    if slv.solve():
        print("SAT")
        model = [x > 0 for x in slv.get_model()]
        model.insert(0, None)

        final_ruler = []
        for i, x in enumerate(ruler):
            if model[x]:
                print(i)
                final_ruler.append(i)

        distances = []
        for i in range(0, len(final_ruler)):
            for j in range(i+1, len(final_ruler)):
                distances.append(final_ruler[j]-final_ruler[i])
        distances.sort()
        for i, cd in enumerate(distances):
            assert(cd != distances[i+1])
    else:
        print("UNSAT")

