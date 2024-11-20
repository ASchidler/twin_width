import cnfgen

variables = 15
clauses = [15, 30, 45, 65, 80, 95, 110]
ks = [2, 3, 5, 7, 10]

for c_cl in clauses:
    for c_k in ks:
        for i in range(1, 11):
            bla = cnfgen.RandomKCNF(c_k, variables, c_cl)
            with open(f"sat_experiments/randoms/ur_{c_k}_{variables}_{c_cl}_{i}.cnf", "w") as outp:
                outp.write(bla.dimacs())
