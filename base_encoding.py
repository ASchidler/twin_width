from os import linesep


class BaseEncoding:
    def __init__(self, stream):
        self.size_limit = 50 * 1000 * 1000 * 1000
        self.vars = 0
        self.clauses = 0
        self.stream = stream
        self.increment = 1
        # Placeholder for header
        self.stream.write(" ".join(["" for _ in range(0, 100)]))

    def write_header(self):
        self.stream.seek(0)
        header = f"p cnf {self.vars} {self.clauses}"
        padding = 100 - len(header) - len(linesep)
        self.stream.write(header)
        self.stream.write(" ".join(["" for _ in range(0, padding)]))
        self.stream.write(linesep)

    def add_var(self):
        self.vars += 1
        return self.vars

    def add_clause(self, *args):
        if len(args) > 0:
            if self.stream.tell() > self.size_limit:
                raise MemoryError("Encoding size too large")

            self.stream.write(' '.join([str(x) for x in args]))
            self.stream.write(" 0\n")
            self.clauses += 1


    '''Encode the conjunction of args as auxiliary variable. 
            The auxiliary variable can then be used instead of the conjunction'''
    def add_auxiliary(self, *args):
        v = self.add_var()
        clause = [v]

        # auxiliary variables
        for a in args:
            self.add_clause(-v, a)
            clause.append(-a)

        self.add_clause(*clause)

        return v

    def encode_cardinality_sat(self, bound, variables):
        if bound == 0:
            for ce1 in variables:
                for ce2 in ce1:
                    self.add_clause(-ce2)
            return

        """Enforces cardinality constraints. Cardinality of 2-D structure variables must not exceed bound"""
        # Counter works like this: ctr[i][j][0] states that an arc from i to j exists
        # These are then summed up incrementally edge by edge

        # TODO: add special case for 0
        # Define counter variables ctr[i][j][l] with 1 <= i <= n, 1 <= j < n, 1 <= l <= min(j, bound)
        ctr = [[[self.add_var()
                 for _ in range(0, min(j, bound))]
                # j has range 0 to n-1. use 1 to n, otherwise the innermost number of elements is wrong
                for j in range(1, len(variables[0]))]
               for _ in range(0, len(variables))]

        for i in range(0, len(variables)):
            for j in range(1, len(variables[i]) - 1):
                # Ensure that the counter never decrements, i.e. ensure carry over
                for ln in range(0, min(len(ctr[i][j-1]), bound)):
                    self.add_clause(-ctr[i][j - 1][ln], ctr[i][j][ln])

                # Increment counter for each arc
                for ln in range(1, min(len(ctr[i][j]), bound)):
                    self.add_clause(-variables[i][j], -ctr[i][j-1][ln-1], ctr[i][j][ln])

        # Ensure that counter is initialized on the first arc
        for i in range(0, len(variables)):
            for j in range(0, len(variables[i]) - 1):
                self.add_clause(-variables[i][j], ctr[i][j][0])

        # Conflict if target is exceeded
        for i in range(0, len(variables)):
            for j in range(bound, len(variables[i])):
                # Since we start to count from 0, bound - 2
                self.add_clause(-variables[i][j], -ctr[i][j-1][bound - 1])

