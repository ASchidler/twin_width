import os
import subprocess
import sys
import resource
from datetime import time
import time


def limit_memory(limit):
    if limit > 0:
        resource.setrlimit(resource.RLIMIT_AS, (limit * 1024 * 1024, (limit + 30) * 1024 * 1024))


class BaseSolver:
    def parse(self, file):
        print("Not implemented")
        raise

    def run(self, input_file, model_file, mem_limit=0):
        print("Not implemented")
        raise


class MiniSatSolver(BaseSolver):
    def run(self, input_file, model_file, mem_limit=0):
        FNULL = open(os.devnull, 'w')
        return subprocess.Popen(['minisat', '-verb=0', f'-mem-lim={mem_limit}', input_file, model_file], stdout=FNULL,
                                      stderr=subprocess.STDOUT, preexec_fn=lambda: limit_memory(mem_limit))

    def parse(self, f):
        first = f.readline()
        if first.startswith("UNSAT"):
            return None

        # TODO: This could be faster using a list...
        model = {}
        vars = f.readline().split()
        for v in vars:
            val = int(v)
            model[abs(val)] = val > 0

        return model


class GlucoseSolver(BaseSolver):
    def run(self, input_file, model_file, mem_limit=0):
        FNULL = open(os.devnull, 'w')
        # , f'-mem-lim={mem_limit}'
        return subprocess.Popen(['/home/aschidler/Downloads/glucose-syrup-4.1/simp/glucose', '-model', '-verb=0', input_file, model_file], stdout=FNULL,
                                      stderr=subprocess.STDOUT)

    def parse(self, f):
        first = f.readline()
        if first.startswith("UNSAT"):
            return None

        # TODO: This could be faster using a list...
        model = {}
        vars = first.split()
        for v in vars:
            val = int(v)
            model[abs(val)] = val > 0

        return model


class CadicalSolver(BaseSolver):
    def run(self, input_file, model_file, mem_limit=0):
        out_file = open(model_file, "w")
        return subprocess.Popen(['cadical', '-q', input_file], stdout=out_file,
                                #      stderr=subprocess.STDOUT, preexec_fn=lambda: limit_memory(mem_limit)
                                )

    def parse(self, f):
        first = f.readline()
        if first.startswith("s UNSAT"):
            return None

        # TODO: This could be faster using a list...
        model = {}
        for _, ln in enumerate(f):
            if ln.startswith("v "):
                vars = ln.split()
                for v in vars[1:]:
                    val = int(v)
                    model[abs(val)] = val > 0

        return model


class WrMaxsatSolver(BaseSolver):
    def supports_timeout(self):
        return True

    def run(self, input_file, model_file, timeout=0):
        with open(model_file, "w") as mf:
            if timeout == 0:
                return subprocess.Popen(['uwrmaxsat', input_file, '-m'], stdout=mf)
            else:
                return subprocess.Popen(['uwrmaxsat', input_file, '-m', f'-cpu-lim={timeout}'],
                                        stdout=mf)

    def parse(self, f):
        model = {}
        for _, cl in enumerate(f):
            # Model data
            if cl.startswith("v "):
                values = cl.split(" ")
                for v in values[1:]:
                    converted = int(v)
                    model[abs(converted)] = converted > 0

        if len(model) == 0:
            return None
        return model


def add_cardinality_constraint(target_arr, limit, encoder):
    """Limits the cardinality of variables in target_arr <= limit. Expects a BaseEncoder"""

    # Create counter variables
    n = len(target_arr)

    # Add a set of vars for all elements. Set counts how many element have been seen up to this element
    ctr = [[] for _ in range(0, n)]
    for i, c in enumerate(ctr):
        for j in range(0, min(i+1, limit)):
            c.append(encoder.add_var())

    for i in range(1, n-1):
        # Carry over previous element
        for ln in range(0, len(ctr[i-1])):
            encoder.add_clause(-ctr[i-1][ln], ctr[i][ln])

        # Increment counter, if current element is true
        for ln in range(1, len(ctr[i])):
            encoder.add_clause(-ctr[i-1][ln-1], -target_arr[i], ctr[i][ln])

    # Initialize counter on first element
    for i in range(0, n-1):
        encoder.add_clause(-target_arr[i], ctr[i][0])

    # Unsat if counter is exceeded
    for i in range(limit, n):
        encoder.add_clause(-target_arr[i], -ctr[i][limit-1])


class SatRunner:
    def __init__(self, encoder, solver, base_path=".", tmp_file=None):
        self.base_path = base_path
        self.tmp_file = tmp_file if tmp_file is not None else os.getpid()
        self.solver = solver
        self.encoder = encoder

    def run(self, starting_bound, g, timeout=0, memlimit=0, u_bound=sys.maxsize):
        l_bound = 0
        c_bound = starting_bound
        enc_size = 0

        enc_file = os.path.join(self.base_path, f"{self.tmp_file}.enc")
        model_file = os.path.join(self.base_path, f"{self.tmp_file}.model")

        start = time.time()
        tree = None
        while l_bound < u_bound:
            # print(f"Running with limit {c_bound}")
            with open(enc_file, "w") as f:
                inst_encoding = self.encoder(f)
                inst_encoding.encode(g, c_bound)
            enc_size = max(enc_size, os.path.getsize(enc_file))

            p1 = self.solver.run(enc_file, model_file, memlimit)

            if timeout == 0:
                p1.wait()
            else:
                try:
                    elapsed = time.time() - start
                    p1.wait(timeout=timeout-elapsed)
                except subprocess.TimeoutExpired:
                    if p1.poll() is None:
                        p1.terminate()
                    return None, enc_size

            if not os.path.exists(model_file):
                os.remove(enc_file)
                return None, enc_size
            print(f"Elapsed {time.time() - start}")
            with open(model_file, "r") as f:
                model = self.solver.parse(f)

                if model is not None and len(model) == 0:
                    return None, enc_size

                if model is None:
                    print(f"None {c_bound}")
                    l_bound = c_bound + inst_encoding.increment
                    c_bound = l_bound
                else:
                    tree = inst_encoding.decode(model, g, c_bound)

                    u_bound = c_bound
                    c_bound -= inst_encoding.increment

            os.remove(model_file)
            os.remove(enc_file)

        return tree, enc_size


class MaxSatRunner:
    def __init__(self, encoder, solver, base_path=".", tmp_file=None):
        self.base_path = base_path
        self.tmp_file = tmp_file if tmp_file is not None else os.getpid()
        self.solver = solver
        self.encoder = encoder

    def run(self, instance, timeout=0):
        enc_file = os.path.join(self.base_path, f"{self.tmp_file}.enc")
        model_file = os.path.join(self.base_path, f"{self.tmp_file}.model")

        with open(enc_file, "w") as f:
            inst_encoding = self.encoder(f)
            inst_encoding.encode(instance)

        p1 = self.solver.run(enc_file, model_file, timeout)

        if timeout == 0 or self.solver.supports_timeout():
            p1.wait()
        else:
            try:
                p1.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if p1.poll() is None:
                    p1.terminate()

        result = None
        with open(model_file, "r") as f:
            model = self.solver.parse(f)
            if model is not None:
                result = inst_encoding.decode(model, instance)

        os.remove(enc_file)
        os.remove(model_file)

        return result
