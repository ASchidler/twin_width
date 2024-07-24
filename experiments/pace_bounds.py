import sys
import tarfile as tf
import os
from collections import defaultdict

input = "logs/tww-pace.tar.bz2"
results_path = "tww-pace.csv"

# input = "logs/tww-twlib.tar.bz2"
# results_path = "tww-twlib.csv"

import numpy as np
import scipy.stats as st

twin_widths = {}
results = {}
lbs = set()

with open(results_path, "r") as inp:
    target_cols = []
    target_names = []
    for ci, cl in enumerate(inp):
        cfields = [cx.strip() for cx in cl.split(",")]
        if ci == 0:
            for cfi, cf in enumerate(cfields):
                if cf.endswith("tww"):
                    target_cols.append(cfi)
                    target_names.append(cf)
        else:
            if len(cfields) > 0:
                for cfiid, cfi in enumerate(target_cols):
                    if len(cfields[cfi]) > 0:
                        new_tww = int(cfields[cfi])
                        if int(cfields[0]) in twin_widths:
                            if twin_widths[int(cfields[0])] != new_tww:
                                print(f"Disagreeing width {target_names[cfiid]} (Instance {cfields[0]}): {twin_widths[int(cfields[0])]} {new_tww}")
                        else:
                            twin_widths[int(cfields[0])] = new_tww


class BoundResult:
    def __init__(self):
        self.tww = None
        self.lbs = defaultdict(int)
        self.ub = None


with tf.open(input) as ctf:
    for clf in ctf:
        name_fields = clf.name.split(".")
        if not name_fields[1].startswith("s") or any(name_fields[0].split("-")[-1].startswith(x) for x in ["sat", "winner"]):
            continue

        instance = int(name_fields[-1])
        cetf = ctf.extractfile(clf)

        if instance not in results:
            results[instance] = BoundResult()
            if instance in twin_widths:
                results[instance].tww = twin_widths[instance]

        only_line = True
        only_field = None
        finished_seen = False
        time_seen = False
        missing_ub = False

        for i, cln in enumerate(cetf):
            cln = cln.decode('ascii').strip()
            if only_field is None and only_line:
                try:
                    only_field = int(cln.strip())
                except ValueError:
                    only_line = False
            else:
                only_line = False

            if cln.strip().startswith("LB"):
                lb_name = cln.split(":")[0].strip()
                lbs.add(lb_name)
                new_lb = int(cln.split(":")[-1].strip())
                if results[instance].lbs[lb_name] < new_lb:
                    results[instance].lbs[lb_name] = new_lb
            elif cln.strip().startswith("UB"):
                new_ub = int(cln.split(":")[-1].strip())
                missing_ub = False
                if results[instance].ub is None or results[instance].ub < new_ub:
                    results[instance].ub = new_ub
            elif cln.strip().startswith("Running component with"):
                missing_ub = True
        if missing_ub:
            results[instance].ub = None


class AggregateResult:
    def __init__(self):
        self.matched = 0
        self.matched_unique = 0
        self.best = 0
        self.unique = 0

# Ignore bad LBs
lbs.discard("LB DG NB")
lbs.discard("LB RD NB")
lbs.discard("LB Sample DG")
lbs.discard("LB Sample RD")
lbs.discard("LB Sample RD NB")

lb_ub = []
ub_result = AggregateResult()
lb_results = {cn: AggregateResult() for cn in lbs}
lb_matched = 0

for cv in results.values():
    best_lb = None
    if len(cv.lbs) > 0:
        best_lb = max(cv.lbs.values())
        if cv.tww is not None and best_lb == cv.tww:
            lb_matched += 1

        is_unique = 1 == len([x for x in cv.lbs.values() if x == best_lb])
        for clb, clbv in cv.lbs.items():
            if clb not in lbs:
                continue
            if clbv == best_lb:
                lb_results[clb].best += 1
                if is_unique:
                    lb_results[clb].unique += 1
            if cv.tww is not None and cv.tww == clbv:
                lb_results[clb].matched += 1
                if is_unique:
                    lb_results[clb].matched_unique += 1
    if cv.tww is not None and cv.ub == cv.tww:
        ub_result.matched += 1

    if cv.ub is not None and best_lb is not None:
        lb_ub.append((best_lb, cv.ub))

for ck in sorted(lbs):
    print(f"{ck}\t{lb_results[ck].best} ({lb_results[ck].unique})\t{lb_results[ck].matched} ({lb_results[ck].matched_unique})")

gfg_data = [y-x for (x, y) in lb_ub]
conf_int = st.t.interval(alpha=0.95, df=len(gfg_data)-1,
              loc=np.mean(gfg_data),
              scale=st.sem(gfg_data))

avg = (conf_int[0] + conf_int[1]) / 2
print(f"Gap ({len(gfg_data)} results): {round(avg, 2)} +- {round(conf_int[1] - avg, 2)}")
print(f"Solved: {len([x for x in gfg_data if x == 0])}")
print(f"UB Matched: {ub_result.matched}")
print(f"LB Matched: {lb_matched}")
