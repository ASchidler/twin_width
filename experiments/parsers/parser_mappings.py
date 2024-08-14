equivalences = {
    "cod": "cdo",
    "codg": "cdog"
}

solver_mapping = {
"c": "BB-CCH C",
"cdo": "BB-CCH + Bounds",
"cod": "BB-CCH + Bounds",
"cdog": "BB-CCH OC B G",
"codg": "BB-CCH OC B G",
"co": "BB-CCH",
"d": "BB-CCH B",
"g": "BB-CCH G",
"plain": "BB-CCH",
"sat0": "SAT Relative",
"sat0ft": "Relative Full Static",
"sat0t": "Relative Light Static",
"sat1": "Abs Quartic",
"sat1c": "SAT Absolute Cubic",
"sat1cft": "SAT Abs$^3$ FS",
"sat1co": "Abs Cubic Order",
"sat1coft": "Abs Cubic Static",
"sat1cot": "Abs Cubic Order Static",
"sat1ct": "Abs Cubic Static",
"sat1cx": "Abs Cubic Order",
"sat3": "Abs Cubic Card",
"sat3ft": "Abs Cubic Card Static",
"sat3o": "Abs Cubic Card Order",
"sat3oft": "Abs Cubic Card Order Static",
"sat3ot": "Abs Cubic Card Order Static",
"sat3t": "Abs Cubic Card Static",
"winner": "Hydra Prime",
"sat": "SAT All"
}

solver_order = [
"winner",
"plain",
"c",
"co",
"g",
"d",
"cdo",
"cod",
"cdog",
"sat0",
"sat0t",
"sat0ft",
"sat1",
"sat1c",
"sat1co",
"sat1ct",
"sat1cot",
"sat3",
"sat3o",
"sat3t",
"sat3ot",
"sat"
]


def resolve_solver(slv):
    if slv.find("-") > -1:
        slv = slv.split("-")[-1]

    if slv in equivalences:
        return equivalences[slv]
    return slv


def map_solver(slv):
    if slv.find("-") > -1:
        slv = slv.split("-")[-1]

    if slv in solver_mapping:
        return solver_mapping[slv]
    return None


def order_solver(slv):
    if slv.find("-") > -1:
        slv = slv.split("-")[-1]
    if slv not in solver_order:
        return None
    return solver_order.index(slv)
