solver_mapping = {
"c": "B\\&B C",
"cdo": "B\\&B OC B",
"cdog": "B\\&B OC B G",
"co": "B\\&B OC",
"d": "B\\&B B",
"g": "B\\&B G",
"plain": "B\\&B",
"sat0": "SAT Rel",
"sat0ft": "SAT Rel FS",
"sat0t": "SAT Rel S",
"sat1": "SAT Abs$^4$",
"sat1c": "SAT Abs$^3$",
"sat1cft": "SAT Abs$^3$ FS",
"sat1co": "SAT Abs$^3$ O",
"sat1coft": "SAT Abs$^3$ O FS",
"sat1cot": "SAT Abs$^3$ O S",
"sat1ct": "SAT Abs$^3$ S",
"sat1cx": "SAT Abs$^3$ O",
"sat3": "SAT Abs$^3$ D",
"sat3ft": "SAT Abs$^3$ D FS",
"sat3o": "SAT Abs$^3$ D O",
"sat3oft": "SAT Abs$^3$ D O FS",
"sat3ot": "SAT Abs$^3$ D O S",
"sat3t": "SAT Abs$^3$ D S",
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

def map_solver(slv):
    if slv in solver_mapping:
        return solver_mapping[slv]
    return None

def order_solver(slv):
    if slv not in solver_order:
        return None
    return solver_order.index(slv)