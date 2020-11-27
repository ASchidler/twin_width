from pysat.formula import CNF
from pysat.card import CardEnc


def amo_commander(vars, vpool, m=2):
    formula = CNF()
    # Separate into list
    cnt = 0
    groups = []
    while cnt < len(vars):
        cg = []
        for i in range(0, min(m, len(vars) - cnt)):
            cg.append(vars[cnt + i])
        groups.append(cg)
        cnt += m

    cmds = []
    # Encode commanders
    for cg in groups:
        if len(cg) > 1:
            ncmd = vpool.id(f"amo_{vpool.top}")
            cmds.append(ncmd)
            cg.append(-ncmd)
            formula.extend(CardEnc.atmost(cg, bound=1, vpool=vpool))
            formula.extend(CardEnc.atleast(cg, bound=1, vpool=vpool))
        else:
            cmds.append(cg[0])

    # Recursive call?
    if len(cmds) < 2 * m:
        formula.extend(CardEnc.atmost(cmds, bound=1, vpool=vpool))
    else:
        formula.extend(amo_commander(cmds, vpool, m=m))

    return formula
