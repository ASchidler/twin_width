from pysat.formula import CNF
from pysat.card import CardEnc
import os


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


def dot_export(g, u, v):
    def cln(name):
        return f"{name}".replace("(", "").replace(")", "").replace(",", "").replace(" ", "_")

    output1 = "strict graph dt {" + os.linesep

    for n in g.nodes:
        cl = 'green' if n == u or n == v else 'black'
        posstr = ""
        if isinstance(n, tuple) and len(n) == 2:
            posstr = f', pos="{n[0]},{n[1]}!"'
        output1 += f"n{cln(n)} [" \
                   f"shape=box, fontsize=11,width=0.3,height=0.2,fixedsize=true,style=filled,fontcolor=white," \
                   f"color={cl}, fillcolor={cl}{posstr}];{os.linesep}"

    for x, y in g.edges:
        cl = 'red' if g[x][y]['red'] else 'black'
        output1 += f"n{cln(x)} -- n{cln(y)} [color={cl}];{os.linesep}"

    # Draw the linegraph
    output2 = "strict graph dt {" + os.linesep
    u, v = min(u, v), max(u, v)
    for x, y in g.edges:
        x, y = min(x, y), max(x, y)
        color = 'green' if x == u and v == y else 'white'
        fillcolor = 'red' if g[x][y]['red'] else 'black'
        output2 += f"n{cln(x)}_{cln(y)} [" \
        f"shape=box, fontsize=11,style=filled,fontcolor={color}," \
        f"color={color}, fillcolor={fillcolor}];{os.linesep}"

    for n in g.nodes:
        for n1 in g[n]:
            x1, x2 = min(n1, n), max(n1, n)
            for n2 in g[n]:
                if n2 > n1:
                    cl = 'green' if n1 == u and n2 == v else 'black'
                    x3, x4 = min(n2, n), max(n2, n)
                    output2 += f"n{cln(x1)}_{cln(x2)} -- n{cln(x3)}_{cln(x4)} [color={cl}];{os.linesep}"




    return output1 + "}", output2 + "}"

