from operator import pos
from latex2sympy2 import latex2sympy, latex2latex
from sympy import *
from sympy.abc import x
import re
from sympy.integrals.manualintegrate import integral_steps

tex = "f:x=x^{2}+5x+10x+5=0"

def charposition(string, char):
    pos = []
    for n in range(len(string)):
        if string[n] == char:
            pos.append(n)

    return pos

#found = charposition(tex, '^')

#for i in range(len(found)):
#    print(tex[found[i]-1])

def searching_for_error(latex):
    
    a = re.search(r'\b(=0)\b', latex)
    new_string = latex[:a.start()] + latex[a.start()+(a.end()-a.start()):]

    return new_string

def calculate(latex):

    sym = latex2sympy(latex)

    if sym == 0:
        sym = latex2sympy(searching_for_error(latex))

    print(sym)
    simply = simplify(sym)
    print(simplify(sym))
    nullstellen = solve(simply, x)
    print("x1: " + str(nullstellen[0]))
    print("x2: " + str(nullstellen[1]))

calculate(tex)