from operator import pos
from latex2sympy2 import latex2sympy, latex2latex
from sympy import *
from sympy import factor
import sympy as sy
from sympy.abc import x, a, b
import re

tex = "\int_{a}^{b}x^2"

def polynom_check(latex, char):
    pos = []
    for n in range(len(latex)):
        if latex[n] == char:
            pos.append(n)

    for i in range(len(pos)):
        if latex[pos[i]-1] == 'x':
            return True
    
    return False

def searching_for_error(latex):
    
    a = re.search(r'\b(=0)\b', latex)
    new_string = latex[:a.start()] + latex[a.start()+(a.end()-a.start()):]

    return new_string

def polynom_solutions(equation):
    solutions = []
    
    # Parabelfunktion
    parabel = sy.S(equation)
    print(parabel)

    # Faktorisieren
    factors = simplify(equation)
    solutions.append("Faktorisiert: " + str(factors))

    # Erste Ableitung
    parabel_diff_1 = sy.diff(parabel, x, 1)
    solutions.append("1. Ableitung: " + str(parabel_diff_1))

    # Zweite Ableitung
    parabel_diff_2 = sy.diff(parabel, x, 2)
    solutions.append("2. Ableitung: " + str(parabel_diff_2))

    # Nullstellen
    print(equation)
    factorize = factor(equation)
    print(factorize)
    zero = sy.solve(factorize)
    print(latex(zero))
    solutions.append("Nullstellen: " + str(zero))

    # Extremstellen
    extremes = sy.solve(sy.Eq(parabel_diff_1, 0))
    x_1 = extremes[0]
    y_1 = parabel.subs(x, x_1)
    solutions.append("Extremstelle bei: " + str(x_1) + " " + str(y_1))

    return solutions

def calculate(latex):

    sym = latex2sympy(latex)

    if sym == 0:
        latex = searching_for_error(latex)

    #if polynom_check(latex, '^') == True:
    #    print("es ist eine Polynom-funktion")
    #    polynom = latex2sympy(latex)
    #    solutions = polynom_solutions(polynom)

    integral = integrate(x**2, x)
    print(integral)
    #print(solutions)
    #print(sym)
    #simply = simplify(sym)
    #print(simplify(sym))
    #nullstellen = solve(simply, x)
    #print("x1: " + str(nullstellen[0]))
    #print("x2: " + str(nullstellen[1]))

calculate(tex)