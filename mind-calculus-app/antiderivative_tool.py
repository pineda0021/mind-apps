import streamlit as st
import sympy as sp
from sympy.abc import x
import matplotlib.pyplot as plt
import numpy as np

sympy_locals = {
    "e": sp.E,
    "pi": sp.pi,
    "oo": sp.oo,
    "-oo": -sp.oo,
    "sqrt": sp.sqrt,
    "ln": sp.log,
    "exp": sp.exp,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "sec": sp.sec,
    "csc": sp.csc,
    "cot": sp.cot,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "log": sp.log,
}

def step_by_step_antiderivative(expr):
    steps = []

    if expr.is_Add:
        steps.append("**Sum Rule:**")
        for term in expr.args:
            steps += step_by_step_antiderivative(term)
        return steps

    if expr.is_Number:
        steps.append("**Constant Rule:**")
        steps.append(rf"$\\int {sp.latex(expr)} \\, dx = {sp.latex(expr)}x$")
        return steps

    if expr.is_Pow and expr.args[0] == x:
        n = expr.args[1]
        if n != -1:
            result = sp.integrate(expr, x)
            steps.append("**Power Rule:**")
            steps.append(rf"$\\int x^{{{sp.latex(n)}}} \\, dx = \\frac{{x^{{{sp.latex(n+1)}}}}}{{{sp.latex(n+1)}}}$")
            steps.append(rf"$= {sp.latex(result)}$")
        else:
            steps.append("**Special Case:**")
            steps.append(rf"$\\int \\frac{{1}}{{x}} \\, dx = \\ln|x|$")
        return steps

    if expr == sp.exp(x):
        steps.append("**Exponential Rule:**")
        steps.append(rf"$\\int e^x \\, dx = e^x$")
        return steps

    if expr == sp.log(x):
        steps.append("**Logarithmic Rule:**")
        steps.append(rf"$\\int \\ln x \\, dx = x\\ln x - x$")
        return steps

    if expr == sp.sin(x):
        steps.append("**Trig Rule:**")
        steps.append(rf"$\\int \\sin x \\, dx = -\\cos x$")
        return steps

    if expr == sp.cos(x):
        steps.append("**Trig Rule:**")
        steps.append(rf"$\\int \\cos x \\, dx = \\sin x$")
        return steps

    if expr == sp.tan(x):
        steps.append("**Trig Rule:**")
        steps.append(rf"$\\int \\tan x \\, dx = -\\ln|\\cos x|$")
        return steps

    if expr == sp.asin(x):
        steps.append("**Inverse Trig Rule:**")
        steps.append(rf"$\\int \\sin^{{-1}} x \\, dx = x \\sin^{{-1}} x + \\sqrt{{1 - x^2}}$")
        return steps

    if expr == sp.acos(x):
        steps.append("**Inverse Trig Rule:**")
        steps.append(rf"$\\int \\cos^{{-1}} x \\, dx = x \\cos^{{-1}} x - \\sqrt{{1 - x^2}}$")
        return steps

    if expr == sp.atan(x):
        steps.append("**Inverse Trig Rule:**")
        steps.append(rf"$\\int \\tan^{{-1}} x \\, dx = x \\tan^{{-1}} x - \\frac{{1}}{{2}} \\ln(1 + x^2)$")
        return steps

    if expr == sp.sinh(x):
        steps.append("**Hyperbolic Rule:**")
        steps.append(rf"$\\int \\sinh x \\, dx = \\cosh x$")
        return steps

    if expr == sp.cosh(x):
        steps.append("**Hyperbolic Rule:**")
        steps.append(rf"$\\int \\cosh x \\, dx = \\sinh x$")
        return steps

    if expr == sp.tanh(x):
        steps.append("**Hyperbolic Rule:**")
        steps.append(rf"$\\int \\tanh x \\, dx = \\ln(\\cosh x)$")
        return steps

    if expr.is_Mul:
        factors = expr.args
        for i in range(len(factors)):
            for j in range(len(factors)):
                if i != j and factors[i] == sp.diff(factors[j], x):
                    u = factors[j]
                    du = factors[i]
                    result = sp.integrate(u, x)
                    steps.append("**Chain Rule (u-substitution):**")
                    steps.append(rf"Let $u = {sp.latex(u)}$, then $du = {sp.latex(sp.diff(u, x))} dx$")
                    steps.append(rf"Rewrite: $\\int {sp.latex(expr)} \\, dx = \\int u \\, du$")
                    steps.append(rf"$= {sp.latex(result)}$")
                    return steps

    if expr.is_Mul and any(arg.has(x) for arg in expr.args):
        u, dv = expr.args
        du = sp.diff(u, x)
        v = sp.integrate(dv, x)
        uv = u * v
        int_vdu = sp.integrate(v * du, x)
        result = uv - int_vdu
        steps.append("**Integration by Parts:**")
        steps.append(rf"$\\int {sp.latex(expr)} \\, dx = uv - \\int v \\, du$")
        steps.append(rf"Let $u = {sp.latex(u)}, dv = {sp.latex(dv)}dx$")
        steps.append(rf"Then $du = {sp.latex(du)}dx$, and $v = {sp.latex(v)}$")
        steps.append(rf"$= {sp.latex(uv)} - \\int {sp.latex(v * du)} \\, dx$")
        steps.append(rf"$= {sp.latex(result)}$")
        return steps

    result = sp.integrate(expr, x)
    steps.append("**General Rule (Auto Integration):**")
    steps.append(rf"$\\int {sp.latex(expr)} \\, dx = {sp.latex(result)}$")
    return steps
