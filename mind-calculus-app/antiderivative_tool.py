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
    "I": sp.I,
    "sqrt": sp.sqrt,
    "ln": sp.log,
    "exp": sp.exp,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "sec": sp.sec,
    "csc": sp.csc,
    "cot": sp.cot,
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
    if expr == sp.exp(x):
        steps.append("**Exponential Rule:**")
        steps.append(rf"$\\int e^x \\, dx = e^x$")
        return steps
    if expr == 1/x:
        steps.append("**Log Rule:**")
        steps.append(rf"$\\int \\frac{{1}}{{x}} \\, dx = \\ln|x|$")
        return steps
    if expr == sp.sin(x):
        steps.append("**Trig Rule:**")
        steps.append(rf"$\\int \\sin x \\, dx = -\\cos x$")
        return steps
    if expr == sp.cos(x):
        steps.append("**Trig Rule:**")
        steps.append(rf"$\\int \\cos x \\, dx = \\sin x$")
        return steps
    result = sp.integrate(expr, x)
    steps.append("**General Rule (Auto Integration):**")
    steps.append(rf"$\\int {sp.latex(expr)} \\, dx = {sp.latex(result)}$")
    return steps

def run():
    st.header("∫ Antiderivative Visualizer")
    st.markdown("Enter a function and explore its antiderivative (indefinite integral) symbolically and graphically.")

    f_input = st.text_input("f(x) =", "x**2 + 1")
    try:
        fx = sp.sympify(f_input, locals=sympy_locals)
        F = sp.integrate(fx, x)
    except:
        st.error("Invalid function. Please enter a valid mathematical expression.")
        return

    st.subheader("📊 Indefinite Integral")
    st.latex(rf"F(x) = \\int {sp.latex(fx)} \\, dx = {sp.latex(F)} + C")

    st.subheader("🔍 Step-by-Step Integration")
    for step in step_by_step_antiderivative(fx):
        st.markdown("- " + step)

    st.subheader("📈 Graph of f(x) and F(x)")
    try:
        f_np = sp.lambdify(x, fx, modules=["numpy"])
        F_np = sp.lambdify(x, F, modules=["numpy"])
        X = np.linspace(-5, 5, 400)
        Y = f_np(X)
        Y_int = F_np(X)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(X, Y, label="f(x)", color="blue")
        ax.plot(X, Y_int, label="F(x)", color="orange")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Function and Antiderivative")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
    except:
        st.error("Unable to generate graph. Please check the input function.")
