import streamlit as st
import sympy as sp
from sympy.abc import x
import matplotlib.pyplot as plt
import numpy as np

def step_by_step_antiderivative(expr):
    steps = []

    # Sum Rule
    if expr.is_Add:
        steps.append("**Sum Rule:**")
        for term in expr.args:
            steps += step_by_step_antiderivative(term)
        return steps

    # Constant Rule
    if expr.is_Number:
        steps.append("**Constant Rule:**")
        steps.append(rf"$\int {sp.latex(expr)} \, dx = {sp.latex(expr)}x$")
        return steps

    # Power Rule
    if expr.is_Pow and expr.args[0] == x:
        n = expr.args[1]
        if n != -1:
            result = sp.integrate(expr, x)
            steps.append("**Power Rule:**")
            steps.append(rf"$\int x^{{{sp.latex(n)}}} \, dx = \frac{{x^{{{sp.latex(n+1)}}}}}{{{sp.latex(n+1)}}}$")
            steps.append(rf"$= {sp.latex(result)}$")
        else:
            steps.append("**Special Case:**")
            steps.append(rf"$\int \frac{{1}}{{x}} \, dx = \ln|x|$")
        return steps

    # Chain Rule (u-substitution)
    if expr.is_Mul:
        factors = expr.args
        for i in range(len(factors)):
            for j in range(len(factors)):
                if i != j and factors[i] == sp.diff(factors[j], x):
                    u = factors[j]
                    du = factors[i]
                    new_expr = u**1
                    result = sp.integrate(u, x)
                    steps.append("**Chain Rule (u-substitution):**")
                    steps.append(rf"Let $u = {sp.latex(u)}$, then $du = {sp.latex(sp.diff(u, x))} dx$")
                    steps.append(rf"Rewrite: $\int {sp.latex(expr)} \, dx = \int u \, du$")
                    steps.append(rf"$= {sp.latex(sp.integrate(u, x))}$")
                    return steps

    # Integration by Parts
    if expr.is_Mul and any(arg.has(x) for arg in expr.args):
        u, dv = expr.args
        du = sp.diff(u, x)
        v = sp.integrate(dv, x)
        uv = u * v
        int_vdu = sp.integrate(v * du, x)
        result = uv - int_vdu
        steps.append("**Integration by Parts:**")
        steps.append(rf"$\int {sp.latex(expr)} \, dx = uv - \int v \, du$")
        steps.append(rf"Let $u = {sp.latex(u)}, dv = {sp.latex(dv)}dx$")
        steps.append(rf"Then $du = {sp.latex(du)}dx$, and $v = {sp.latex(v)}$")
        steps.append(rf"$= {sp.latex(uv)} - \int {sp.latex(v * du)} \, dx$")
        steps.append(rf"$= {sp.latex(result)}$")
        return steps

    # Trig/Exp/Log
    if expr == sp.exp(x):
        steps.append("**Exponential Rule:**")
        steps.append(rf"$\int e^x \, dx = e^x$")
        return steps

    if expr == 1/x:
        steps.append("**Log Rule:**")
        steps.append(rf"$\int \frac{{1}}{{x}} \, dx = \ln|x|$")
        return steps

    if expr == sp.sin(x):
        steps.append("**Trig Rule:**")
        steps.append(rf"$\int \sin x \, dx = -\cos x$")
        return steps

    if expr == sp.cos(x):
        steps.append("**Trig Rule:**")
        steps.append(rf"$\int \cos x \, dx = \sin x$")
        return steps

    # Fallback
    result = sp.integrate(expr, x)
    steps.append("**General Rule (Auto Integration):**")
    steps.append(rf"$\int {sp.latex(expr)} \, dx = {sp.latex(result)}$")
    return steps

def definite_integral_steps(fx, a, b):
    steps = []
    F = sp.integrate(fx, x)
    Fa = F.subs(x, a)
    Fb = F.subs(x, b)
    area = Fb - Fa
    steps.append("**Fundamental Theorem of Calculus:**")
    steps.append(rf"$\int_{{{a}}}^{{{b}}} {sp.latex(fx)} \, dx = F({b}) - F({a})$")
    steps.append(rf"$= {sp.latex(F)} \Big|_{{{a}}}^{{{b}}} = {sp.latex(Fb)} - {sp.latex(Fa)} = {sp.latex(area)}$")
    return steps

def run():
    st.header("∫ Antiderivative Visualizer")
    st.markdown("""
    Enter a function and explore its antiderivative (indefinite or definite integral) symbolically and graphically.
    """)

    st.subheader("📥 Enter a Function")
    f_input = st.text_input("f(x) =", "x**2 + 1")
    try:
        fx = sp.sympify(f_input)
        F = sp.integrate(fx, x)
    except:
        st.error("Invalid function. Please enter a valid mathematical expression.")
        return

    st.subheader("🧮 Symbolic Antiderivative")
    st.latex(rf"F(x) = \int {sp.latex(fx)} \, dx = {sp.latex(F)} + C")

    st.subheader("🔎 Step-by-Step Integration")
    for step in step_by_step_antiderivative(fx):
        st.markdown("- " + step)

    st.subheader("📈 Graph of f(x) and F(x)")
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
    ax.grid(True)
    ax.set_title("Function and Antiderivative")
    ax.legend()
    st.pyplot(fig)

    # Area Visualization
    st.subheader("📊 Visualizing Accumulated Area")
    col1, col2 = st.columns(2)
    a_expr = col1.text_input("Start point a =", "-2")
    b_expr = col2.text_input("End point b =", "2")

    try:
        a_val = sp.sympify(a_expr)
        b_val = sp.sympify(b_expr)
        area_val = sp.integrate(fx, (x, a_val, b_val))

        st.latex(rf"\int_{{{sp.latex(a_val)}}}^{{{sp.latex(b_val)}}} {sp.latex(fx)} \, dx = {sp.latex(area_val)}")

        st.subheader("📐 Step-by-Step for Definite Integral")
        for step in definite_integral_steps(fx, a_val, b_val):
            st.markdown("- " + step)

        # Highlight Area
        x_fill = np.linspace(float(a_val.evalf()), float(b_val.evalf()), 300)
        y_fill = f_np(x_fill)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(X, Y, label="f(x)", color="blue")
        ax2.fill_between(x_fill, y_fill, alpha=0.3, color="green", label="Accumulated Area")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("Accumulated Area from a to b")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

    except:
        st.error("Invalid a or b. Please enter expressions like -2, pi, sqrt(2), or oo")

# Run the app
run()
