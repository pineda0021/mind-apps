import streamlit as st
import sympy as sp
import re
import numpy as np
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

transformations = standard_transformations + (implicit_multiplication_application,)
x = sp.symbols('x')

def parse_latex_integral(latex_input):
    # Normalize input
    latex_input = latex_input.replace(" ", "").replace("dx", "")
    pattern_definite = r"\\int_({[^}]+})\^({[^}]+})(.+)"
    pattern_indefinite = r"\\int(.+)"

    match_def = re.match(pattern_definite, latex_input)
    if match_def:
        a = float(match_def.group(1).strip("{}"))
        b = float(match_def.group(2).strip("{}"))
        expr = match_def.group(3)
        fx = parse_expr(expr, transformations=transformations)
        return "definite", fx, x, a, b

    match_indef = re.match(pattern_indefinite, latex_input)
    if match_indef:
        expr = match_indef.group(1)
        fx = parse_expr(expr, transformations=transformations)
        return "indefinite", fx, x, None, None

    raise ValueError("Unrecognized LaTeX integral format. Try '\\int x^2 dx' or '\\int_0^1 x^2 dx'")

def run():
    st.set_page_config("Antiderivative LaTeX Visualizer", layout="wide")
    st.header("∫ LaTeX Integral Visualizer")
    st.markdown("Enter a LaTeX-style integral like `\\int x^2 dx` or `\\int_0^1 sqrt(x+1) dx`.")

    user_input = st.text_input("Enter integral (LaTeX style):", "\\int_0^1 x^2 dx")

    try:
        mode, fx, var, a, b = parse_latex_integral(user_input)
    except Exception as e:
        st.error(f"❌ {e}")
        return

    if mode == "indefinite":
        F = sp.integrate(fx, var)
        st.subheader("🧮 Indefinite Integral")
        st.latex(rf"\int {sp.latex(fx)} \, d{sp.latex(var)} = {sp.latex(F)} + C")
    else:
        F = sp.integrate(fx, (var, a, b))
        st.subheader("🧮 Definite Integral")
        st.latex(rf"\int_{{{a}}}^{{{b}}} {sp.latex(fx)} \, d{sp.latex(var)} = {sp.latex(F)}")

        # Step-by-step breakdown
        antiderivative = sp.integrate(fx, var)
        Fa = antiderivative.subs(var, a)
        Fb = antiderivative.subs(var, b)
        st.markdown("**Step-by-Step:**")
        st.latex(rf"F(x) = {sp.latex(antiderivative)}")
        st.latex(rf"\int_{{{a}}}^{{{b}}} {sp.latex(fx)} \, dx = F({b}) - F({a}) = {sp.latex(Fb)} - {sp.latex(Fa)} = {sp.latex(F)}")

    # Graph f(x) and filled area if definite
    st.subheader("📈 Graph of f(x)")
    f_np = sp.lambdify(var, fx, modules=["numpy"])
    X = np.linspace(a - 1 if a else -5, b + 1 if b else 5, 400)
    Y = f_np(X)
    fig, ax = plt.subplots()
    ax.plot(X, Y, label="f(x)", color="blue")
    if mode == "definite":
        x_fill = np.linspace(a, b, 300)
        y_fill = f_np(x_fill)
        ax.fill_between(x_fill, y_fill, alpha=0.3, color="green", label="Area")
    ax.axhline(0, color='black', lw=0.5)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    run()
