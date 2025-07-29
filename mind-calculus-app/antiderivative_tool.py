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
        steps.append(r"\\textbf{Sum Rule:}")
        for term in expr.args:
            steps += step_by_step_antiderivative(term)
        return steps

    if expr.is_Number:
        steps.append(r"\\textbf{Constant Rule:}")
        steps.append(r"\\int %s \\, dx = %sx + C" % (sp.latex(expr), sp.latex(expr)))
        return steps

    if expr.is_Pow and expr.args[0] == x:
        n = expr.args[1]
        if n != -1:
            result = sp.integrate(expr, x)
            steps.append(r"\\textbf{Power Rule:}")
            steps.append(r"\\int x^{%s} \\, dx = \\frac{x^{%s}}{%s} + C" % (sp.latex(n), sp.latex(n+1), sp.latex(n+1)))
            steps.append(r"= %s + C" % sp.latex(result))
        else:
            steps.append(r"\\textbf{Special Case:}")
            steps.append(r"\\int \\frac{1}{x} \\, dx = \\ln|x| + C")
        return steps

    if expr == sp.exp(x):
        steps.append(r"\\textbf{Exponential Rule:}")
        steps.append(r"\\int e^x \\, dx = e^x + C")
        return steps

    if expr == sp.exp(-x):
        steps.append(r"\\textbf{Exponential Rule (Negative Exponent):}")
        steps.append(r"\\int e^{-x} \\, dx = -e^{-x} + C")
        return steps

    if expr == sp.log(x):
        steps.append(r"\\textbf{Logarithmic Rule:}")
        steps.append(r"\\int \\ln x \\, dx = x\\ln x - x + C")
        return steps

    if expr == sp.sin(x):
        steps.append(r"\\textbf{Trig Rule:}")
        steps.append(r"\\int \\sin x \\, dx = -\\cos x + C")
        return steps

    if expr == sp.cos(x):
        steps.append(r"\\textbf{Trig Rule:}")
        steps.append(r"\\int \\cos x \\, dx = \\sin x + C")
        return steps

    if expr == sp.tan(x):
        steps.append(r"\\textbf{Trig Rule:}")
        steps.append(r"\\int \\tan x \\, dx = -\\ln|\\cos x| + C")
        return steps

    if expr == sp.asin(x):
        steps.append(r"\\textbf{Inverse Trig Rule:}")
        steps.append(r"\\int \\sin^{-1} x \\, dx = x \\sin^{-1} x + \\sqrt{1 - x^2} + C")
        return steps

    if expr == sp.acos(x):
        steps.append(r"\\textbf{Inverse Trig Rule:}")
        steps.append(r"\\int \\cos^{-1} x \\, dx = x \\cos^{-1} x - \\sqrt{1 - x^2} + C")
        return steps

    if expr == sp.atan(x):
        steps.append(r"\\textbf{Inverse Trig Rule:}")
        steps.append(r"\\int \\tan^{-1} x \\, dx = x \\tan^{-1} x - \\frac{1}{2} \\ln(1 + x^2) + C")
        return steps

    if expr == sp.sinh(x):
        steps.append(r"\\textbf{Hyperbolic Rule:}")
        steps.append(r"\\int \\sinh x \\, dx = \\cosh x + C")
        return steps

    if expr == sp.cosh(x):
        steps.append(r"\\textbf{Hyperbolic Rule:}")
        steps.append(r"\\int \\cosh x \\, dx = \\sinh x + C")
        return steps

    if expr == sp.tanh(x):
        steps.append(r"\\textbf{Hyperbolic Rule:}")
        steps.append(r"\\int \\tanh x \\, dx = \\ln(\\cosh x) + C")
        return steps

    if expr.is_Mul and any(arg.has(x) for arg in expr.args):
        if len(expr.args) == 2:
            u, dv = expr.args
            du = sp.diff(u, x)
            v = sp.integrate(dv, x)
            uv = u * v
            int_vdu = sp.integrate(v * du, x)
            result = uv - int_vdu
            steps.append(r"\\textbf{Integration by Parts:}")
            steps.append(r"\\begin{align*}")
            steps.append(r"& \text{Let } u = %s, \quad dv = %s \, dx \\" % (sp.latex(u), sp.latex(dv)))
            steps.append(r"& \text{Then } du = %s \, dx, \quad v = %s \\" % (sp.latex(du), sp.latex(v)))
            steps.append(r"& \int %s \, dx = uv - \int v \, du \\" % sp.latex(expr))
            steps.append(r"& = %s - \int %s \, dx \\" % (sp.latex(uv), sp.latex(v * du)))
            steps.append(r"& = %s + C" % sp.latex(result))
            steps.append(r"\\end{align*}")
            return steps

    if expr.is_rational_function(x):
        num, den = expr.as_numer_denom()
        if den.as_poly(x).degree() > 1:
            result = sp.apart(expr, x)
            steps.append(r"\\textbf{Partial Fractions Decomposition:}")
            steps.append(r"Rewrite: \( %s = %s \)" % (sp.latex(expr), sp.latex(result)))
            steps.append(r"Now integrate each term:")
            for term in result.as_ordered_terms():
                steps += step_by_step_antiderivative(term)
            return steps

    result = sp.integrate(expr, x)
    steps.append(r"\\textbf{General Rule (Auto Integration):}")
    steps.append(r"This function does not match a standard rule. Computing using built-in integration:")
    steps.append(r"\\begin{align*}")
    steps.append(r"& \text{Let } f(x) = %s \\" % sp.latex(expr))
    steps.append(r"& \int f(x) \, dx = %s + C" % sp.latex(result))
    steps.append(r"\\end{align*}")
    return steps


def run():
    st.header("∫ Antiderivative Visualizer")
    st.markdown("Enter a function to compute its antiderivative and view integration steps.")

    user_input = st.text_input("Enter a function f(x):", "x*exp(x)")
    if not user_input:
        return

    try:
        expr = sp.sympify(user_input, locals=sympy_locals)
    except Exception as e:
        st.error(f"Invalid input: {e}")
        return

    st.subheader("🧮 Antiderivative")
    st.latex(rf"\int {sp.latex(expr)} \, dx = {sp.latex(sp.integrate(expr, x))} + C")

    st.subheader("🔎 Step-by-Step")
    for step in step_by_step_antiderivative(expr):
        st.latex(step)

    f_np = sp.lambdify(x, expr, modules=["numpy"])
    F_np = sp.lambdify(x, sp.integrate(expr, x), modules=["numpy"])
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

    st.subheader("📊 Definite Integral")
    a_str = st.text_input("Lower bound a (e.g., 0, pi, -oo):", "0")
    b_str = st.text_input("Upper bound b (e.g., 1, pi/2, oo):", "1")
    try:
        a_val = sp.sympify(a_str, locals=sympy_locals)
        b_val = sp.sympify(b_str, locals=sympy_locals)
        definite_result = sp.integrate(expr, (x, a_val, b_val))
        st.latex(rf"\int_{{{sp.latex(a_val)}}}^{{{sp.latex(b_val)}}} {sp.latex(expr)} \, dx = {sp.latex(definite_result)}")
    except Exception as e:
        st.warning(f"Could not compute definite integral: {e}")

if __name__ == "__main__":
    run()

