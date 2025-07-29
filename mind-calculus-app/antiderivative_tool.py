try:
    import streamlit as st
except ModuleNotFoundError:
    raise ImportError("The Streamlit package is not installed. Please run 'pip install streamlit' in your environment.")

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
        steps.append("Sum Rule:")
        for term in expr.args:
            steps += step_by_step_antiderivative(term)
        return steps

    if expr.is_Number:
        steps.append("Constant Rule:")
        steps.append(r"$$ \int %s \, dx = %sx + C $$" % (sp.latex(expr), sp.latex(expr)))
        return steps

    if expr.is_Pow and expr.args[0] == x:
        n = expr.args[1]
        if n != -1:
            result = sp.integrate(expr, x)
            steps.append("Power Rule:")
            steps.append(r"$$ \int x^{%s} \, dx = \frac{x^{%s}}{%s} + C $$" % (sp.latex(n), sp.latex(n+1), sp.latex(n+1)))
        else:
            steps.append("Special Case:")
            steps.append(r"$$ \int \frac{1}{x} \, dx = \ln|x| + C $$")
        return steps

    if expr == sp.exp(x):
        steps.append("Exponential Rule:")
        steps.append(r"$$ \int e^x \, dx = e^x + C $$")
        return steps

    if expr == sp.exp(-x):
        steps.append("Exponential Rule (Negative Exponent):")
        steps.append(r"$$ \int e^{-x} \, dx = -e^{-x} + C $$")
        return steps

    if expr == sp.log(x):
        steps.append("Logarithmic Rule:")
        steps.append(r"$$ \int \ln x \, dx = x\ln x - x + C $$")
        return steps

    if expr == sp.sin(x):
        steps.append("Trig Rule:")
        steps.append(r"$$ \int \sin x \, dx = -\cos x + C $$")
        return steps

    if expr == sp.cos(x):
        steps.append("Trig Rule:")
        steps.append(r"$$ \int \cos x \, dx = \sin x + C $$")
        return steps

    if expr == sp.tan(x):
        steps.append("Trig Rule:")
        steps.append(r"$$ \int \tan x \, dx = -\ln|\cos x| + C $$")
        return steps

    if expr == sp.asin(x):
        steps.append("Inverse Trig Rule:")
        steps.append(r"$$ \int \sin^{-1} x \, dx = x \sin^{-1} x + \sqrt{1 - x^2} + C $$")
        return steps

    if expr == sp.acos(x):
        steps.append("Inverse Trig Rule:")
        steps.append(r"$$ \int \cos^{-1} x \, dx = x \cos^{-1} x - \sqrt{1 - x^2} + C $$")
        return steps

    if expr == sp.atan(x):
        steps.append("Inverse Trig Rule:")
        steps.append(r"$$ \int \tan^{-1} x \, dx = x \tan^{-1} x - \frac{1}{2} \ln(1 + x^2) + C $$")
        return steps

    if expr == sp.sinh(x):
        steps.append("Hyperbolic Rule:")
        steps.append(r"$$ \int \sinh x \, dx = \cosh x + C $$")
        return steps

    if expr == sp.cosh(x):
        steps.append("Hyperbolic Rule:")
        steps.append(r"$$ \int \cosh x \, dx = \sinh x + C $$")
        return steps

    if expr == sp.tanh(x):
        steps.append("Hyperbolic Rule:")
        steps.append(r"$$ \int \tanh x \, dx = \ln(\cosh x) + C $$")
        return steps

    if expr.is_Mul:
        factors = expr.args
        for i in range(len(factors)):
            for j in range(len(factors)):
                if i != j and factors[i] == sp.diff(factors[j], x):
                    u = factors[j]
                    du = factors[i]
                    steps.append("**Chain Rule (u-substitution):**")
                    steps.append(rf"Let $u = {sp.latex(u)}$, then $du = {sp.latex(sp.diff(u, x))} \, dx$")
                    steps.append(rf"Rewrite: $\int {sp.latex(expr)} \, dx = \int u \, du$")
                    steps.append(rf"$= {sp.latex(sp.integrate(u, x))} + C$")
                    return steps

        if len(expr.args) == 2:
            u, dv = expr.args
            du = sp.diff(u, x)
            v = sp.integrate(dv, x)
            uv = u * v
            int_vdu = sp.integrate(v * du, x)
            result = uv - int_vdu
            steps.append("Integration by Parts:")
            steps.append(r"$$\begin{aligned}")
            steps.append(r"\textbf{Let:}\quad u = %s,\quad dv = %s \\" % (sp.latex(u), sp.latex(dv)))
            steps.append(r"\textbf{Then:}\quad du = %s,\quad v = %s \\" % (sp.latex(du), sp.latex(v)))
            steps.append(r"\int %s \, dx = uv - \int v \, du \\" % sp.latex(expr))
            steps.append(r"= %s - \int %s \, dx \\" % (sp.latex(uv), sp.latex(v * du)))
            steps.append(r"= %s + C" % sp.latex(result))
            steps.append(r"\end{aligned}$$")
            return steps

    result = sp.integrate(expr, x)
    steps.append("General Rule (Auto Integration):")
    steps.append(r"$$\begin{aligned}")
    steps.append(r"\text{Let } f(x) = %s \\" % sp.latex(expr))
    steps.append(r"\int f(x) \, dx = %s + C" % sp.latex(result))
    steps.append(r"\end{aligned}$$")
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
        st.markdown(f"{step}", unsafe_allow_html=True)

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

