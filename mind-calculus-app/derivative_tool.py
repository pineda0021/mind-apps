def run():

import streamlit as st
import sympy as sp
from sympy.abc import x
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd


def step_by_step_derivation(expr):
    steps = []
    if expr.is_Mul:
        u, v = expr.args
        u_diff = sp.diff(u, x)
        v_diff = sp.diff(v, x)
        steps.append(rf"Product Rule: $\frac{{d}}{{dx}}[{sp.latex(u)} \cdot {sp.latex(v)}] = {sp.latex(u)} \cdot \frac{{d}}{{dx}}[{sp.latex(v)}] + {sp.latex(v)} \cdot \frac{{d}}{{dx}}[{sp.latex(u)}]$")
        steps.append(rf"= {sp.latex(u)}({sp.latex(v_diff)}) + {sp.latex(v)}({sp.latex(u_diff)})")
        steps.append(rf"= {sp.latex(u * v_diff + v * u_diff)}")
    elif expr.is_Pow:
        base, exp = expr.args
        if base.has(x):
            steps.append(rf"Chain Rule: $\frac{{d}}{{dx}}[{sp.latex(base)}^{{{sp.latex(exp)}}}] = {sp.latex(exp)} \cdot {sp.latex(base)}^{{{sp.latex(exp - 1)}}} \cdot \frac{{d}}{{dx}}[{sp.latex(base)}]$")
        else:
            steps.append(rf"Power Rule: $\frac{{d}}{{dx}}[{sp.latex(expr)}] = {sp.latex(sp.diff(expr, x))}$")
    else:
        steps.append(rf"Derivative: $\frac{{d}}{{dx}}[{sp.latex(expr)}] = {sp.latex(sp.diff(expr, x))}$")
    return steps

def run():
    st.header("\U0001F9E0 Derivative Visualizer")
    st.markdown("""
    Enter a function and explore its derivative both symbolically and graphically. Compare f(x) and f'(x), and investigate the slope of the tangent line at any point.
    """)

    # Function input
    st.subheader("\U0001F4E5 Enter a Function")
    f_input = st.text_input("f(x) =", "x**3 - 3*x + 1")
    try:
        fx = sp.sympify(f_input)
        dfx = sp.diff(fx, x)
    except:
        st.error("Invalid function. Please enter a valid mathematical expression.")
        return

    # Display symbolic derivative
    st.subheader("\U0001F9EE Symbolic Derivative")
    st.latex(rf"f'(x) = \frac{{d}}{{dx}}[{sp.latex(fx)}] = {sp.latex(dfx)}")

    # Step-by-step explanation
    st.subheader("\U0001FA5C Step-by-Step Derivation")
    for step in step_by_step_derivation(fx):
        st.markdown(step)

    # Graphs of f(x) and f'(x)
    st.subheader("\U0001F4C8 Graph of f(x) and f'(x)")
    f_np = sp.lambdify(x, fx, modules=["numpy"])
    df_np = sp.lambdify(x, dfx, modules=["numpy"])

    X = np.linspace(-5, 5, 400)
    Y = f_np(X)
    Y_prime = df_np(X)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X, Y, label="f(x)", color="blue")
    ax.plot(X, Y_prime, label="f'(x)", color="orange")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Function and Derivative")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Tangent line visualization
    st.subheader("\U0001F4CD Tangent Line at a Point")
    a_val = st.slider("Choose a point x = a", -5.0, 5.0, value=1.0, step=0.1)
    f_a = f_np(a_val)
    df_a = df_np(a_val)
    tangent_line = lambda x_val: df_a * (x_val - a_val) + f_a

    Y_tangent = tangent_line(X)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(X, Y, label="f(x)", color="blue")
    ax2.plot(X, Y_tangent, label=f"Tangent at x = {a_val}", linestyle="--", color="red")
    ax2.scatter([a_val], [f_a], color="black", zorder=5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Tangent Line Visualization")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    # Derivative Table Animation (h -> 0)
    st.subheader("\U0001F4C9 Derivative Table: Definition of Derivative")
    h_vals = [1, 0.5, 0.1, 0.01, 0.001]
    slopes = [(f_np(a_val + h) - f_np(a_val)) / h for h in h_vals]
    df_exact = df_np(a_val)

    table_data = {
        "h": h_vals,
        "[f(a+h) - f(a)] / h": [round(s, 6) for s in slopes]
    }
    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table)
    st.latex(rf"\lim_{{h \to 0}} \frac{{f({a_val}+h)-f({a_val})}}{{h}} = {df_exact:.6f}")

    # Quiz
    st.subheader("\U0001F4CA Multiple Choice Quiz: Derivative")
    correct = sp.diff(fx, x)
    correct_str = sp.latex(correct)

    # Generate distractors
    distractors = []
    while len(distractors) < 3:
        wrong = correct + random.choice([1, -1, 2, -2, x, -x])
        wrong_str = sp.latex(wrong)
        if wrong_str != correct_str and wrong_str not in distractors:
            distractors.append(wrong_str)

    options = [correct_str] + distractors
    random.shuffle(options)

    st.markdown(f"**What is the derivative of** $f(x) = {sp.latex(fx)}$?")
    selected = st.radio("Choose the correct answer:", options)

    if st.button("✅ Submit Derivative Answer"):
        if selected == correct_str:
            st.success("Correct! 🎉 Great job.")
        else:
            st.error("Oops! That's not quite right. Try reviewing the rules of differentiation.")

if __name__ == "__main__":
    run()

