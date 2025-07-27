import streamlit as st
import sympy as sp
from sympy.abc import x
import matplotlib.pyplot as plt
import numpy as np
import random


def run():
    st.header("🧠 Antiderivative Visualizer")
    st.markdown("""
    Enter a function and explore its antiderivative (indefinite integral) symbolically and graphically.
    """)

    # Function input
    st.subheader("📥 Enter a Function")
    f_input = st.text_input("f(x) =", "x**2 + 1")
    try:
        fx = sp.sympify(f_input)
        F = sp.integrate(fx, x)
    except:
        st.error("Invalid function. Please enter a valid mathematical expression.")
        return

    # Display symbolic antiderivative
    st.subheader("🧮 Symbolic Antiderivative")
    st.latex(rf"F(x) = \int {sp.latex(fx)} \, dx = {sp.latex(F)} + C")

    # Graphs
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

    # -------------------------------
    # 📊 Multiple Choice Quiz Section
    # -------------------------------
    st.subheader("📊 Multiple Choice Quiz: Antiderivative")

    quiz_fx = fx  # use the main input function
    quiz_expr_latex = sp.latex(fx)

    # Compute the correct antiderivative symbolically
    F_correct = sp.integrate(quiz_fx, x)
    F_correct_str = sp.latex(F_correct) + " + C"

    # Generate distractors
    distractors = []
    while len(distractors) < 3:
        wrong = F_correct + random.choice([1, -1, 2, -2, x, -x])
        wrong_str = sp.latex(wrong) + " + C"
        if wrong_str != F_correct_str and wrong_str not in distractors:
            distractors.append(wrong_str)

    # Combine and shuffle
    options = [F_correct_str] + distractors
    random.shuffle(options)

    # Display the question
    st.markdown(f"**What is an antiderivative of** $f(x) = {quiz_expr_latex}$?")
    answer = st.radio("Choose the correct answer:", options)

    # Check answer
    if st.button("✅ Submit Antiderivative Answer"):
        if answer == F_correct_str:
            st.success("Correct! 🎉 That's the right antiderivative.")
        else:
            st.error("Oops! That's not quite right. Review the integration process.")

    # -------------------------------
    # 📉 Accumulated Area Visualization
    # -------------------------------
    st.subheader("📊 Visualizing Accumulated Area")
    a_val = st.slider("Choose starting point a for the integral", -5.0, 5.0, value=-2.0, step=0.1)
    b_val = st.slider("Move the endpoint b to accumulate area", a_val, 5.0, value=2.0, step=0.1)

    area_val = sp.integrate(fx, (x, a_val, b_val))
    st.latex(rf"\int_{{{a_val}}}^{{{b_val}}} {sp.latex(fx)} \, dx = {sp.latex(area_val)}")

    # Highlight the area under the curve
    x_fill = np.linspace(a_val, b_val, 300)
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
