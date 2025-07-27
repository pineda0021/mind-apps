# antiderivative_tool.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.abc import x

st.subheader("∫ Antiderivative Visualizer")

st.markdown("""
Visualize a function and its symbolic antiderivative. Understand how integration accumulates area and reverses differentiation.
""")

# Function input
f_input = st.text_input("Function f(x):", "cos(x)")

try:
    fx = sp.sympify(f_input)
    F = sp.integrate(fx, x)
    f = sp.lambdify(x, fx, modules=['numpy'])
    F_func = sp.lambdify(x, F, modules=['numpy'])
except Exception as e:
    st.error(f"Invalid input: {e}")
    st.stop()

# Plotting
x_vals = np.linspace(-10, 10, 500)
y_vals = f(x_vals)
Fy_vals = F_func(x_vals)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_vals, y_vals, label=f"f(x) = {f_input}", color='blue')
ax.plot(x_vals, Fy_vals, label=f"F(x) = {sp.latex(F)} + C", color='green')
ax.axhline(0, color='gray', lw=0.5)
ax.set_title("Function and its Antiderivative")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Challenge
st.markdown("### ✅ Challenge")
st.markdown(f"What is the antiderivative of \( f(x) = {sp.latex(fx)} \)?")
user_input = st.text_input("Type your answer (omit +C):")

try:
    user_expr = sp.sympify(user_input)
    if sp.simplify(user_expr - F) == 0:
        st.success("✅ Correct symbolic antiderivative!")
    else:
        st.error(f"❌ Incorrect. Correct antiderivative is: {sp.latex(F)} + C")
except:
    if user_input:
        st.warning("⚠️ Unable to parse your input.")
