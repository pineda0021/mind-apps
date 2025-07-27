# derivative_tool.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.abc import x

st.subheader("📈 Derivative Visualizer")

st.markdown("""
Explore how a function and its derivative relate visually and symbolically. Enter a function below to compute and plot its derivative.
""")

# Function input
f_input = st.text_input("Function f(x):", "x**3 - 3*x**2 + 2")

try:
    fx = sp.sympify(f_input)
    dfx = sp.diff(fx, x)
    f = sp.lambdify(x, fx, modules=['numpy'])
    df = sp.lambdify(x, dfx, modules=['numpy'])
except Exception as e:
    st.error(f"Invalid input: {e}")
    st.stop()

# Plotting
x_vals = np.linspace(-10, 10, 500)
y_vals = f(x_vals)
dy_vals = df(x_vals)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_vals, y_vals, label=f"f(x) = {f_input}", color='blue')
ax.plot(x_vals, dy_vals, label=f"f'(x) = {sp.latex(dfx)}", color='orange')
ax.axhline(0, color='gray', lw=0.5)
ax.set_title("Function and its Derivative")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Challenge
st.markdown("### ✅ Challenge")
x_val = st.number_input("Pick a value of x:", value=1.0)
true_slope = df(x_val)
user_slope = st.text_input(f"What is f'({x_val})?")

if user_slope:
    try:
        user_expr = sp.sympify(user_slope)
        if sp.simplify(user_expr - true_slope) == 0:
            st.success("✅ Correct slope at that point!")
        else:
            st.error(f"Incorrect. f'({x_val}) = {true_slope}")
    except:
        st.warning("⚠️ Could not interpret your input.")
