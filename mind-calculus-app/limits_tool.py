# limits_tool.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp
from sympy.abc import x
from IPython.display import HTML

st.subheader("✴️ Limits Visualizer")

st.markdown("""
This tool helps visualize what happens to a function as \( x \to c \). It includes animated graphs and table views for a better understanding of removable and non-removable discontinuities.
""")

# Function input
func_str = st.text_input("Function (e.g. (x**2 - 5*x + 6)/(x - 2)):", "(x**2 - 5*x + 6)/(x - 2)")
c = st.number_input("Approach x → c:", value=2.0)

try:
    f = sp.sympify(func_str)
    f_lambdified = sp.lambdify(x, f, modules=["numpy"])
except Exception as e:
    st.error(f"Invalid function: {e}")
    st.stop()

x_vals = np.array([c - 0.1, c - 0.01, c - 0.001, c, c + 0.001, c + 0.01, c + 0.1])
y_vals = []
for val in x_vals:
    try:
        y = f_lambdified(val)
        y_vals.append(round(y, 6))
    except:
        y_vals.append("undefined")

# Table output
st.markdown("### 📋 Table of Values")
st.table({"x": x_vals, "f(x)": y_vals})

# Plot function around x=c
X = np.linspace(c - 1, c + 1, 400)
Y = f_lambdified(X)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(X, Y, label=f"f(x) = {func_str}", color="blue")

# Check if limit exists
try:
    y_at_c = f_lambdified(c)
    ax.plot(c, y_at_c, 'ro', markerfacecolor='white', label=f"f({c})")
except:
    y_left = f_lambdified(c - 0.0001)
    y_right = f_lambdified(c + 0.0001)
    y_avg = (y_left + y_right) / 2
    ax.plot(c, y_avg, 'ro', markerfacecolor='white', label=f"Hole at x={c}")

ax.axvline(c, color='red', linestyle='--', label=f"x = {c}")
ax.set_title("Function Behavior Near x = c")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Challenge
st.markdown("### ✅ Challenge")
user_limit = st.text_input(f"What is the limit as x → {c}?")
true_limit = sp.limit(f, x, c)

if user_limit:
    try:
        user_expr = sp.sympify(user_limit)
        if sp.simplify(user_expr - true_limit) == 0:
            st.success("Correct! ✅")
        else:
            st.error(f"Incorrect. The correct limit is: {true_limit}")
    except:
        st.warning("Could not interpret your input.")
