import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.abc import x
from matplotlib.patches import Rectangle

st.title("🧠 MIND: Riemann Sum Visualization Tool")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

st.markdown("""
<p style='font-size: 12px; line-height: 1.6;'>
Explore the meaning of area under a curve using <strong>Riemann sums</strong>. Adjust functions, intervals, and the number of rectangles to visually compare <em>Left</em>, <em>Right</em>, <em>Midpoint</em>, <em>Trapezoidal</em>, and <em>Upper/Lower</em> approximations with the exact integral.
</p>
""", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("🔧 Parameters")
f_input = st.sidebar.text_input("Function f(x):", "0.5*x**2 + 1")
st.sidebar.caption("Try: x, sin(x), exp(x), ln(x), x**3 - 2*x + 1, ...")
example_funcs = {
    "x": "x",
    "x² + 1": "x**2 + 1",
    "sin(x)": "sin(x)",
    "eˣ (exp(x))": "exp(x)",
    "ln(x)": "ln(x)"
}
example = st.sidebar.selectbox("Example functions:", list(example_funcs.keys()))
if st.sidebar.button("Use example"):
    f_input = example_funcs[example]

a = st.sidebar.number_input("Start of interval (a):", value=-2.0)
b = st.sidebar.number_input("End of interval (b):", value=3.0)
n = st.sidebar.slider("Number of subintervals (n):", min_value=1, max_value=100, value=10)
sum_type = st.sidebar.selectbox("Riemann Sum Type:", ["Left", "Right", "Midpoint", "Trapezoidal", "Upper/Lower"])

# Define function safely
try:
    fx = sp.sympify(f_input)
    f = sp.lambdify(x, fx, modules=["numpy"])
except:
    st.error("Invalid function. Please check your syntax.")
    st.stop()

# Calculations
X = np.linspace(a, b, 1000)
Y = f(X)

xi = np.linspace(a, b, n + 1)
dx = (b - a) / n

if sum_type == "Left":
    x_sample = xi[:-1]
    height = f(x_sample)
    approx = np.sum(height * dx)
elif sum_type == "Right":
    x_sample = xi[1:]
    height = f(x_sample)
    approx = np.sum(height * dx)
elif sum_type == "Midpoint":
    x_sample = (xi[:-1] + xi[1:]) / 2
    height = f(x_sample)
    approx = np.sum(height * dx)
elif sum_type == "Trapezoidal":
    height = f(xi)
    approx = (dx / 2) * np.sum(height[:-1] + height[1:])
elif sum_type == "Upper/Lower":
    x_left = xi[:-1]
    x_right = xi[1:]
    h_left = f(x_left)
    h_right = f(x_right)
    lower = np.sum(np.minimum(h_left, h_right) * dx)
    upper = np.sum(np.maximum(h_left, h_right) * dx)
    approx = None

# Plotting
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(X, Y, 'k', label=f"f(x) = {f_input}")

if sum_type != "Upper/Lower":
    for i in range(n):
        ax.add_patch(Rectangle((xi[i], 0), dx, height[i], facecolor='skyblue', edgecolor='black', alpha=0.6))
else:
    for i in range(n):
        ax.add_patch(Rectangle((xi[i], 0), dx, np.minimum(h_left[i], h_right[i]), color='pink', alpha=0.5))
        ax.add_patch(Rectangle((xi[i], np.minimum(h_left[i], h_right[i])), dx, np.abs(h_left[i] - h_right[i]), color='violet', alpha=0.3))

ax.set_xlim(a - 0.5, b + 0.5)
ax.set_ylim(min(Y) - 1, max(Y) + 1)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Riemann Sum Visualization")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Output
if sum_type == "Upper/Lower":
    st.latex(rf"\text{{Lower Sum}} = {lower:.4f}")
    st.latex(rf"\text{{Upper Sum}} = {upper:.4f}")
else:
    st.latex(rf"\text{{{sum_type} Riemann Sum}} = {approx:.4f}")

# Exact value
F = sp.integrate(fx, (x, a, b))
true_val = float(F)
st.latex(rf"\text{{True Area}} = \int_{{{a}}}^{{{b}}} {sp.latex(fx)} \, dx = {true_val:.4f}")

# Error Calculation (optional)
if sum_type != "Upper/Lower":
    abs_error = abs(true_val - approx)
    rel_error = abs_error / abs(true_val) if true_val != 0 else float('nan')
    st.markdown(f"**Absolute Error:** {abs_error:.6f}")
    st.markdown(f"**Relative Error:** {rel_error:.6%}")

# Reflection box
with st.expander("📘 What does this mean?"):
    st.markdown(f"""
    - This app shows how **Riemann sums** estimate the **area under a curve**.
    - As infinity, the approximation becomes more accurate.
    - Different types (left, right, midpoint, trapezoid) use different sample points.
    - The **absolute error** is the difference from the exact area.
    - The **relative error** shows how significant that difference is.
    - Try changing the function, interval, and number of rectangles!
    """)
