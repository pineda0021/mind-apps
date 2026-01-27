import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.abc import x
from matplotlib.patches import Rectangle

def add_rect(ax, x0, dx, h, **kwargs):
    """Rectangle from y=0 to y=h (works for negative h too)."""
    y0 = min(0, h)
    ax.add_patch(Rectangle((x0, y0), dx, abs(h), **kwargs))

def run():
    st.subheader("∑ Riemann Sum Explorer")

    st.markdown("""
    Explore how rectangles approximate the area under a curve using different types of Riemann sums.
    """)

    f_input = st.text_input("Function f(x):", "x**2")
    a = st.number_input("Start of interval a:", value=0.0)
    b = st.number_input("End of interval b:", value=2.0)
    n = st.slider("Number of subintervals n:", 1, 100, 10)

    # NEW labels
    sum_type = st.selectbox(
        "Sum Type:",
        ["Lower Sum", "Upper Sum", "Midpoint", "Trapezoidal"]
    )

    # (Optional) makes Upper/Lower truly stay out/in even if curve peaks inside a subinterval
    samples_per_interval = st.slider("Samples per subinterval (Upper/Lower accuracy):", 2, 200, 50)

    try:
        fx = sp.sympify(f_input)
        f = sp.lambdify(x, fx, modules=['numpy'])
    except Exception as e:
        st.error(f"Invalid function: {e}")
        return

    X = np.linspace(a, b, 1000)
    Y = f(X)

    xi = np.linspace(a, b, n + 1)
    dx = (b - a) / n

    # Precompute subinterval sample grid for Upper/Lower
    # shape: (n, samples_per_interval)
    t = np.linspace(0, 1, samples_per_interval)
    grid = xi[:-1, None] + dx * t[None, :]
    vals = f(grid)

    if sum_type == "Lower Sum":
        height = np.min(vals, axis=1)   # inf approx
        approx = float(np.sum(height * dx))
    elif sum_type == "Upper Sum":
        height = np.max(vals, axis=1)   # sup approx
        approx = float(np.sum(height * dx))
    elif sum_type == "Midpoint":
        x_mid = (xi[:-1] + xi[1:]) / 2
        height = f(x_mid)
        approx = float(np.sum(height * dx))
    elif sum_type == "Trapezoidal":
        h = f(xi)
        approx = float((dx / 2) * np.sum(h[:-1] + h[1:]))

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X, Y, 'k', label=f"f(x) = {f_input}")

    if sum_type in ["Lower Sum", "Upper Sum", "Midpoint"]:
        for i in range(n):
            add_rect(
                ax, xi[i], dx, height[i],
                facecolor='skyblue', edgecolor='black', alpha=0.6
            )

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Riemann Sum Visualization")
    ax.grid(True)
    ax.legend()

    # Y-limits that won't clip rectangles
    y_candidates = [0, np.nanmin(Y), np.nanmax(Y)]
    if sum_type in ["Lower Sum", "Upper Sum", "Midpoint"]:
        y_candidates += [np.nanmin(height), np.nanmax(height)]
    y_min, y_max = float(np.nanmin(y_candidates)), float(np.nanmax(y_candidates))
    pad = 0.08 * (y_max - y_min if y_max != y_min else 1.0)

    ax.set_xlim(a - 0.05 * (b - a), b + 0.05 * (b - a))
    ax.set_ylim(y_min - pad, y_max + pad)

    st.pyplot(fig)

    # Output
    st.latex(rf"\text{{{sum_type}}} = {approx:.4f}")

    # Exact value
    F = sp.integrate(fx, (x, a, b))
    true_val = float(F)
    st.latex(rf"\text{{True Area}} = \int_{{{a}}}^{{{b}}} {sp.latex(fx)} \, dx = {true_val:.4f}")

    # Error
    abs_error = abs(true_val - approx)
    rel_error = abs_error / abs(true_val) if true_val != 0 else float('nan')
    st.markdown(f"**Absolute Error:** {abs_error:.6f}")
    st.markdown(f"**Relative Error:** {rel_error:.6%}")

if __name__ == "__main__":
    run()

