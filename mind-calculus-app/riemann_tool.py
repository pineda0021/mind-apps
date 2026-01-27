import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.abc import x
from matplotlib.patches import Rectangle

def add_rect(ax, x0, dx, h, **kwargs):
    """Draw a rectangle from y=0 to y=h, correctly handling negative h."""
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

    # Renamed options to match what you want
    sum_type = st.selectbox(
        "Sum Type:",
        ["Lower Sum", "Upper Sum", "Midpoint", "Trapezoidal", "Show Both (Lower & Upper)"]
    )

    try:
        fx = sp.sympify(f_input)
        f = sp.lambdify(x, fx, modules=["numpy"])
    except Exception as e:
        st.error(f"Invalid function: {e}")
        return

    X = np.linspace(a, b, 1000)
    Y = f(X)

    xi = np.linspace(a, b, n + 1)
    dx = (b - a) / n

    # Endpoint values for each subinterval
    x_left = xi[:-1]
    x_right = xi[1:]
    h_left = f(x_left)
    h_right = f(x_right)

    approx = None
    lower = upper = None
    heights_to_plot = None

    if sum_type == "Lower Sum":
        heights_to_plot = np.minimum(h_left, h_right)
        approx = float(np.sum(heights_to_plot * dx))
    elif sum_type == "Upper Sum":
        heights_to_plot = np.maximum(h_left, h_right)
        approx = float(np.sum(heights_to_plot * dx))
    elif sum_type == "Midpoint":
        x_mid = (x_left + x_right) / 2
        heights_to_plot = f(x_mid)
        approx = float(np.sum(heights_to_plot * dx))
    elif sum_type == "Trapezoidal":
        heights = f(xi)
        approx = float((dx / 2) * np.sum(heights[:-1] + heights[1:]))
    elif sum_type == "Show Both (Lower & Upper)":
        lower_heights = np.minimum(h_left, h_right)
        upper_heights = np.maximum(h_left, h_right)
        lower = float(np.sum(lower_heights * dx))
        upper = float(np.sum(upper_heights * dx))

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X, Y, "k", label=f"f(x) = {f_input}")

    if sum_type in ["Lower Sum", "Upper Sum", "Midpoint"]:
        for i in range(n):
            add_rect(
                ax, xi[i], dx, heights_to_plot[i],
                facecolor="skyblue", edgecolor="black", alpha=0.6
            )

        ax.legend([f"f(x) = {f_input}", sum_type])
    elif sum_type == "Trapezoidal":
        # Optional: visualize trapezoids as rectangles is misleading; keep curve only or implement trapezoids.
        ax.legend()
    else:  # Show Both
        lower_heights = np.minimum(h_left, h_right)
        upper_heights = np.maximum(h_left, h_right)

        for i in range(n):
            # lower (inside) in pink
            add_rect(ax, xi[i], dx, lower_heights[i], facecolor="pink", edgecolor="black", alpha=0.45)
            # upper (outside) in violet
            add_rect(ax, xi[i], dx, upper_heights[i], facecolor="violet", edgecolor="black", alpha=0.25)

        ax.legend([f"f(x) = {f_input}", "Lower (pink) & Upper (violet)"])

    # Axis limits that INCLUDE rectangles so nothing clips
    candidates = [0, np.nanmin(Y), np.nanmax(Y), np.nanmin(h_left), np.nanmax(h_left), np.nanmin(h_right), np.nanmax(h_right)]
    if sum_type in ["Lower Sum", "Upper Sum", "Midpoint"] and heights_to_plot is not None:
        candidates += [np.nanmin(heights_to_plot), np.nanmax(heights_to_plot)]
    if sum_type == "Show Both (Lower & Upper)":
        candidates += [np.nanmin(np.minimum(h_left, h_right)), np.nanmax(np.maximum(h_left, h_right))]

    y_min = float(np.nanmin(candidates))
    y_max = float(np.nanmax(candidates))
    pad = 0.08 * (y_max - y_min if y_max != y_min else 1.0)

    ax.set_xlim(a - 0.05 * (b - a), b + 0.05 * (b - a))
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Riemann Sum Visualization")
    ax.grid(True)
    st.pyplot(fig)

    # Output
    if sum_type == "Show Both (Lower & Upper)":
        st.latex(rf"\text{{Lower Sum}} = {lower:.4f}")
        st.latex(rf"\text{{Upper Sum}} = {upper:.4f}")
    else:
        st.latex(rf"\text{{{sum_type}}} = {approx:.4f}")

    # Exact value
    F = sp.integrate(fx, (x, a, b))
    true_val = float(F)
    st.latex(rf"\text{{True Area}} = \int_{{{a}}}^{{{b}}} {sp.latex(fx)} \, dx = {true_val:.4f}")

    # Error (only when approx exists)
    if approx is not None:
        abs_error = abs(true_val - approx)
        rel_error = abs_error / abs(true_val) if true_val != 0 else float("nan")
        st.markdown(f"**Absolute Error:** {abs_error:.6f}")
        st.markdown(f"**Relative Error:** {rel_error:.6%}")

    with st.expander("📘 What does this mean?"):
        st.markdown("""
        - **Lower Sum** uses the smaller endpoint value on each subinterval (stays under the curve).
        - **Upper Sum** uses the larger endpoint value on each subinterval (covers the curve).
        - Midpoint and Trapezoidal often give better accuracy.
        """)

if __name__ == "__main__":
    run()
