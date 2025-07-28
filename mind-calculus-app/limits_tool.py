import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import sympy as sp
from sympy.abc import x

def run():
    st.header("♾️ Limits Visualizer")
    st.markdown("Explore removable discontinuities, symbolic limits, and interactive graphs.")

    user_fx_input = st.text_input("Enter a function f(x):", "Piecewise((x**2, x < 2), (3*x, x >= 2))")
    user_fx_input = user_fx_input.replace("sqrt", "sp.sqrt")
    user_a = st.number_input("Approach x → a:", value=2.0, step=0.1)

    st.markdown("### 🔍 Zoom Range")
    x_min = st.number_input("x-axis min:", value=float(user_a - 4), step=0.1)
    x_max = st.number_input("x-axis max:", value=float(user_a + 4), step=0.1)
    if x_min >= x_max:
        st.warning("x-min must be less than x-max")
        return

    try:
        fx_expr = sp.sympify(user_fx_input)
        simplified_expr = sp.simplify(fx_expr)
        f_np = sp.lambdify(x, fx_expr, modules=["numpy"])
    except Exception as e:
        st.error(f"Function parsing failed: {e}")
        return

    st.subheader("🧮 Symbolic Simplification")
    st.latex(f"f(x) = {sp.latex(fx_expr)}")
    st.markdown(f"**Simplified:** $f(x) = {sp.latex(simplified_expr)}$")

    # --- Hole Detection ---
    hole_exists = False
    try:
        original_val = fx_expr.subs(x, user_a)
        limit_val = sp.limit(fx_expr, x, user_a)
        if not original_val.is_real and limit_val.is_real:
            hole_exists = True
        elif original_val != simplified_expr.subs(x, user_a):
            hole_exists = True
        y_hole = float(limit_val)
    except:
        y_hole = None

    # --- 3D Graph ---
    st.subheader("📊 3D Graph with Hole")
    x_vals = np.linspace(x_min, x_max, 400)
    x_vals_filtered = x_vals[np.abs(x_vals - user_a) > 1e-6]
    try:
        y_vals = f_np(x_vals_filtered)
    except:
        st.error("Numerical evaluation failed.")
        return

    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(
        x=x_vals_filtered, y=[0]*len(x_vals_filtered), z=y_vals,
        mode='lines', name='f(x)', line=dict(color='blue')))
    if hole_exists and y_hole is not None:
        fig3d.add_trace(go.Scatter3d(
            x=[float(user_a)], y=[0], z=[y_hole],
            mode='markers', marker=dict(size=8, color='red', symbol='circle-open'),
            name=f"Hole at x = {user_a}"))
    fig3d.update_layout(scene=dict(xaxis_title='x', yaxis_title='', zaxis_title='f(x)'),
                        height=500, showlegend=True)
    st.plotly_chart(fig3d, use_container_width=True)

    # --- 2D Graph ---
    st.subheader("📈 2D Cross-Section")
    fig2d, ax = plt.subplots()
    ax.plot(x_vals_filtered, y_vals, label="f(x)", color="blue")
    if hole_exists and y_hole is not None:
        ax.plot(user_a, y_hole, 'ro', markerfacecolor='white', markersize=8,
                label=f"Hole at x = {user_a}")
    ax.set_title("2D Graph of f(x)")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig2d)

    # --- Table of Values ---
    st.subheader(f"🧮 Limit Table Around x = {user_a}")
    deltas = [0.1, 0.01, 0.001]
    points = [round(user_a - d, 6) for d in deltas[::-1]] + [round(user_a + d, 6) for d in deltas]
    rows = []
    for pt in points:
        try:
            val = round(f_np(pt), 6)
        except:
            val = "undefined"
        rows.append((pt, val))
    st.table({"x": [r[0] for r in rows], "f(x)": [r[1] for r in rows]})

    # --- Step-by-Step Limit ---
    st.subheader("🧠 Step-by-Step Limit Solving")
    try:
        factored = sp.factor(fx_expr)
        canceled = sp.cancel(fx_expr)
        st.markdown(f"**1. Original Expression:** $f(x) = {sp.latex(fx_expr)}$")
        st.markdown(f"**2. Factored Form:** $f(x) = {sp.latex(factored)}$")
        st.markdown(f"**3. Cancelled Form:** $f(x) = {sp.latex(canceled)}$")
        st.markdown(f"**4. Limit:** $\\lim_{{x \\to {user_a}}} f(x) = {sp.latex(limit_val)}$")
    except:
        st.warning("Could not compute symbolic limit.")

    st.subheader("💬 Reflection")
    feedback = st.text_area("What did you learn about limits today?")
    if feedback:
        st.success("Thanks for your reflection!")

if __name__ == "__main__":
    run()
