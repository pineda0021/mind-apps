import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
from sympy.abc import x
import plotly.graph_objs as go

def run():
    st.header("♾️ Limits Visualizer (with Piecewise + One-Sided Support)")
    st.markdown("Explore symbolic limits, removable discontinuities, and visual behavior around a number.")

    # User input
    fx_input = st.text_input("Enter a function f(x):", "Piecewise((x**2, x < 2), (3*x, x >= 2))")
    fx_input = fx_input.replace("^", "**").replace("sqrt", "sp.sqrt")
    user_a = st.number_input("Approach x → a:", value=2.0, step=0.1)

    direction = st.radio("Limit direction:", ["Two-sided", "Left-hand", "Right-hand"])

    try:
        fx_expr = sp.sympify(fx_input, evaluate=False)
        f_np = sp.lambdify(x, fx_expr, modules=["numpy"])
    except Exception as e:
        st.error(f"Invalid function: {e}")
        return

    simplified_expr = sp.simplify(fx_expr)

    st.subheader("🧮 Symbolic Simplification")
    st.latex(f"f(x) = {sp.latex(fx_expr)}")
    st.markdown(f"Simplified:  \n$f(x) = {sp.latex(simplified_expr)}$")

    # --- 3D Plot ---
    st.subheader("📈 3D Plot of f(x)")
    x_vals = np.linspace(user_a - 4, user_a + 4, 400)
    x_vals_filtered = x_vals[np.abs(x_vals - user_a) > 1e-6]
    try:
        y_vals = f_np(x_vals_filtered)
    except:
        st.error("Could not evaluate function.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x_vals_filtered.tolist(),
                               y=[0]*len(x_vals_filtered),
                               z=y_vals.tolist(),
                               mode='lines', name='f(x)', line=dict(color='blue')))

    # Discontinuity/hole marker
    try:
        original_val = fx_expr.subs(x, user_a)
        simplified_val = simplified_expr.subs(x, user_a)
        if not original_val.is_real and simplified_val.is_real:
            fig.add_trace(go.Scatter3d(x=[float(user_a)], y=[0], z=[float(simplified_val)],
                                       mode='markers',
                                       marker=dict(size=6, color='red', symbol='circle-open'),
                                       name=f"Hole at x = {user_a}"))
    except:
        pass

    fig.update_layout(
        title="Graph of f(x)",
        scene=dict(
            xaxis_title='x',
            yaxis_title='',
            zaxis_title='f(x)',
            camera=dict(eye=dict(x=1.4, y=0.6, z=1.5))
        ),
        height=600,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Table of values ---
    st.subheader(f"📊 Limit Table Around x = {user_a}")
    deltas = [0.1, 0.01, 0.001]
    x_left = [round(user_a - d, 6) for d in reversed(deltas)]
    x_right = [round(user_a + d, 6) for d in deltas]
    x_points = x_left + x_right

    rows = []
    for xi in x_points:
        try:
            yi = round(f_np(xi), 6)
        except:
            yi = "undefined"
        rows.append((xi, yi))

    df = pd.DataFrame(rows, columns=["x", "f(x)"])
    st.dataframe(df)

    # --- Symbolic limit ---
    st.subheader("🔍 Step-by-Step Limit Evaluation")

    try:
        lim_type = {"Two-sided": "both", "Left-hand": "-", "Right-hand": "+"}[direction]
        factored = sp.factor(fx_expr)
        canceled = sp.cancel(fx_expr)

        # Choose limit type
        if lim_type == "both":
            lim_val = sp.limit(fx_expr, x, user_a)
            lim_notation = r"\lim_{x \to " + str(user_a) + "} f(x)"
        elif lim_type == "-":
            lim_val = sp.limit(fx_expr, x, user_a, dir='-')
            lim_notation = r"\lim_{x \to " + str(user_a) + "^-} f(x)"
        else:
            lim_val = sp.limit(fx_expr, x, user_a, dir='+')
            lim_notation = r"\lim_{x \to " + str(user_a) + "^+} f(x)"

        # Step-by-step display
        st.markdown(f"**1. Original Expression:**  \n$f(x) = {sp.latex(fx_expr)}$")
        st.markdown(f"**2. Factored Form:**  \n$f(x) = {sp.latex(factored)}$")
        st.markdown(f"**3. Simplified:**  \n$f(x) = {sp.latex(canceled)}$")
        st.markdown(f"**4. Compute the Limit:**")
        st.latex(lim_notation + " = " + sp.latex(lim_val))

    except Exception as e:
        st.warning("⚠️ Unable to compute symbolic limit. Try a simpler function.")

    # --- Reflection ---
    st.subheader("💭 Reflection")
    feedback = st.text_area("What did you learn about limits today?")
    if feedback:
        st.info("🧠 Thanks for sharing your thoughts!")

if __name__ == "__main__":
    run()
