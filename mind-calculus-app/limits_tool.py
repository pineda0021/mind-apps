import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sympy.abc import x

def run():
    st.header("♾️ Limits Visualizer")
    st.markdown("Explore symbolic limits, piecewise behavior, removable discontinuities, and visual zoom.")

    # --- Input ---
    fx_input = st.text_input("Enter a function f(x):", "Piecewise((x**2, x < 2), (3*x, x >= 2))")
    fx_input = fx_input.replace("^", "**").replace("sqrt", "sp.sqrt")
    user_a = st.number_input("Approach x → a:", value=2.0, step=0.1)
    direction = st.radio("Limit direction:", ["Two-sided", "Left-hand", "Right-hand"])

    # --- Domain Range Control ---
    st.markdown("### 🔍 Domain Zoom (x-axis range)")
    default_range = 4.0
    x_min = st.number_input("x-axis start:", value=float(user_a - default_range), step=0.1, key="x_min")
    x_max = st.number_input("x-axis end:", value=float(user_a + default_range), step=0.1, key="x_max")
    if x_min >= x_max:
        st.warning("⚠️ Make sure that x-start < x-end.")
        return

    # --- Try to parse function ---
    try:
        fx_expr = sp.sympify(fx_input, evaluate=False)
        f_np = sp.lambdify(x, fx_expr, modules=["numpy"])
        simplified_expr = sp.simplify(fx_expr)
    except Exception as e:
        st.error(f"Invalid function: {e}")
        return

    st.subheader("🧮 Symbolic Simplification")
    st.latex(f"f(x) = {sp.latex(fx_expr)}")
    st.markdown(f"Simplified:  \n$f(x) = {sp.latex(simplified_expr)}$")

    # --- Plot Preparation ---
    x_vals = np.linspace(x_min, x_max, 400)
    x_vals_filtered = x_vals[np.abs(x_vals - user_a) > 1e-6]
    try:
        y_vals = f_np(x_vals_filtered)
    except:
        st.error("Could not evaluate function.")
        return

    # --- Layout ---
    st.subheader("📈 Graph of f(x) with Discontinuity Highlight")
    col1, col2 = st.columns(2)

    # --- 3D Plot ---
    with col1:
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(
            x=x_vals_filtered.tolist(),
            y=[0]*len(x_vals_filtered),
            z=y_vals.tolist(),
            mode='lines',
            name='f(x)',
            line=dict(color='blue')
        ))

        # Hole detection
        try:
            original_val = fx_expr.subs(x, user_a)
            simplified_val = simplified_expr.subs(x, user_a)
            if (original_val.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or not sp.sympify(original_val).is_real) \
               and sp.sympify(simplified_val).is_real:
                fig3d.add_trace(go.Scatter3d(
                    x=[float(user_a)],
                    y=[0],
                    z=[float(simplified_val)],
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='circle-open'),
                    name=f"Hole at x = {user_a}"
                ))
        except:
            pass

        fig3d.update_layout(
            title="3D View of f(x)",
            scene=dict(
                xaxis_title='x',
                yaxis_title='',
                zaxis_title='f(x)',
                camera=dict(eye=dict(x=1.5, y=0.5, z=1.5))
            ),
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # --- 2D Plot ---
    with col2:
        fig2d, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_vals_filtered, y_vals, color='blue', label='f(x)')

        try:
            if (original_val.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or not sp.sympify(original_val).is_real) \
               and sp.sympify(simplified_val).is_real:
                ax.plot(float(user_a), float(simplified_val), 'ro', markerfacecolor='white', markersize=8,
                        label=f"Hole at x = {user_a}")
        except:
            pass

        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title("2D Cross-Section")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig2d)

    # --- Table Around x = a ---
    st.subheader(f"📊 Table of Values Near x = {user_a}")
    delta_list = [0.1, 0.01, 0.001]
    x_points = [round(user_a - d, 6) for d in reversed(delta_list)] + [round(user_a + d, 6) for d in delta_list]
    rows = []
    for xi in x_points:
        try:
            yi = round(f_np(xi), 6)
        except:
            yi = "undefined"
        rows.append((xi, yi))
    st.dataframe(pd.DataFrame(rows, columns=["x", "f(x)"]))

    # --- Step-by-Step Limit ---
    st.subheader("🔍 Step-by-Step Limit Evaluation")
    try:
        lim_type = {"Two-sided": "both", "Left-hand": "-", "Right-hand": "+"}[direction]
        factored = sp.factor(fx_expr)
        canceled = sp.cancel(fx_expr)

        if lim_type == "both":
            lim_val = sp.limit(fx_expr, x, user_a)
            lim_tex = r"\lim_{x \to " + str(user_a) + "} f(x)"
        elif lim_type == "-":
            lim_val = sp.limit(fx_expr, x, user_a, dir='-')
            lim_tex = r"\lim_{x \to " + str(user_a) + "^-} f(x)"
        else:
            lim_val = sp.limit(fx_expr, x, user_a, dir='+')
            lim_tex = r"\lim_{x \to " + str(user_a) + "^+} f(x)"

        st.markdown(f"**1. Original Expression:**  \n$f(x) = {sp.latex(fx_expr)}$")
        st.markdown(f"**2. Factored Form:**  \n$f(x) = {sp.latex(factored)}$")
        st.markdown(f"**3. Simplified:**  \n$f(x) = {sp.latex(canceled)}$")
        st.markdown(f"**4. Limit Evaluation:**")
        st.latex(lim_tex + " = " + sp.latex(lim_val))

    except Exception as e:
        st.warning("⚠️ Unable to compute symbolic limit.")

    # --- Reflection ---
    st.subheader("💭 Reflection")
    feedback = st.text_area("What did you learn about limits today?")
    if feedback:
        st.info("🧠 Thanks for your insight!")

if __name__ == "__main__":
    run()
