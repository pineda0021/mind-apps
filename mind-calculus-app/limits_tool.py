import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objs as go
import sympy as sp
from sympy.abc import x
from sympy import symbols, sympify, integrate, pi, latex, simplify, Rational
import streamlit.components.v1 as components
import random


def run():
    st.header("♾️ Limits Visualizer")
    st.markdown("""
    Explore removable discontinuities, limits from a table, animation, symbolic simplification, and tangent lines.
    """)

    user_fx_input = st.text_input("Enter a function f(x):", "(x**2 - 5*x + 6)/(x - 2)")
    user_fx_input = user_fx_input.replace("sqrt", "sp.sqrt")
    user_a = st.number_input("Approach x → a:", value=2.0, step=0.1, format="%.2f")

    try:
        fx_expr = sp.sympify(user_fx_input)
        f = sp.lambdify(x, fx_expr, modules=['numpy'])
    except Exception as e:
        st.error(f"Invalid function: {e}")
        return

    simplified_expr = sp.simplify(fx_expr)
    derivative_expr = sp.diff(simplified_expr, x)
    derivative_func = sp.lambdify(x, derivative_expr, modules=['numpy'])

    st.subheader("🧮 Symbolic Simplification")
    st.latex(rf"f(x) = {sp.latex(fx_expr)}")
    st.markdown(f"The simplified expression is: $f(x) = {sp.latex(simplified_expr)}$, if it exists.")

    st.subheader("📈 Graph")
    x_vals_full = np.linspace(user_a - 4, user_a + 4, 400)
    x_vals = x_vals_full[np.abs(x_vals_full - user_a) > 1e-6]
    try:
        y_vals = f(x_vals)
    except:
        st.error("Error evaluating function for plotting.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x_vals.tolist(), y=[0]*len(x_vals), z=y_vals.tolist(), mode='lines', name='f(x)', line=dict(color='blue')))

    # Add hole in red if removable discontinuity
    try:
        hole_y_val = float(simplified_expr.subs(x, user_a))
        fig.add_trace(go.Scatter3d(x=[float(user_a)], y=[0], z=[hole_y_val], mode='markers',
                                   marker=dict(size=6, color='red', symbol='circle-open'),
                                   name=f"Hole at x = {user_a}"))
    except:
        pass

    # Animated tangent line (approximation by shifting a small amount)
    try:
        tangent_frames = []
        for shift in np.linspace(-1, 1, 20):
            a_val = user_a + shift * 0.01
            m = float(derivative_expr.subs(x, a_val))
            b = float(simplified_expr.subs(x, a_val) - m * a_val)
            tangent_x = np.linspace(a_val - 1, a_val + 1, 100)
            tangent_y = m * tangent_x + b
            tangent_frames.append(go.Scatter3d(
                x=tangent_x.tolist(), y=[0]*len(tangent_x), z=tangent_y.tolist(),
                mode='lines', line=dict(color='orange', dash='dash'), name='Tangent'))

        # Add last tangent line
        fig.add_trace(tangent_frames[-1])
    except:
        pass

    fig.update_layout(title=dict(text=f"Graph of f(x)", x=0.5),
                      scene=dict(
                          xaxis_title='x',
                          yaxis_title='depth (for visual separation)',
                          zaxis_title='f(x)',
                          camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
                      ),
                      showlegend=True,
                      height=600,
                      margin=dict(l=0, r=0, b=0, t=30))

    try:
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying Plotly chart: {e}")

    # Table of values around a
    st.subheader(f"Limit Table Around x = {user_a} ⟲")
    delta_list = [0.1, 0.01, 0.001]
    x_input = [round(user_a - d, 6) for d in delta_list[::-1]] + [round(user_a + d, 6) for d in delta_list]
    table_data = []
    for xi in x_input:
        try:
            table_data.append((xi, round(f(xi), 6)))
        except:
            table_data.append((xi, "undefined"))

    st.table({"x": [r[0] for r in table_data], "f(x)": [r[1] for r in table_data]})

    st.subheader("🎯 Challenge: Estimate the Limit")
    user_limit = st.number_input(f"What do you think is the limit of f(x) as x approaches {user_a}?", step=0.01)
    try:
        actual_limit = round(float(simplified_expr.subs(x, user_a)), 6)
        if st.button("Check Answer"):
            if abs(user_limit - actual_limit) < 1e-3:
                st.success(f"✅ Correct! The limit is {actual_limit}.")
            else:
                st.error(f"❌ Not quite. The limit appears to be {actual_limit}.")
    except:
        st.warning("Limit may not exist or is not numerically evaluable.")

    st.subheader("🧠 Reflection")
    feedback = st.text_area("What did you learn about limits today?")
    if feedback:
        st.info("Thanks for sharing your reflection! 💬")



