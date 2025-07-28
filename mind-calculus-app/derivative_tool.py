import streamlit as st
import sympy as sp
from sympy.abc import x
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def parse_input(expr_str):
    expr_str = expr_str.replace('^', '**').replace('sqrt', 'sp.sqrt')
    return sp.sympify(expr_str)

def step_by_step_derivation(expr):
    steps = []

    # Constant
    if expr.is_Number:
        steps.append("**Constant Rule:**  \n" + rf"$\frac{{d}}{{dx}}[{sp.latex(expr)}] = 0$")
        return steps

    # Sum Rule
    if expr.is_Add:
        steps.append("**Sum Rule:**")
        for term in expr.args:
            steps.extend(step_by_step_derivation(term))
        return steps

    # Product Rule
    if expr.is_Mul:
        u, v = expr.args
        u_diff = sp.diff(u, x)
        v_diff = sp.diff(v, x)
        steps.append("**Product Rule:**")
        steps.append(rf"$\frac{{d}}{{dx}}[{sp.latex(u)} \cdot {sp.latex(v)}] = {sp.latex(u)} \cdot \frac{{d}}{{dx}}[{sp.latex(v)}] + {sp.latex(v)} \cdot \frac{{d}}{{dx}}[{sp.latex(u)}]$")
        steps.append(rf"$= {sp.latex(u)} \cdot ({sp.latex(v_diff)}) + {sp.latex(v)} \cdot ({sp.latex(u_diff)})$")
        steps.append(rf"$= {sp.latex(sp.simplify(u * v_diff + v * u_diff))}$")
        return steps

    # Power Rule & Chain Rule
    if expr.is_Pow:
        base, exp = expr.args
        base_diff = sp.diff(base, x)
        if base.has(x):
            steps.append("**Chain Rule (Power):**")
            steps.append(rf"$\frac{{d}}{{dx}}[{sp.latex(base)}^{{{sp.latex(exp)}}}] = {sp.latex(exp)} \cdot {sp.latex(base)}^{{{sp.latex(exp - 1)}}} \cdot \frac{{d}}{{dx}}[{sp.latex(base)}]$")
            steps.append(rf"$= {sp.latex(exp)} \cdot {sp.latex(base**(exp - 1))} \cdot {sp.latex(base_diff)}$")
            steps.append(rf"$= {sp.latex(sp.simplify(exp * base**(exp - 1) * base_diff))}$")
        else:
            steps.append("**Power Rule (constant base):**")
            steps.append(rf"$\frac{{d}}{{dx}}[{sp.latex(expr)}] = 0$")
        return steps

    # Trig / Log / Exp
    if expr.func in [sp.sin, sp.cos, sp.tan, sp.exp, sp.log]:
        steps.append(f"**Rule for {expr.func.__name__}:**")
        steps.append(rf"$\frac{{d}}{{dx}}[{sp.latex(expr)}] = {sp.latex(sp.diff(expr, x))}$")
        return steps

    # Default fallback
    steps.append(rf"$\frac{{d}}{{dx}}[{sp.latex(expr)}] = {sp.latex(sp.diff(expr, x))}$")
    return steps

def run():
    st.header("𝒅𝒚/𝒅𝒙 Derivative Visualizer")
    st.markdown("""
    Enter a function and explore its derivative symbolically, graphically, and numerically.
    """)

    # Input
    st.subheader("📥 Enter a Function")
    f_input = st.text_input("f(x) =", "x^3 - 3x + 1")
    try:
        fx = parse_input(f_input)
        dfx = sp.diff(fx, x)
    except Exception as e:
        st.error(f"Invalid input: {e}")
        return

    # Symbolic Derivative
    st.subheader("🧠 Symbolic Derivative")
    st.latex(rf"f'(x) = \frac{{d}}{{dx}}[{sp.latex(fx)}] = {sp.latex(dfx)}")

    # Step-by-step
    st.subheader("🔎 Step-by-Step Derivation")
    for step in step_by_step_derivation(fx):
        st.markdown("- " + step)

    # 2D Plot: f(x) and f'(x)
    st.subheader("📈 Graph of $f(x)$ and $f'(x)$")
    f_np = sp.lambdify(x, fx, modules=["numpy"])
    df_np = sp.lambdify(x, dfx, modules=["numpy"])
    X = np.linspace(-5, 5, 400)
    Y = f_np(X)
    Y_prime = df_np(X)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X, Y, label="f(x)", color="blue")
    ax.plot(X, Y_prime, label="f'(x)", color="orange")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Plot")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # 3D ZOOM-IN Plot
    st.subheader("🔭 3D Zoomable Plot of $f(x)$ and $f'(x)$")
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(x=X, y=Y, z=Y_prime, mode='lines', name='f(x) vs f\'(x)',
                                 line=dict(color='royalblue')))
    fig3d.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='f(x)',
            zaxis_title="f'(x)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1)),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title="Zoom and Rotate to Explore!"
    )
    st.plotly_chart(fig3d)

    # Tangent line visualization
    st.subheader("📍 Tangent Line at a Point")
    a_val = st.slider("Choose a point x = a", -5.0, 5.0, value=1.0, step=0.1)
    f_a = f_np(a_val)
    df_a = df_np(a_val)
    tangent_line = lambda x_val: df_a * (x_val - a_val) + f_a
    Y_tangent = tangent_line(X)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(X, Y, label="f(x)", color="blue")
    ax2.plot(X, Y_tangent, label=f"Tangent at x = {a_val}", linestyle="--", color="red")
    ax2.scatter([a_val], [f_a], color="black", zorder=5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Tangent Line Visualization")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    # Derivative Table
    st.subheader("📊 Derivative Table: Definition")
    h_vals = [1, 0.5, 0.1, 0.01, 0.001]
    slopes = [(f_np(a_val + h) - f_np(a_val)) / h for h in h_vals]
    table_data = {
        "h": h_vals,
        "[f(a+h) - f(a)] / h": [round(s, 6) for s in slopes]
    }
    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table)
    st.latex(rf"\lim_{{h \to 0}} \frac{{f({a_val}+h)-f({a_val})}}{{h}} = {df_a:.6f}")

    # Reflection
    st.subheader("💭 Reflection")
    feedback = st.text_area("What did you learn about derivatives today?")
    if feedback:
        st.info("Thanks for sharing your reflection! 💬")

if __name__ == "__main__":
    run()
