import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp
from sympy.abc import x
import streamlit.components.v1 as components
import random


def run():
    st.header("♾️ Limits Visualizer")
    st.markdown("""
    Explore removable discontinuities, limits from a table, animation, symbolic simplification, tangent lines, and ε–δ reasoning.
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
    st.latex(f"f(x) = {sp.latex(fx_expr)}")
    st.markdown(f"The simplified expression is: $f(x) = {sp.latex(simplified_expr)}$, if it exists.")

    st.subheader("📈 Graph & Tangent Line")
    x_vals_full = np.linspace(user_a - 4, user_a + 4, 400)
    x_vals = x_vals_full[np.abs(x_vals_full - user_a) > 1e-6]
    try:
        y_vals = f(x_vals)
        y_hole = f(user_a - 1e-5)  # approximate value for hole
    except:
        st.error("Error evaluating function for plotting.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(min(x_vals_full), max(x_vals_full))
    ax.set_ylim(np.nanmin(y_vals) - 1, np.nanmax(y_vals) + 1)
    ax.set_title(rf"Graph of $f(x) = {sp.latex(fx_expr)}$")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    line, = ax.plot([], [], lw=2, label='f(x)')
    hole, = ax.plot([], [], 'o', color='red', markerfacecolor='white', markersize=8, label=f"Hole at x = {user_a}")

    # Optional tangent line
    try:
        m = float(derivative_expr.subs(x, user_a))
        b = float(simplified_expr.subs(x, user_a) - m * user_a)
        tangent_x = np.linspace(user_a - 2, user_a + 2, 100)
        tangent_y = m * tangent_x + b
        ax.plot(tangent_x, tangent_y, '--', label="Tangent line", color='orange')
    except:
        pass

    def init():
        line.set_data([], [])
        hole.set_data([], [])
        return line, hole

    def animate(i):
        x_draw = x_vals[:i]
        y_draw = y_vals[:i]
        line.set_data(x_draw, y_draw)
        if i > len(x_vals) // 2:
            hole.set_data([user_a], [y_hole])
        return line, hole

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x_vals), interval=10, blit=True)
    components.html(ani.to_jshtml(), height=500)

    # Table of values around a
    st.subheader(f"Limit Table Around x = {user_a}")
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
