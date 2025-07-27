import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp
from sympy.abc import x
import streamlit.components.v1 as components

# Define the function with a removable discontinuity
def f(x_val):
    return (x_val**2 - 5*x_val + 6) / (x_val - 2)

def run():
    st.header("Limits Visualizer")
    st.markdown("""
    Explore removable discontinuities, limits from a table, and animation.
    """)

    # Animation of the function with a removable discontinuity
    x_vals_full = np.linspace(-2, 6, 400)
    x_vals = x_vals_full[np.abs(x_vals_full - 2) > 1e-9]
    y_vals = f(x_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-2, 6)
    ax.set_ylim(-6, 4)
    ax.set_title(r"Graph of $f(x) = \frac{x^2 - 5x + 6}{x - 2}$")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    line, = ax.plot([], [], lw=2, label='f(x)')
    x_hole = 2
    y_hole = x_hole - 3
    hole, = ax.plot([], [], 'o', color='red', markerfacecolor='white', markersize=8, label='Hole at x = 2')

    def init():
        line.set_data([], [])
        hole.set_data([], [])
        return line, hole

    def animate(i):
        x = x_vals[:i]
        y = y_vals[:i]
        line.set_data(x, y)
        if i > len(x_vals) // 2:
            hole.set_data([x_hole], [y_hole])
        return line, hole

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x_vals), interval=10, blit=True)
    components.html(ani.to_jshtml(), height=500)

    # Table of values around x = 2
    st.subheader("Limit Table Around x = 2")
    x_input = [1.9, 1.99, 1.999, 2.001, 2.01, 2.1]
    table_data = []
    for xi in x_input:
        if xi == 2:
            table_data.append((xi, "undefined"))
        else:
            table_data.append((xi, round(f(xi), 6)))

    st.table({"x": [r[0] for r in table_data], "f(x)": [r[1] for r in table_data]})

    st.markdown("""
    From both sides, the function approaches \( f(x) \to -1 \) as \( x \to 2 \).
    Therefore, \( \lim_{x \to 2} f(x) = -1 \), even though \( f(2) \) is undefined.
    """)
