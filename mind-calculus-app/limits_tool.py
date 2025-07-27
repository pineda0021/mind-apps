import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp
from sympy.abc import x
import streamlit.components.v1 as components
import random

# Define the function with a removable discontinuity
def f(x_val):
    return (x_val**2 - 5*x_val + 6) / (x_val - 2)

def run():
    st.header("♾️ Limits Visualizer")
    st.markdown("""
    Explore removable discontinuities, limits from a table, animation, symbolic simplification, and ε–δ reasoning.
    """)

    # Symbolic simplification
    st.subheader("\U0001F9EE Symbolic Simplification")
    numerator = sp.expand((x - 2)*(x - 3))
    original_expr = (x**2 - 5*x + 6)/(x - 2)
    simplified_expr = sp.simplify(original_expr)

    st.latex(r"\frac{x^2 - 5x + 6}{x - 2} = \frac{(x - 2)(x - 3)}{x - 2} = x - 3, \text{ for } x \ne 2")
    st.markdown(f"The simplified expression is: $f(x) = {sp.latex(simplified_expr)}$ for $x \ne 2$.")

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

    # Interactive Challenge
    st.subheader("\U0001F3AF Challenge: Estimate the Limit")
    user_limit = st.number_input("What do you think is the limit of f(x) as x approaches 2?", step=0.01)
    if st.button("Check Answer"):
        if abs(user_limit + 1) < 1e-3:
            st.success("✅ Correct! The limit is -1.")
        else:
            st.error("❌ Not quite. Try looking at the animation and the table again.")

    # One-Sided Limits Challenge
    st.subheader("🔍 One-Sided Limits Challenge")
    left_limit = st.number_input("Limit as x approaches 2 from the left (x → 2⁻):", key="left")
    right_limit = st.number_input("Limit as x approaches 2 from the right (x → 2⁺):", key="right")
    if st.button("Check One-Sided Limits"):
        correct = abs(left_limit + 1) < 1e-3 and abs(right_limit + 1) < 1e-3
        if correct:
            st.success("✅ Both one-sided limits are correct! So the two-sided limit exists and equals -1.")
        else:
            st.warning("⚠️ One or both one-sided limits are incorrect. Remember to read values closely from the table or animation.")

    # ε–δ Definition Challenge
    st.subheader("📐 ε–δ Definition Reasoning")
    st.markdown(r"""
    If \( \epsilon = 0.1 \), can you find a \( \delta \) such that whenever \( 0 < |x - 2| < \delta \), then \( |f(x) + 1| < \epsilon \)?
    """)
    user_delta = st.number_input("Your choice of δ:", step=0.001, format="%0.3f", key="delta")
    if st.button("Check ε–δ Condition"):
        delta_valid = True
        for test_x in [2 - user_delta / 2, 2 + user_delta / 2]:
            if abs(f(test_x) + 1) >= 0.1:
                delta_valid = False
                break
        if delta_valid:
            st.success("✅ Great! That δ works for ε = 0.1.")
        else:
            st.error("❌ That δ does not satisfy the ε–δ condition. Try a smaller δ.")

    # ε–δ Band Visualization
    st.subheader("📊 ε–δ Graphical Representation")
    epsilon = 0.1
    delta = user_delta
    xx = np.linspace(2 - delta * 1.5, 2 + delta * 1.5, 400)
    yy = f(xx)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(xx, yy, label="f(x)", color="blue")
    ax2.axhline(-1 + epsilon, color="green", linestyle="--", label=r"y = -1 ± ε")
    ax2.axhline(-1 - epsilon, color="green", linestyle="--")
    ax2.axvline(2 - delta, color="red", linestyle=":", label=r"x = 2 ± δ")
    ax2.axvline(2 + delta, color="red", linestyle=":")
    ax2.scatter([2], [-1], color='black', zorder=5)
    ax2.set_title("ε–δ Visualization")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # Multiple Choice Quiz
    st.subheader("📝 Quick Quiz")
    question = "What is the value of the limit as x approaches 2 for f(x)?"
    options = ["-2", "0", "1", "-1"]
    random.shuffle(options)
    answer = st.radio(question, options)
    if st.button("Submit Answer"):
        if answer == "-1":
            st.success("✅ Correct! f(x) approaches -1 as x approaches 2.")
        else:
            st.error("❌ Not quite. Review the animation and table above.")

    # Reflection Box
    st.subheader("🧠 Reflection")
    feedback = st.text_area("What did you learn about limits today?")
    if feedback:
        st.info("Thanks for sharing your reflection! 💬")
