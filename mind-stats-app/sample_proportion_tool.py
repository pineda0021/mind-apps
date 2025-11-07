import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Uniform Distribution Calculator
# ==========================================================
def run():
    st.header("📏 Uniform Distribution Calculator")

    st.markdown("""
    The **Uniform Distribution** models continuous data that has equal probability over an interval \([a, b]\).

    The probability density function is given by:
    """)
    st.latex(r"f(x) = \frac{1}{b - a}, \quad a \le x \le b")

    st.markdown("and the cumulative probability satisfies:")
    st.latex(r"P(X < x) = P(X \le x)")

    st.markdown("---")

    # --- Inputs ---
    a = st.number_input("Enter the minimum value (a):", value=0.0)
    b = st.number_input("Enter the maximum value (b):", value=10.0)
    if b <= a:
        st.error("⚠️ The upper bound (b) must be greater than the lower bound (a).")
        return

    decimal = st.number_input("Decimal places for output:", min_value=0, max_value=10, value=4, step=1)
    rp = lambda v: round(v, decimal)

    calc_type = st.selectbox(
        "Choose a probability calculation:",
        ["P(X ≤ x)", "P(X ≥ x)", "P(a < X < b)", "Find x for a given probability"],
        index=0
    )

    show_steps = st.checkbox("📖 Show Step-by-Step Solution")

    # --- Derived quantities ---
    pdf_value = 1 / (b - a)
    st.write(f"**Constant PDF:** f(x) = 1/({b} - {a}) = **{rp(pdf_value)}**")

    # --- Calculations ---
    x = np.linspace(a - (b - a) * 0.2, b + (b - a) * 0.2, 500)
    y = np.where((x >= a) & (x <= b), pdf_value, 0)

    if calc_type == "P(X ≤ x)":
        x_val = st.number_input("Enter x value:", value=a + (b - a) / 2)
        if st.button("Calculate"):
            if x_val <= a:
                prob = 0.0
            elif x_val >= b:
                prob = 1.0
            else:
                prob = (x_val - a) / (b - a)
            st.success(f"P(X ≤ {x_val}) = {rp(prob)}")
            if show_steps:
                st.write(f"P(X ≤ x) = (x − a) / (b − a) = ({x_val} − {a}) / ({b} − {a}) = {rp(prob)}")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, y, color='blue', lw=2)
            ax.fill_between(x, 0, y, where=(x <= x_val) & (x >= a), color='skyblue', alpha=0.6)
            ax.axvline(x_val, color="red", linestyle="--")
            ax.set_title("Uniform Distribution")
            ax.set_xlabel("X")
            ax.set_ylabel("f(x)")
            st.pyplot(fig)

    elif calc_type == "P(X ≥ x)":
        x_val = st.number_input("Enter x value:", value=a + (b - a) / 2)
        if st.button("Calculate"):
            if x_val <= a:
                prob = 1.0
            elif x_val >= b:
                prob = 0.0
            else:
                prob = (b - x_val) / (b - a)
            st.success(f"P(X ≥ {x_val}) = {rp(prob)}")
            if show_steps:
                st.write(f"P(X ≥ x) = (b − x) / (b − a) = ({b} − {x_val}) / ({b} − {a}) = {rp(prob)}")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, y, color='blue', lw=2)
            ax.fill_between(x, 0, y, where=(x >= x_val) & (x <= b), color='lightgreen', alpha=0.6)
            ax.axvline(x_val, color="red", linestyle="--")
            ax.set_title("Uniform Distribution")
            ax.set_xlabel("X")
            ax.set_ylabel("f(x)")
            st.pyplot(fig)

    elif calc_type == "P(a < X < b)":
        a1 = st.number_input("Lower bound (a₁):", value=a + (b - a) * 0.25)
        b1 = st.number_input("Upper bound (b₁):", value=a + (b - a) * 0.75)
        if st.button("Calculate"):
            if a1 < a:
                a1 = a
            if b1 > b:
                b1 = b
            if a1 >= b1:
                st.error("⚠️ Lower bound must be less than upper bound.")
                return
            prob = (b1 - a1) / (b - a)
            st.success(f"P({a1} < X < {b1}) = {rp(prob)}")
            if show_steps:
                st.write(f"P(a < X < b) = (b₁ − a₁) / (b − a) = ({b1} − {a1}) / ({b} − {a}) = {rp(prob)}")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, y, color='blue', lw=2)
            ax.fill_between(x, 0, y, where=(x >= a1) & (x <= b1), color='orange', alpha=0.6)
            ax.axvline(a1, color="red", linestyle="--")
            ax.axvline(b1, color="red", linestyle="--")
            ax.set_title("Uniform Distribution")
            ax.set_xlabel("X")
            ax.set_ylabel("f(x)")
            st.pyplot(fig)

    elif calc_type == "Find x for a given probability":
        p = st.number_input("Enter probability (0 < p < 1):", min_value=0.0, max_value=1.0, value=0.5)
        direction = st.selectbox("Select tail:", ["Left tail: P(X ≤ x) = p", "Right tail: P(X ≥ x) = p"])
        if st.button("Calculate"):
            if direction == "Left tail: P(X ≤ x) = p":
                x_val = a + p * (b - a)
                st.success(f"x = {rp(x_val)} for P(X ≤ x) = {p}")
            else:
                x_val = b - p * (b - a)
                st.success(f"x = {rp(x_val)} for P(X ≥ x) = {p}")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, y, color='blue', lw=2)
            if direction == "Left tail: P(X ≤ x) = p":
                ax.fill_between(x, 0, y, where=(x <= x_val) & (x >= a), color='skyblue', alpha=0.6)
            else:
                ax.fill_between(x, 0, y, where=(x >= x_val) & (x <= b), color='lightgreen', alpha=0.6)
            ax.axvline(x_val, color="red", linestyle="--")
            ax.set_title("Uniform Distribution")
            ax.set_xlabel("X")
            ax.set_ylabel("f(x)")
            st.pyplot(fig)


# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":
    run()
