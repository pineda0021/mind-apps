import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# MIND: Continuous Probability Distributions (Uniform Only)
# ==========================================================
def run():
    st.header("📘 Continuous Probability Distributions")

    # Dropdown with only one option
    choice = st.selectbox(
        "Choose a distribution:",
        ["Uniform Distribution"],
        index=0
    )

    if choice == "Uniform Distribution":
        st.subheader("📏 Uniform Distribution Calculator")

        st.markdown("""
        The **Uniform Distribution** models continuous data where all values within an interval \([a, b]\)
        are equally likely.  
        """)

        st.latex(r"f(x) = \frac{1}{b - a}, \quad a \le x \le b")

        st.markdown("The cumulative probability satisfies:")
        st.latex(r"P(X < x) = P(X \le x)")

        st.markdown("and the expected value and variance are:")
        st.latex(r"E[X] = \frac{a + b}{2}, \quad Var[X] = \frac{(b - a)^2}{12}")

        st.markdown("---")

        # ---------- Inputs ----------
        a = st.number_input("Enter the minimum value (a):", value=0.0, key="a_min")
        b = st.number_input("Enter the maximum value (b):", value=10.0, key="b_max")
        if b <= a:
            st.error("⚠️ The upper bound (b) must be greater than the lower bound (a).")
            return

        decimal = st.number_input("Decimal places for output:", min_value=0, max_value=10, value=4, step=1, key="decimal")
        rp = lambda v: round(v, decimal)

        # ---------- Basic Calculations ----------
        mean = (a + b) / 2
        variance = ((b - a) ** 2) / 12
        pdf_value = 1 / (b - a)

        st.write(f"**Mean (E[X]) =** {rp(mean)}  |  **Variance (Var[X]) =** {rp(variance)}  |  **f(x) =** {rp(pdf_value)}")

        st.markdown("---")

        # ---------- Selection ----------
        calc_type = st.selectbox(
            "Choose a probability calculation:",
            ["P(X ≤ x)", "P(X ≥ x)", "P(a < X < b)", "Find x for a given probability"],
            index=0,
            key="calc_type"
        )

        show_steps = st.checkbox("📖 Show Step-by-Step Solution", key="steps")

        # ---------- Plot Base ----------
        x = np.linspace(a - (b - a) * 0.2, b + (b - a) * 0.2, 500)
        y = np.where((x >= a) & (x <= b), pdf_value, 0)

        # ==========================================================
        # CASE 1: P(X ≤ x)
        # ==========================================================
        if calc_type == "P(X ≤ x)":
            x_val = st.number_input("Enter x value:", value=a + (b - a) / 2, key="x_le")
            if st.button("Calculate", key="calc_le"):
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

        # ==========================================================
        # CASE 2: P(X ≥ x)
        # ==========================================================
        elif calc_type == "P(X ≥ x)":
            x_val = st.number_input("Enter x value:", value=a + (b - a) / 2, key="x_ge")
            if st.button("Calculate", key="calc_ge"):
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

        # ==========================================================
        # CASE 3: P(a < X < b)
        # ==========================================================
        elif calc_type == "P(a < X < b)":
            a1 = st.number_input("Lower bound (a₁):", value=a + (b - a) * 0.25, key="a1")
            b1 = st.number_input("Upper bound (b₁):", value=a + (b - a) * 0.75, key="b1")
            if st.button("Calculate", key="calc_between"):
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

        # ==========================================================
        # CASE 4: Inverse - Find x for given probability
        # ==========================================================
        elif calc_type == "Find x for a given probability":
            p = st.number_input("Enter probability (0 < p < 1):", min_value=0.0, max_value=1.0, value=0.5, key="p_val")
            direction = st.selectbox("Select tail:", ["Left tail: P(X ≤ x) = p", "Right tail: P(X ≥ x) = p"], key="direction")

            if st.button("Calculate", key="calc_inverse"):
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
# Run App
# ==========================================================
if __name__ == "__main__":
    run()
