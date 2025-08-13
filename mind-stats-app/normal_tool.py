import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

def parse_stdev(sd_expr):
    try:
        # Allow sqrt only
        return eval(sd_expr, {"__builtins__": None}, {"sqrt": math.sqrt})
    except Exception as e:
        st.error(f"Invalid standard deviation input: {e}")
        return None

def run():
    st.header("🛎 Normal Distribution Calculator")

    mean = st.number_input("Enter the mean (μ)", value=0.0, format="%.4f")
    sd_expr = st.text_input("Enter the standard deviation (σ)", value="1")
    decimal = st.number_input("Decimal places for output", min_value=0, max_value=10, value=4, step=1)

    sd = parse_stdev(sd_expr)
    if sd is None:
        return

    st.markdown("---")
    st.write("Choose a probability calculation:")
    option = st.radio("", [
        "1. Probability to the left of x (P(X < x))",
        "2. Probability to the right of x (P(X > x))",
        "3. Probability between two values (P(a < X < b))"
    ])

    x_val = None
    a = None
    b = None
    if option == "1. Probability to the left of x (P(X < x))":
        x_val = st.number_input("Enter the value of x", value=mean)
    elif option == "2. Probability to the right of x (P(X > x))":
        x_val = st.number_input("Enter the value of x", value=mean)
    elif option == "3. Probability between two values (P(a < X < b))":
        a = st.number_input("Enter the lower bound a", value=mean - sd)
        b = st.number_input("Enter the upper bound b", value=mean + sd)
        if b < a:
            st.error("Upper bound b must be greater than or equal to lower bound a.")
            return

    if st.button("📊 Calculate Probability"):
        x = np.linspace(mean - 4 * sd, mean + 4 * sd, 1000)
        y = stats.norm.pdf(x, mean, sd)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, label="Normal Distribution", color='blue')

        def round_prob(p):
            return round(p, decimal)

        if option == "1. Probability to the left of x (P(X < x))":
            prob = stats.norm.cdf(x_val, mean, sd)
            st.success(f"P(X < {x_val}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x <= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        elif option == "2. Probability to the right of x (P(X > x))":
            prob = 1 - stats.norm.cdf(x_val, mean, sd)
            st.success(f"P(X > {x_val}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x >= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        elif option == "3. Probability between two values (P(a < X < b))":
            prob = stats.norm.cdf(b, mean, sd) - stats.norm.cdf(a, mean, sd)
            st.success(f"P({a} < X < {b}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), color='lightblue')
            ax.axvline(a, color='red', linestyle='--')
            ax.axvline(b, color='red', linestyle='--')

        ax.set_xlabel("X")
        ax.set_ylabel("Density")
        ax.set_title("Normal Distribution")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

if __name__ == "__main__":
    run()
