import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sympy as sp

def evaluate_expression(expr):
    """Evaluate a math expression input as a string (e.g., '1/2', '1 - 0.15')."""
    try:
        return float(sp.sympify(expr))
    except (sp.SympifyError, ValueError):
        st.error(f"Invalid expression: {expr}")
        return None

def plot_normal_distribution(mean, sd, result=None, bounds=None):
    """Plot the normal distribution curve with optional vertical lines."""
    x = np.linspace(mean - 4 * sd, mean + 4 * sd, 1000)
    y = stats.norm.pdf(x, mean, sd)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, label="Normal Distribution", color='blue')

    if result is not None:
        ax.axvline(result, color='red', linestyle='--', label=f'x = {result}')

    if bounds is not None:
        ax.axvline(bounds[0], color='red', linestyle='--', label=f'x1 = {bounds[0]}')
        ax.axvline(bounds[1], color='green', linestyle='--', label=f'x2 = {bounds[1]}')

    ax.set_xlabel("X")
    ax.set_ylabel("Density")
    ax.set_title("Normal Distribution")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

def run():
    st.header("🔔 Inverse Normal Distribution Calculator")

    mean = st.number_input("Enter the mean (μ)", value=0.0, format="%.4f")
    sd_expr = st.text_input("Enter the standard deviation (σ)", value="1")
    decimal = st.number_input("Decimal places for output", min_value=0, max_value=10, value=4, step=1)

    sd = evaluate_expression(sd_expr)
    if sd is None:
        return

    st.markdown("---")
    st.write("Choose calculation option:")
    option = st.radio("", [
        "1. Find x for a given left probability (P(X < x))",
        "2. Find x for a given right probability (P(X > x))",
        "3. Find x values for given cumulative probability range (P(x1 < X < x2))"
    ])

    if option == "1. Find x for a given left probability (P(X < x))":
        p_str = st.text_input("Enter the left probability (p)", value="0.5")
        p = evaluate_expression(p_str)
        if p is None or not (0 <= p <= 1):
            st.error("Probability must be between 0 and 1.")
            return

        if st.button("📊 Calculate"):
            result = round(stats.norm.ppf(p, mean, sd), decimal)
            st.success(f"X for P(X < x) = {p} is {result}")
            plot_normal_distribution(mean, sd, result=result)

    elif option == "2. Find x for a given right probability (P(X > x))":
        p_str = st.text_input("Enter the right probability (p)", value="0.5")
        p = evaluate_expression(p_str)
        if p is None or not (0 <= p <= 1):
            st.error("Probability must be between 0 and 1.")
            return

        if st.button("Calculate"):
            result = round(stats.norm.ppf(p, mean, sd), decimal)
            st.success(f"X for P(X > x) = {p} is {result}")
            plot_normal_distribution(mean, sd, result=result)

    elif option == "3. Find x values for given cumulative probability range (P(x1 < X < x2))":
        a_str = st.text_input("Enter the lower cumulative probability (a)", value="0.25")
        b_str = st.text_input("Enter the upper cumulative probability (b)", value="0.75")

        a = evaluate_expression(a_str)
        b = evaluate_expression(b_str)

        if None in (a, b) or not (0 <= a < b <= 1):
            st.error("Ensure 0 ≤ a < b ≤ 1 for cumulative probabilities.")
            return

        if st.button("Calculate"):
            lower = round(stats.norm.ppf(a, mean, sd), decimal)
            upper = round(stats.norm.ppf(b, mean, sd), decimal)
            st.success(f"X values for cumulative range {a} to {b} are {lower} and {upper}")
            st.write(f"P({lower} < X < {upper}) = {round(b - a, decimal)}")
            plot_normal_distribution(mean, sd, bounds=(lower, upper))

if __name__ == "__main__":
    run()
