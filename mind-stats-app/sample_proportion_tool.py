import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

def run():
    st.header("🔔 Sampling Distribution of the Sample Proportion Calculator")

    p = st.number_input("Enter the population proportion (p)", min_value=0.0, max_value=1.0, value=0.5, format="%.4f")
    n = st.number_input("Enter the sample size (n)", min_value=1, step=1, value=30)
    decimal = st.number_input("Decimal places for output", min_value=0, max_value=10, value=4, step=1)

    if n <= 0:
        st.error("Sample size n must be greater than 0.")
        return

    # Calculate sample proportion mean and standard deviation (standard error)
    sample_mean = p
    sample_sd = math.sqrt(p * (1 - p) / n)

    st.markdown("---")
    st.write(f"**Sample Proportion Mean (p̂):** {sample_mean:.{decimal}f}")
    st.write(f"**Sample Proportion Standard Deviation (Standard Error):** {sample_sd:.{decimal}f}")

    st.write("Choose a probability calculation:")
    option = st.radio("", [
        "1. Probability to the left of x (P(p̂ < x))",
        "2. Probability to the right of x (P(p̂ > x))",
        "3. Probability between two values (P(a < p̂ < b))"
    ])

    x_val = None
    a = None
    b = None
    if option == "1. Probability to the left of x (P(p̂ < x))":
        x_val = st.number_input("Enter the value of p̂", min_value=0.0, max_value=1.0, value=sample_mean, format="%.4f")
    elif option == "2. Probability to the right of x (P(p̂ > x))":
        x_val = st.number_input("Enter the value of p̂", min_value=0.0, max_value=1.0, value=sample_mean, format="%.4f")
    elif option == "3. Probability between two values (P(a < p̂ < b))":
        a = st.number_input("Enter the lower bound a", min_value=0.0, max_value=1.0, value=max(0.0, sample_mean - sample_sd), format="%.4f")
        b = st.number_input("Enter the upper bound b", min_value=0.0, max_value=1.0, value=min(1.0, sample_mean + sample_sd), format="%.4f")
        if b < a:
            st.error("Upper bound b must be greater than or equal to lower bound a.")
            return

    if st.button("📊 Calculate Probability"):
        x = np.linspace(max(0, sample_mean - 4 * sample_sd), min(1, sample_mean + 4 * sample_sd), 1000)
        y = stats.norm.pdf(x, sample_mean, sample_sd)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, label="Sampling Distribution of the Sample Proportion", color='blue')

        def round_prob(p_val):
            return round(p_val, decimal)

        if option == "1. Probability to the left of x (P(p̂ < x))":
            prob = stats.norm.cdf(x_val, sample_mean, sample_sd)
            st.success(f"P(p̂ < {x_val}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x <= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        elif option == "2. Probability to the right of x (P(p̂ > x))":
            prob = 1 - stats.norm.cdf(x_val, sample_mean, sample_sd)
            st.success(f"P(p̂ > {x_val}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x >= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        elif option == "3. Probability between two values (P(a < p̂ < b))":
            prob = stats.norm.cdf(b, sample_mean, sample_sd) - stats.norm.cdf(a, sample_mean, sample_sd)
            st.success(f"P({a} < p̂ < {b}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), color='lightblue')
            ax.axvline(a, color='red', linestyle='--')
            ax.axvline(b, color='red', linestyle='--')

        ax.set_xlabel("Sample Proportion (p̂)")
        ax.set_ylabel("Density")
        ax.set_title("Sampling Distribution of the Sample Proportion")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

if __name__ == "__main__":
    run()
