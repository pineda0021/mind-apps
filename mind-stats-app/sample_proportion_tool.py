import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

def run():
    st.header("🔔 Sampling Distribution of the Sample Proportion Calculator")

    p_hat = st.number_input("Enter the population proportion (p)", min_value=0.0, max_value=1.0, value=0.5, format="%.4f")
    n = st.number_input("Enter the sample size (n)", min_value=1, step=1, value=30)
    decimal = st.number_input("Decimal places for output", min_value=0, max_value=10, value=4, step=1)

    # Calculate sample proportion distribution parameters
    sample_mean = p_hat
    # Standard error for proportion = sqrt(p(1-p)/n)
    sample_se = math.sqrt(p_hat * (1 - p_hat) / n)

    st.markdown("---")
    st.write(f"**Sample Proportion Mean (p̂):** {sample_mean:.{decimal}f}")
    st.write(f"**Sample Proportion Standard Error:** {sample_se:.{decimal}f}")

    st.write("Choose a calculation:")
    option = st.radio("", [
        "1. Probability to the left of x (P(p̂ < x))",
        "2. Probability to the right of x (P(p̂ > x))",
        "3. Probability between two values (P(a < p̂ < b))",
        "4. Inverse left: Find x for given P(p̂ < x)",
        "5. Inverse right: Find x for given P(p̂ > x)",
        "6. Inverse between: Find x1 and x2 for given P(x1 < p̂ < x2)"
    ])

    x_val = None
    a = None
    b = None
    p = None
    p_lower = None
    p_upper = None

    if option == "1. Probability to the left of x (P(p̂ < x))":
        x_val = st.number_input("Enter the value of p̂", min_value=0.0, max_value=1.0, value=sample_mean, format="%.4f")
    elif option == "2. Probability to the right of x (P(p̂ > x))":
        x_val = st.number_input("Enter the value of p̂", min_value=0.0, max_value=1.0, value=sample_mean, format="%.4f")
    elif option == "3. Probability between two values (P(a < p̂ < b))":
        a = st.number_input("Enter the lower bound a", min_value=0.0, max_value=1.0, value=max(0.0, sample_mean - sample_se), format="%.4f")
        b = st.number_input("Enter the upper bound b", min_value=0.0, max_value=1.0, value=min(1.0, sample_mean + sample_se), format="%.4f")
        if b < a:
            st.error("Upper bound b must be greater than or equal to lower bound a.")
            return
    elif option == "4. Inverse left: Find x for given P(p̂ < x)":
        p = st.number_input("Enter cumulative probability p (0 < p < 1)", min_value=0.0, max_value=1.0, value=0.5)
    elif option == "5. Inverse right: Find x for given P(p̂ > x)":
        p = st.number_input("Enter cumulative probability p (0 < p < 1)", min_value=0.0, max_value=1.0, value=0.5)
    elif option == "6. Inverse between: Find x1 and x2 for given P(x1 < p̂ < x2)":
        p_lower = st.number_input("Enter lower cumulative probability (0 < p_lower < 1)", min_value=0.0, max_value=1.0, value=0.2)
        p_upper = st.number_input("Enter upper cumulative probability (p_lower < p_upper < 1)", min_value=0.0, max_value=1.0, value=0.8)
        if p_upper <= p_lower:
            st.error("Upper probability must be greater than lower probability.")
            return

    if st.button("📊 Calculate"):
        x = np.linspace(0, 1, 1000)
        y = stats.norm.pdf(x, sample_mean, sample_se)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, label="Sampling Distribution of the Sample Proportion", color='blue')

        def round_prob(prob_val):
            return round(prob_val, decimal)

        if option == "1. Probability to the left of x (P(p̂ < x))":
            prob = stats.norm.cdf(x_val, sample_mean, sample_se)
            st.success(f"P(p̂ < {x_val}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x <= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        elif option == "2. Probability to the right of x (P(p̂ > x))":
            prob = 1 - stats.norm.cdf(x_val, sample_mean, sample_se)
            st.success(f"P(p̂ > {x_val}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x >= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        elif option == "3. Probability between two values (P(a < p̂ < b))":
            prob = stats.norm.cdf(b, sample_mean, sample_se) - stats.norm.cdf(a, sample_mean, sample_se)
            st.success(f"P({a} < p̂ < {b}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), color='lightblue')
            ax.axvline(a, color='red', linestyle='--')
            ax.axvline(b, color='red', linestyle='--')

        elif option == "4. Inverse left: Find x for given P(p̂ < x)":
            x_val = stats.norm.ppf(p, sample_mean, sample_se)
            st.success(f"x such that P(p̂ < x) = {p} is {round_prob(x_val)}")
            ax.fill_between(x, 0, y, where=(x <= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        elif option == "5. Inverse right: Find x for given P(p̂ > x)":
            x_val = stats.norm.ppf(1 - p, sample_mean, sample_se)
            st.success(f"x such that P(p̂ > x) = {p} is {round_prob(x_val)}")
            ax.fill_between(x, 0, y, where=(x >= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        elif option == "6. Inverse between: Find x1 and x2 for given P(x1 < p̂ < x2)":
            x1 = stats.norm.ppf(p_lower, sample_mean, sample_se)
            x2 = stats.norm.ppf(p_upper, sample_mean, sample_se)
            prob_between = p_upper - p_lower
            st.success(f"x1 such that P(p̂ < x1) = {p_lower} is {round_prob(x1)}")
            st.success(f"x2 such that P(p̂ < x2) = {p_upper} is {round_prob(x2)}")
            st.success(f"P({round_prob(x1)} < p̂ < {round_prob(x2)}) = {round_prob(prob_between)}")
            ax.fill_between(x, 0, y, where=(x >= x1) & (x <= x2), color='lightblue')
            ax.axvline(x1, color='red', linestyle='--')
            ax.axvline(x2, color='red', linestyle='--')

        ax.set_xlabel("Sample Proportion (p̂)")
        ax.set_ylabel("Density")
        ax.set_title("Sampling Distribution of the Sample Proportion")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

if __name__ == "__main__":
    run()
