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
    st.header("🔔 Sampling Distribution of the Sample Mean Calculator")

    mean = st.number_input("Enter the population mean (μ)", value=0.0, format="%.4f")
    sd_expr = st.text_input("Enter the population standard deviation (σ)", value="1")
    n = st.number_input("Enter the sample size (n)", min_value=1, step=1, value=30)
    decimal = st.number_input("Decimal places for output", min_value=0, max_value=10, value=4, step=1)

    sd = parse_stdev(sd_expr)
    if sd is None:
        return

    # Calculate sample mean distribution parameters
    sample_mean = mean
    sample_sd = sd / math.sqrt(n)

    st.markdown("---")
    st.write(f"**Sample Mean Mean (μₓ̄):** {sample_mean:.{decimal}f}")
    st.write(f"**Sample Mean Standard Deviation (Standard Error):** {sample_sd:.{decimal}f}")

    st.write("Choose a calculation:")
    option = st.radio("", [
        "1. Probability to the left of x (P(X̄ < x))",
        "2. Probability to the right of x (P(X̄ > x))",
        "3. Probability between two values (P(a < X̄ < b))",
        "4. Inverse: Find x for a given cumulative probability"
    ])

    x_val = None
    a = None
    b = None
    p = None
    if option == "1. Probability to the left of x (P(X̄ < x))":
        x_val = st.number_input("Enter the value of x̄", value=sample_mean)
    elif option == "2. Probability to the right of x (P(X̄ > x))":
        x_val = st.number_input("Enter the value of x̄", value=sample_mean)
    elif option == "3. Probability between two values (P(a < X̄ < b))":
        a = st.number_input("Enter the lower bound a", value=sample_mean - sample_sd)
        b = st.number_input("Enter the upper bound b", value=sample_mean + sample_sd)
        if b < a:
            st.error("Upper bound b must be greater than or equal to lower bound a.")
            return
    elif option == "4. Inverse: Find x for a given cumulative probability":
        p = st.number_input("Enter the cumulative probability (between 0 and 1)", min_value=0.0, max_value=1.0, value=0.5)

    if st.button("📊 Calculate"):
        x = np.linspace(sample_mean - 4 * sample_sd, sample_mean + 4 * sample_sd, 1000)
        y = stats.norm.pdf(x, sample_mean, sample_sd)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, label="Sampling Distribution of the Sample Mean", color='blue')

        def round_prob(prob_val):
            return round(prob_val, decimal)

        if option == "1. Probability to the left of x (P(X̄ < x))":
            prob = stats.norm.cdf(x_val, sample_mean, sample_sd)
            st.success(f"P(X̄ < {x_val}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x <= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        elif option == "2. Probability to the right of x (P(X̄ > x))":
            prob = 1 - stats.norm.cdf(x_val, sample_mean, sample_sd)
            st.success(f"P(X̄ > {x_val}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x >= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        elif option == "3. Probability between two values (P(a < X̄ < b))":
            prob = stats.norm.cdf(b, sample_mean, sample_sd) - stats.norm.cdf(a, sample_mean, sample_sd)
            st.success(f"P({a} < X̄ < {b}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), color='lightblue')
            ax.axvline(a, color='red', linestyle='--')
            ax.axvline(b, color='red', linestyle='--')

        elif option == "4. Inverse: Find x for a given cumulative probability":
            x_val = stats.norm.ppf(p, sample_mean, sample_sd)
            st.success(f"x such that P(X̄ < x) = {p} is {round_prob(x_val)}")
            ax.fill_between(x, 0, y, where=(x <= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        ax.set_xlabel("Sample Mean (X̄)")
        ax.set_ylabel("Density")
        ax.set_title("Sampling Distribution of the Sample Mean")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

if __name__ == "__main__":
    run()
