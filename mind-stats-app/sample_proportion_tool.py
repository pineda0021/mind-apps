import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

def parse_stdev(sd_expr):
    try:
        return eval(sd_expr, {"__builtins__": None}, {"sqrt": math.sqrt})
    except Exception as e:
        st.error(f"Invalid standard deviation input: {e}")
        return None

def run():
    st.header("🔔 Distribution of the Sample Mean Calculator")

    mean = st.number_input("Enter the population mean (μ)", value=0.0, format="%.4f")
    sd_expr = st.text_input("Enter the population standard deviation (σ)", value="1")
    sample_size = st.number_input("Enter the sample size (n)", min_value=1, step=1, value=30)
    decimal = st.number_input("Decimal places for output", min_value=0, max_value=10, value=4, step=1)

    sd = parse_stdev(sd_expr)
    if sd is None:
        return

    # Compute standard deviation of sample mean
    sample_mean_sd = sd / math.sqrt(sample_size)

    st.markdown("---")
    option = st.radio("Choose a calculation:", [
        ("left", "1. Probability to the left of x (P(X̄ < x))"),
        ("right", "2. Probability to the right of x (P(X̄ > x))"),
        ("between", "3. Probability between two values (P(a < X̄ < b))"),
        ("inverse", "4. Inverse: Find x for a given cumulative probability")
    ], format_func=lambda x: x[1])

    x_val = None
    a = None
    b = None
    p = None

    if option[0] == "left":
        x_val = st.number_input("Enter the value of x", value=mean)
    elif option[0] == "right":
        x_val = st.number_input("Enter the value of x", value=mean)
    elif option[0] == "between":
        a = st.number_input("Enter the lower bound a", value=mean - sample_mean_sd)
        b = st.number_input("Enter the upper bound b", value=mean + sample_mean_sd)
        if b < a:
            st.error("Upper bound b must be greater than or equal to lower bound a.")
            return
    elif option[0] == "inverse":
        p = st.number_input("Enter the cumulative probability (0 to 1)", min_value=0.0, max_value=1.0, value=0.5)

    if st.button("📊 Calculate Probability"):
        x = np.linspace(mean - 4 * sample_mean_sd, mean + 4 * sample_mean_sd, 1000)
        y = stats.norm.pdf(x, mean, sample_mean_sd)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, label="Distribution of Sample Mean", color='blue')

        def round_prob(val):
            return round(val, decimal)

        if option[0] == "left":
            prob = stats.norm.cdf(x_val, mean, sample_mean_sd)
            st.success(f"P(X̄ < {x_val}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x <= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        elif option[0] == "right":
            prob = 1 - stats.norm.cdf(x_val, mean, sample_mean_sd)
            st.success(f"P(X̄ > {x_val}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x >= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        elif option[0] == "between":
            prob = stats.norm.cdf(b, mean, sample_mean_sd) - stats.norm.cdf(a, mean, sample_mean_sd)
            st.success(f"P({a} < X̄ < {b}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), color='lightblue')
            ax.axvline(a, color='red', linestyle='--')
            ax.axvline(b, color='red', linestyle='--')

        elif option[0] == "inverse":
            x_val = stats.norm.ppf(p, mean, sample_mean_sd)
            st.success(f"x such that P(X̄ < x) = {p} is approximately {round_prob(x_val)}")
            ax.fill_between(x, 0, y, where=(x <= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        ax.set_xlabel("X̄")
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of the Sample Mean (n={sample_size})")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

if __name__ == "__main__":
    run()
