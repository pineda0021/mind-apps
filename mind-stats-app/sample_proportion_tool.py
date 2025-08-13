import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

def run():
    st.header("🔔 Distribution of the Sample Proportion Calculator")

    pop_p = st.number_input("Enter the population proportion (p)", min_value=0.0, max_value=1.0, value=0.5, format="%.4f")
    sample_size = st.number_input("Enter the sample size (n)", min_value=1, step=1, value=30)
    decimal = st.number_input("Decimal places for output", min_value=0, max_value=10, value=4, step=1)

    # Compute sampling distribution parameters
    mean = pop_p
    sd = math.sqrt(pop_p * (1 - pop_p) / sample_size)

    st.markdown("---")
    st.write("Choose a calculation:")
    option = st.radio("",
        [
            "1. Probability to the left of x (P(𝑝̂ < x))",
            "2. Probability to the right of x (P(𝑝̂ > x))",
            "3. Probability between two values (P(a < 𝑝̂ < b))",
            "4. Inverse: Find x for a given cumulative probability (P(𝑝̂ < x) = p)"
        ]
    )

    x_val = None
    a = None
    b = None
    p = None

    if option.startswith("1"):
        x_val = st.number_input("Enter the value of x", min_value=0.0, max_value=1.0, value=mean, format="%.4f")
    elif option.startswith("2"):
        x_val = st.number_input("Enter the value of x", min_value=0.0, max_value=1.0, value=mean, format="%.4f")
    elif option.startswith("3"):
        a = st.number_input("Enter the lower bound a", min_value=0.0, max_value=1.0, value=max(0.0, mean - sd), format="%.4f")
        b = st.number_input("Enter the upper bound b", min_value=0.0, max_value=1.0, value=min(1.0, mean + sd), format="%.4f")
        if b < a:
            st.error("Upper bound b must be greater than or equal to lower bound a.")
            return
    elif option.startswith("4"):
        p = st.number_input("Enter the cumulative probability p (0 to 1)", min_value=0.0, max_value=1.0, value=0.5, format="%.4f")

    if st.button("📊 Calculate"):
        x = np.linspace(0, 1, 1000)
        y = stats.norm.pdf(x, mean, sd)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, label="Sampling Distribution of Sample Proportion", color='blue')

        def round_prob(val):
            return round(val, decimal)

        if option.startswith("1"):
            prob = stats.norm.cdf(x_val, mean, sd)
            st.success(f"P(𝑝̂ < {x_val}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x <= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        elif option.startswith("2"):
            prob = 1 - stats.norm.cdf(x_val, mean, sd)
            st.success(f"P(𝑝̂ > {x_val}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x >= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        elif option.startswith("3"):
            prob = stats.norm.cdf(b, mean, sd) - stats.norm.cdf(a, mean, sd)
            st.success(f"P({a} < 𝑝̂ < {b}) = {round_prob(prob)}")
            ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), color='lightblue')
            ax.axvline(a, color='red', linestyle='--')
            ax.axvline(b, color='red', linestyle='--')

        elif option.startswith("4"):
            x_val_inv = stats.norm.ppf(p, mean, sd)
            # Clip inverse x between 0 and 1 (proportion range)
            x_val_inv = min(max(x_val_inv, 0.0), 1.0)
            st.success(f"x such that P(𝑝̂ < x) = {p} is x = {round_prob(x_val_inv)}")
            ax.fill_between(x, 0, y, where=(x <= x_val_inv), color='lightblue')
            ax.axvline(x_val_inv, color='red', linestyle='--')

        ax.set_xlabel("Sample Proportion (𝑝̂)")
        ax.set_ylabel("Density")
        ax.set_title(f"Sampling Distribution of Sample Proportion (n={sample_size})")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

if __name__ == "__main__":
    run()

