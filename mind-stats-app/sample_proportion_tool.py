import streamlit as st
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

# ==========================================================
# Helper Function
# ==========================================================
def parse_expression(expr):
    expr = expr.strip().lower()
    try:
        if "sqrt" in expr:
            return math.sqrt(float(expr[5:-1]))
        elif "/" in expr:
            a, b = expr.split("/")
            return float(a) / float(b)
        else:
            return float(expr)
    except Exception:
        st.error("⚠️ Invalid input format. Use a number, a/b, or sqrt(x).")
        return None


# ==========================================================
# Normal Distribution
# ==========================================================
def normal_distribution(decimal):
    st.markdown("### 📈 **Normal Distribution**")
    st.latex(r"Z = \frac{X - \mu}{\sigma}")

    st.write("The normal distribution models continuous data that follows a bell-shaped curve, such as IQ or height.")

    mean = st.number_input("Population mean (μ):", value=0.0)
    sd = st.number_input("Standard deviation (σ):", min_value=0.0001, value=1.0)
    x_val = st.number_input("Enter value of X:", value=0.0)
    z = (x_val - mean) / sd
    st.success(f"Z = ({x_val} − {mean}) / {sd} = {round(z, decimal)}")

    # Visualization
    x = np.linspace(mean - 4*sd, mean + 4*sd, 500)
    y = norm.pdf(x, mean, sd)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, y, lw=2, color="blue")
    ax.fill_between(x, 0, y, where=(x <= x_val), color="skyblue", alpha=0.6)
    ax.axvline(x_val, color="red", linestyle="--")
    ax.set_xlabel("X")
    ax.set_ylabel("Density")
    ax.set_title("Normal Distribution")
    st.pyplot(fig)


# ==========================================================
# Sampling Distribution of the Mean
# ==========================================================
def sampling_mean(decimal):
    st.markdown("### 📘 **Sampling Distribution of the Mean**")
    st.latex(r"\mu = \mu_{\bar{X}}")
    st.latex(r"Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}}")

    st.write("Used when analyzing sample means from a population — shows how the mean varies with sample size.")

    mu_expr = st.text_input("Population mean (μ):", value="0")
    sigma_expr = st.text_input("Population SD (σ):", value="1")
    n = st.number_input("Sample size (n):", min_value=1, step=1, value=30)

    mu = parse_expression(mu_expr)
    sigma = parse_expression(sigma_expr)
    if mu is None or sigma is None:
        return

    se = sigma / math.sqrt(n)
    st.write(f"**Standard Error (σₓ̄) =** {round(se, decimal)}")

    xbar = st.number_input("Enter sample mean (x̄):", value=mu)
    z = (xbar - mu) / se
    st.success(f"Z = ({xbar} − {mu}) / {round(se, decimal)} = {round(z, decimal)}")

    x = np.linspace(mu - 4*se, mu + 4*se, 500)
    y = norm.pdf(x, mu, se)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, y, lw=2, color="blue")
    ax.fill_between(x, 0, y, where=(x <= xbar), color="skyblue", alpha=0.6)
    ax.axvline(xbar, color="red", linestyle="--")
    ax.set_xlabel("x̄")
    ax.set_ylabel("Density")
    ax.set_title("Sampling Distribution of the Mean")
    st.pyplot(fig)


# ==========================================================
# MAIN APP
# ==========================================================
def run():
    st.header("🧠 MIND: Continuous Probability Distributions")

    dist_choice = st.radio(
        "Select Distribution Type:",
        ["Normal Distribution", "Sampling Distribution of the Mean"],
        horizontal=True
    )

    decimal = st.number_input("Decimal places for output:", min_value=0, max_value=10, value=4, step=1)

    if dist_choice == "Normal Distribution":
        normal_distribution(decimal)
    elif dist_choice == "Sampling Distribution of the Mean":
        sampling_mean(decimal)


# ==========================================================
# Run App
# ==========================================================
if __name__ == "__main__":
    run()
