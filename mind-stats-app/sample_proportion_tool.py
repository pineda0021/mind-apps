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

    calc_type = st.selectbox("Choose a calculation:", [
        "P(X < x)",
        "P(X > x)",
        "P(a < X < b)",
        "Inverse: Find x for given probability"
    ])

    x = np.linspace(mean - 4*sd, mean + 4*sd, 500)
    y = norm.pdf(x, mean, sd)

    if calc_type == "P(X < x)":
        x_val = st.number_input("Enter x value:", value=0.0)
        prob = norm.cdf(x_val, mean, sd)
        z = (x_val - mean) / sd
        st.success(f"P(X < {x_val}) = {round(prob, decimal)}")
        st.write(f"Z = ({x_val} − {mean}) / {sd} = {round(z, decimal)}")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x <= x_val), color="skyblue", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    elif calc_type == "P(X > x)":
        x_val = st.number_input("Enter x value:", value=0.0)
        prob = 1 - norm.cdf(x_val, mean, sd)
        z = (x_val - mean) / sd
        st.success(f"P(X > {x_val}) = {round(prob, decimal)}")
        st.write(f"Z = ({x_val} − {mean}) / {sd} = {round(z, decimal)}")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x >= x_val), color="lightgreen", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    elif calc_type == "P(a < X < b)":
        a = st.number_input("Lower bound (a):", value=mean - sd)
        b = st.number_input("Upper bound (b):", value=mean + sd)
        prob = norm.cdf(b, mean, sd) - norm.cdf(a, mean, sd)
        z1 = (a - mean) / sd
        z2 = (b - mean) / sd
        st.success(f"P({a} < X < {b}) = {round(prob, decimal)}")
        st.write(f"Z₁ = {round(z1, decimal)}, Z₂ = {round(z2, decimal)} → P(Z₁ < Z < Z₂) = {round(prob, decimal)}")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), color="orange", alpha=0.6)
        ax.axvline(a, color="red", linestyle="--")
        ax.axvline(b, color="red", linestyle="--")
        st.pyplot(fig)

    elif calc_type == "Inverse: Find x for given probability":
        tail = st.selectbox("Select tail:", ["Left tail", "Right tail", "Middle area"])
        p = st.number_input("Enter probability (0 < p < 1):", min_value=0.0, max_value=1.0, value=0.95)
        if tail == "Left tail":
            x_val = norm.ppf(p, mean, sd)
            st.success(f"x = {round(x_val, decimal)} for P(X < x) = {p}")
            a_val, b_val = None, None
        elif tail == "Right tail":
            x_val = norm.ppf(1 - p, mean, sd)
            st.success(f"x = {round(x_val, decimal)} for P(X > x) = {p}")
            a_val, b_val = None, None
        else:
            tail_prob = (1 - p) / 2
            a_val = norm.ppf(tail_prob, mean, sd)
            b_val = norm.ppf(1 - tail_prob, mean, sd)
            st.success(f"{p*100:.1f}% of data lies between {round(a_val, decimal)} and {round(b_val, decimal)}")
            x_val = None

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color="blue")
        if tail == "Middle area":
            ax.fill_between(x, 0, y, where=(x >= a_val) & (x <= b_val), color="orange", alpha=0.6)
            ax.axvline(a_val, color="red", linestyle="--")
            ax.axvline(b_val, color="red", linestyle="--")
        elif tail == "Left tail":
            ax.fill_between(x, 0, y, where=(x <= x_val), color="skyblue", alpha=0.6)
            ax.axvline(x_val, color="red", linestyle="--")
        else:
            ax.fill_between(x, 0, y, where=(x >= x_val), color="lightgreen", alpha=0.6)
            ax.axvline(x_val, color="red", linestyle="--")
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

    calc_type = st.selectbox("Choose a calculation:", [
        "P(X̄ < x)",
        "P(X̄ > x)",
        "P(a < X̄ < b)",
        "Inverse: Find x for given probability"
    ])

    x = np.linspace(mu - 4*se, mu + 4*se, 500)
    y = norm.pdf(x, mu, se)

    if calc_type == "P(X̄ < x)":
        x_val = st.number_input("Enter x̄ value:", value=mu)
        prob = norm.cdf(x_val, mu, se)
        z = (x_val - mu) / se
        st.success(f"P(X̄ < {x_val}) = {round(prob, decimal)}")
        st.write(f"Z = ({x_val} − {mu}) / {round(se, decimal)} = {round(z, decimal)}")

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x <= x_val), color="skyblue", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    elif calc_type == "P(X̄ > x)":
        x_val = st.number_input("Enter x̄ value:", value=mu)
        prob = 1 - norm.cdf(x_val, mu, se)
        z = (x_val - mu) / se
        st.success(f"P(X̄ > {x_val}) = {round(prob, decimal)}")
        st.write(f"Z = ({x_val} − {mu}) / {round(se, decimal)} = {round(z, decimal)}")

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x >= x_val), color="lightgreen", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    elif calc_type == "P(a < X̄ < b)":
        a = st.number_input("Lower bound (a):", value=mu - se)
        b = st.number_input("Upper bound (b):", value=mu + se)
        prob = norm.cdf(b, mu, se) - norm.cdf(a, mu, se)
        z1 = (a - mu) / se
        z2 = (b - mu) / se
        st.success(f"P({a} < X̄ < {b}) = {round(prob, decimal)}")
        st.write(f"Z₁ = {round(z1, decimal)}, Z₂ = {round(z2, decimal)} → P(Z₁ < Z < Z₂) = {round(prob, decimal)}")

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), color="orange", alpha=0.6)
        ax.axvline(a, color="red", linestyle="--")
        ax.axvline(b, color="red", linestyle="--")
        st.pyplot(fig)

    elif calc_type == "Inverse: Find x for given probability":
        p = st.number_input("Enter cumulative probability (0 < p < 1):", min_value=0.0, max_value=1.0, value=0.95)
        x_val = norm.ppf(p, mu, se)
        st.success(f"x̄ = {round(x_val, decimal)} for P(X̄ < x̄) = {p}")

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x <= x_val), color="skyblue", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
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
