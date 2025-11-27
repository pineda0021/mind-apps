# ==========================================================
# sample_proportion_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

# ==========================================================
# GLOBAL UNIVERSAL READABILITY THEME (Dark/Light Mode Safe)
# ==========================================================
plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "#2B2B2B",
    "axes.facecolor": "#2B2B2B",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "grid.color": "white",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.3,
})

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
    st.info("📘 Parameters: μ = population mean, σ = population standard deviation")

    st.latex(r"Z = \frac{X - \mu}{\sigma}")

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

    # ---------- P(X < x)
    if calc_type == "P(X < x)":
        x_val = st.number_input("Enter x value:", value=0.0)
        prob = norm.cdf(x_val, mean, sd)
        z = (x_val - mean) / sd

        st.latex(rf"""
        \text{{🧮 Step-by-step}} \\[4pt]
        Z = \frac{{{x_val} - {mean}}}{{{sd}}} = {z:.4f} \\[6pt]
        P(Z < {z:.4f}) = {prob:.4f} \\[6pt]
        \boxed{{P(X \le {x_val}) = {prob:.4f}}}
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.grid(True)
        ax.fill_between(x, 0, y, where=(x <= x_val), color="#6BB6FF", alpha=0.6)
        ax.axvline(x_val, color="white", linestyle="--")
        st.pyplot(fig)

    # ---------- P(X > x)
    elif calc_type == "P(X > x)":
        x_val = st.number_input("Enter x value:", value=0.0)
        prob = 1 - norm.cdf(x_val, mean, sd)
        z = (x_val - mean) / sd

        st.latex(rf"""
        \text{{🧮 Step-by-step}} \\[4pt]
        Z = \frac{{{x_val} - {mean}}}{{{sd}}} = {z:.4f} \\[6pt]
        P(Z > {z:.4f}) = {prob:.4f} \\[6pt]
        \boxed{{P(X \ge {x_val}) = {prob:.4f}}}
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.grid(True)
        ax.fill_between(x, 0, y, where=(x >= x_val), color="#90EE90", alpha=0.6)
        ax.axvline(x_val, color="white", linestyle="--")
        st.pyplot(fig)

    # ---------- P(a < X < b)
    elif calc_type == "P(a < X < b)":
        a = st.number_input("Lower bound (a):", value=mean - sd)
        b = st.number_input("Upper bound (b):", value=mean + sd)
        prob = norm.cdf(b, mean, sd) - norm.cdf(a, mean, sd)
        z1 = (a - mean) / sd
        z2 = (b - mean) / sd

        st.latex(rf"""
        Z_1 = {z1:.4f}, \quad Z_2 = {z2:.4f} \\[6pt]
        P({z1:.4f} < Z < {z2:.4f}) = {prob:.4f}
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.grid(True)
        ax.fill_between(x, 0, y, where=(a <= x) & (x <= b), color="#FFB347", alpha=0.6)
        ax.axvline(a, color="white", linestyle="--")
        ax.axvline(b, color="white", linestyle="--")
        st.pyplot(fig)

    # ---------- Inverse
    elif calc_type == "Inverse: Find x for given probability":
        tail = st.selectbox("Select tail:", ["Left tail", "Right tail"])
        p = st.number_input("Enter probability (0 < p < 1):",
                            value=0.95, min_value=0.0, max_value=1.0)

        if tail == "Left tail":
            z = norm.ppf(p)
            x_val = mean + z * sd
        else:
            z = norm.ppf(1 - p)
            x_val = mean + z * sd

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.grid(True)
        if tail == "Left tail":
            ax.fill_between(x, 0, y, where=(x <= x_val), color="#6BB6FF", alpha=0.6)
        else:
            ax.fill_between(x, 0, y, where=(x >= x_val), color="#90EE90", alpha=0.6)
        ax.axvline(x_val, color="white", linestyle="--")
        st.pyplot(fig)


# ==========================================================
# Sampling Distribution of the Mean
# ==========================================================
def sampling_mean(decimal):
    st.markdown("### 📘 **Sampling Distribution of the Mean**")
    st.info("📘 Parameters: μ = population mean, σ = population SD, n = sample size")
    st.latex(r"\mu_{\bar{X}} = \mu, \quad \sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}")

    mu_expr = st.text_input("Population mean (μ):", value="0")
    sigma_expr = st.text_input("Population SD (σ):", value="1")
    n = st.number_input("Sample size (n):", min_value=1, value=30)

    mu = parse_expression(mu_expr)
    sigma = parse_expression(sigma_expr)
    if mu is None or sigma is None:
        return

    se = sigma / math.sqrt(n)
    st.write(f"**Standard Error (σₓ̄) = {round(se, decimal)}**")

    calc_type = st.selectbox("Choose a calculation:", [
        "P(X̄ < x)",
        "P(X̄ > x)",
        "P(a < X̄ < b)"
    ])

    x = np.linspace(mu - 4*se, mu + 4*se, 500)
    y = norm.pdf(x, mu, se)

    # ----- All figures follow same new theme -----

    if calc_type == "P(X̄ < x)":
        x_val = st.number_input("Enter sample mean (x̄):", value=mu)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.grid(True)
        ax.fill_between(x, 0, y, where=(x <= x_val), color="#6BB6FF", alpha=0.6)
        ax.axvline(x_val, color="white", linestyle="--")
        st.pyplot(fig)

    elif calc_type == "P(X̄ > x)":
        x_val = st.number_input("Enter sample mean (x̄):", value=mu)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.grid(True)
        ax.fill_between(x, 0, y, where=(x >= x_val), color="#90EE90", alpha=0.6)
        ax.axvline(x_val, color="white", linestyle="--")
        st.pyplot(fig)

    elif calc_type == "P(a < X̄ < b)":
        a = st.number_input("Lower bound (a):", value=mu - se)
        b = st.number_input("Upper bound (b):", value=mu + se)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.grid(True)
        ax.fill_between(x, 0, y, where=(a <= x) & (x <= b), color="#FFB347", alpha=0.6)
        ax.axvline(a, color="white", linestyle="--")
        ax.axvline(b, color="white", linestyle="--")
        st.pyplot(fig)


# ==========================================================
# Sampling Distribution of the Proportion
# ==========================================================
def sampling_proportion(decimal):
    st.markdown("### 📗 **Sampling Distribution of the Proportion**")
    st.info("📘 Parameters: p = population proportion, n = sample size")
    st.latex(r"\mu_{\hat{p}} = p, \quad \sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}")

    p_expr = st.text_input("Population proportion (p):", value="0.5")
    n = st.number_input("Sample size (n):", min_value=1, value=30)

    p = parse_expression(p_expr)
    if p is None or not (0 < p < 1):
        st.error("p must be between 0 and 1.")
        return

    q = 1 - p
    se = math.sqrt(p * q / n)

    calc_type = st.selectbox("Choose a calculation:", [
        "P(p̂ < x)",
        "P(p̂ > x)",
        "P(a < p̂ < b)"
    ])

    x = np.linspace(p - 4*se, p + 4*se, 500)
    y = norm.pdf(x, p, se)

    if calc_type == "P(p̂ < x)":
        x_val = st.number_input("Enter sample proportion (p̂):", value=p)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.grid(True)
        ax.fill_between(x, 0, y, where=(x <= x_val), color="#6BB6FF", alpha=0.6)
        ax.axvline(x_val, color="white", linestyle="--")
        st.pyplot(fig)

    elif calc_type == "P(p̂ > x)":
        x_val = st.number_input("Enter sample proportion (p̂):", value=p)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.grid(True)
        ax.fill_between(x, 0, y, where=(x >= x_val), color="#90EE90", alpha=0.6)
        ax.axvline(x_val, color="white", linestyle="--")
        st.pyplot(fig)

    elif calc_type == "P(a < p̂ < b)":
        a = st.number_input("Lower bound (a):", value=p - se)
        b = st.number_input("Upper bound (b):", value=p + se)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.grid(True)
        ax.fill_between(x, 0, y, where=(a <= x) & (x <= b), color="#FFB347", alpha=0.6)
        ax.axvline(a, color="white", linestyle="--")
        ax.axvline(b, color="white", linestyle="--")
        st.pyplot(fig)


# ==========================================================
# Uniform Distribution
# ==========================================================
def uniform_distribution(decimal):
    st.markdown("### 🎲 **Uniform Distribution**")
    st.info("📘 Parameters: a = lower bound, b = upper bound")
    st.latex(r"f(x) = \frac{1}{b - a}, \quad a \le x \le b")

    a = st.number_input("Lower bound (a):", value=0.0)
    b = st.number_input("Upper bound (b):", value=10.0)

    if b <= a:
        st.error("⚠️ Upper bound must be greater than lower bound.")
        return

    pdf = 1 / (b - a)
    x = np.linspace(a - (b - a) * 0.2, b + (b - a) * 0.2, 500)
    y = np.where((x >= a) & (x <= b), pdf, 0)

    calc_type = st.selectbox("Choose a calculation:", [
        "P(X < x)",
        "P(X = x)",
        "P(X > x)",
        "P(a < X < b)",
        "Inverse: Find x"
    ])

    if calc_type == "P(X < x)":
        x_val = st.number_input("Enter x value:", value=(a+b)/2)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y)
        ax.grid(True)
        ax.fill_between(x, 0, y, where=(x <= x_val) & (x >= a), color="#6BB6FF", alpha=0.6)
        ax.axvline(x_val, color="white", linestyle="--")
        st.pyplot(fig)

    if calc_type == "P(X = x)":
        x_val = st.number_input("Enter x value:", value=(a+b)/2)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y)
        ax.grid(True)
        ax.axvline(x_val, color="white", linestyle="--")
        st.pyplot(fig)

    if calc_type == "P(X > x)":
        x_val = st.number_input("Enter x value:", value=(a+b)/2)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y)
        ax.grid(True)
        ax.fill_between(x, 0, y, where=(x >= x_val) & (x <= b), color="#90EE90", alpha=0.6)
        ax.axvline(x_val, color="white", linestyle="--")
        st.pyplot(fig)

    if calc_type == "P(a < X < b)":
        low = st.number_input("Lower bound:", value=a)
        high = st.number_input("Upper bound:", value=b)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y)
        ax.grid(True)
        ax.fill_between(x, 0, y, where=(x >= low) & (x <= high), color="#FFB347", alpha=0.6)
        ax.axvline(low, color="white", linestyle="--")
        ax.axvline(high, color="white", linestyle="--")
        st.pyplot(fig)

    if calc_type == "Inverse: Find x":
        p = st.number_input("Probability p (0 < p < 1):", value=0.5)
        x_val = a + p * (b - a)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y)
        ax.grid(True)
        ax.fill_between(x, 0, y, where=(x >= a) & (x <= x_val), color="#6BB6FF", alpha=0.6)
        ax.axvline(x_val, color="white", linestyle="--")
        st.pyplot(fig)

# ==========================================================
# Run App
# ==========================================================
def run():
    st.header("🧠 MIND: Continuous Probability Distributions")
    st.markdown("""
    ---
    **Quick Reference:**
    - μ = population mean  
    - σ = population standard deviation  
    - n = sample size  
    - x̄ = sample mean  
    - p̂ = sample proportion  
    ---
    """)

    dist_choice = st.radio(
        "Select Distribution:",
        [
            "Uniform Distribution",
            "Normal Distribution",
            "Sampling Distribution of the Mean",
            "Sampling Distribution of the Proportion"
        ],
        horizontal=True
    )

    decimal = st.number_input("Decimal places:", min_value=0, max_value=10, value=4, step=1)

    if dist_choice == "Uniform Distribution":
        uniform_distribution(decimal)
    elif dist_choice == "Normal Distribution":
        normal_distribution(decimal)
    elif dist_choice == "Sampling Distribution of the Mean":
        sampling_mean(decimal)
    elif dist_choice == "Sampling Distribution of the Proportion":
        sampling_proportion(decimal)


if __name__ == "__main__":
    run()

