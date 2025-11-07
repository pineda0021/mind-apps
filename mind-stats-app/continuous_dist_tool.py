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
        ax.fill_between(x, 0, y, where=(x <= x_val), color="skyblue", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
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
        ax.fill_between(x, 0, y, where=(x >= x_val), color="lightgreen", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # ---------- P(a < X < b)
    elif calc_type == "P(a < X < b)":
        a = st.number_input("Lower bound (a):", value=mean - sd)
        b = st.number_input("Upper bound (b):", value=mean + sd)
        prob = norm.cdf(b, mean, sd) - norm.cdf(a, mean, sd)
        z1 = (a - mean) / sd
        z2 = (b - mean) / sd

        st.latex(rf"""
        \text{{🧮 Step-by-step}} \\[4pt]
        Z_1 = {z1:.4f}, \quad Z_2 = {z2:.4f} \\[6pt]
        P({z1:.4f} < Z < {z2:.4f}) = {prob:.4f} \\[6pt]
        \boxed{{P({a} < X < {b}) = {prob:.4f}}}
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), color="orange", alpha=0.6)
        ax.axvline(a, color="red", linestyle="--")
        ax.axvline(b, color="red", linestyle="--")
        st.pyplot(fig)

    # ---------- Inverse
    elif calc_type == "Inverse: Find x for given probability":
        tail = st.selectbox("Select tail:", ["Left tail", "Right tail", "Middle area"])
        p = st.number_input("Enter probability (0 < p < 1):", value=0.95, min_value=0.0, max_value=1.0)

        if tail == "Left tail":
            z = norm.ppf(p)
            x_val = mean + z * sd
            st.latex(rf"""
            \text{{🧮 Step-by-step}} \\[4pt]
            Z_p = {z:.4f} \\[6pt]
            x = \mu + Z_p \cdot \sigma = {mean} + ({z:.4f})({sd}) = {x_val:.4f} \\[6pt]
            \boxed{{P(X < {x_val:.4f}) = {p}}}
            """)
        elif tail == "Right tail":
            z = norm.ppf(1 - p)
            x_val = mean + z * sd
            st.latex(rf"""
            \text{{🧮 Step-by-step}} \\[4pt]
            Z_{{1-p}} = {z:.4f} \\[6pt]
            x = \mu + Z_{{1-p}} \cdot \sigma = {mean} + ({z:.4f})({sd}) = {x_val:.4f} \\[6pt]
            \boxed{{P(X > {x_val:.4f}) = {p}}}
            """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        if tail == "Left tail":
            ax.fill_between(x, 0, y, where=(x <= x_val), color="skyblue", alpha=0.6)
        else:
            ax.fill_between(x, 0, y, where=(x >= x_val), color="lightgreen", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)


# ==========================================================
# Sampling Distribution of the Mean
# ==========================================================
def sampling_mean(decimal):
    st.markdown("### 📘 **Sampling Distribution of the Mean**")
    st.info("📘 Parameters: μ = population mean, σ = population SD, n = sample size")
    st.latex(r"\mu_{\bar{X}} = \mu, \quad \sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}")
    st.latex(r"Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}}")

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
        "P(a < X̄ < b)",
        "Inverse: Find x̄ for given probability"
    ])

    x = np.linspace(mu - 4*se, mu + 4*se, 500)
    y = norm.pdf(x, mu, se)

    # ---------- P(X̄ < x)
    if calc_type == "P(X̄ < x)":
        x_val = st.number_input("Enter sample mean (x̄):", value=mu)
        z = (x_val - mu) / se
        prob = norm.cdf(z)
        st.latex(rf"""
        \text{{🧮 Step-by-step}} \\[4pt]
        Z = \frac{{{x_val} - {mu}}}{{{sigma}/\sqrt{{{n}}}}} = {z:.4f} \\[6pt]
        P(Z < {z:.4f}) = {prob:.4f} \\[6pt]
        \boxed{{P(\bar{{X}} < {x_val}) = {prob:.4f}}}
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x <= x_val), color="skyblue", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # ---------- P(X̄ > x)
    elif calc_type == "P(X̄ > x)":
        x_val = st.number_input("Enter sample mean (x̄):", value=mu)
        z = (x_val - mu) / se
        prob = 1 - norm.cdf(z)
        st.latex(rf"""
        \text{{🧮 Step-by-step}} \\[4pt]
        Z = \frac{{{x_val} - {mu}}}{{{sigma}/\sqrt{{{n}}}}} = {z:.4f} \\[6pt]
        P(Z > {z:.4f}) = {prob:.4f} \\[6pt]
        \boxed{{P(\bar{{X}} > {x_val}) = {prob:.4f}}}
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= x_val), color="lightgreen", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # ---------- P(a < X̄ < b)
    elif calc_type == "P(a < X̄ < b)":
        a = st.number_input("Lower bound (a):", value=mu - se)
        b = st.number_input("Upper bound (b):", value=mu + se)
        z1 = (a - mu) / se
        z2 = (b - mu) / se
        prob = norm.cdf(z2) - norm.cdf(z1)
        st.latex(rf"""
        \text{{🧮 Step-by-step}} \\[4pt]
        Z_1 = {z1:.4f}, \quad Z_2 = {z2:.4f} \\[6pt]
        P({z1:.4f} < Z < {z2:.4f}) = {prob:.4f} \\[6pt]
        \boxed{{P({a} < \bar{{X}} < {b}) = {prob:.4f}}}
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), color="orange", alpha=0.6)
        ax.axvline(a, color="red", linestyle="--")
        ax.axvline(b, color="red", linestyle="--")
        st.pyplot(fig)


# ==========================================================
# Sampling Distribution of the Proportion
# ==========================================================
def sampling_proportion(decimal):
    st.markdown("### 📗 **Sampling Distribution of the Proportion**")
    st.info("📘 Parameters: p = population proportion, n = sample size")
    st.latex(r"\mu_{\hat{p}} = p, \quad \sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}")
    st.latex(r"Z = \frac{\hat{p} - p}{\sqrt{p(1-p)/n}}")

    p_expr = st.text_input("Population proportion (p):", value="0.5")
    n = st.number_input("Sample size (n):", min_value=1, value=30)

    p = parse_expression(p_expr)
    if p is None or not (0 < p < 1):
        st.error("p must be between 0 and 1.")
        return

    q = 1 - p
    se = math.sqrt(p * q / n)
    st.write(f"**Standard Error (σₚ̂) = {round(se, decimal)}**")

    calc_type = st.selectbox("Choose a calculation:", [
        "P(p̂ < x)",
        "P(p̂ > x)",
        "P(a < p̂ < b)"
    ])

    x = np.linspace(p - 4*se, p + 4*se, 500)
    y = norm.pdf(x, p, se)

    # ---------- P(p̂ < x)
    if calc_type == "P(p̂ < x)":
        x_val = st.number_input("Enter sample proportion (p̂):", value=p)
        z = (x_val - p) / se
        prob = norm.cdf(z)
        st.latex(rf"""
        \text{{🧮 Step-by-step}} \\[4pt]
        Z = \frac{{{x_val} - {p}}}{{\sqrt{{{p}(1-{p})/{n}}}}} = {z:.4f} \\[6pt]
        P(Z < {z:.4f}) = {prob:.4f} \\[6pt]
        \boxed{{P(\hat{{p}} < {x_val}) = {prob:.4f}}}
        """)

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x <= x_val), color="skyblue", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # ---------- P(p̂ > x)
    elif calc_type == "P(p̂ > x)":
        x_val = st.number_input("Enter sample proportion (p̂):", value=p)
        z = (x_val - p) / se
        prob = 1 - norm.cdf(z)
        st.latex(rf"""
        \text{{🧮 Step-by-step}} \\[4pt]
        Z = \frac{{{x_val} - {p}}}{{\sqrt{{{p}(1-{p})/{n}}}}} = {z:.4f} \\[6pt]
        P(Z > {z:.4f}) = {prob:.4f} \\[6pt]
        \boxed{{P(\hat{{p}} > {x_val}) = {prob:.4f}}}
        """)

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= x_val), color="lightgreen", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # ---------- P(a < p̂ < b)
    elif calc_type == "P(a < p̂ < b)":
        a = st.number_input("Lower bound (a):", value=p - se)
        b = st.number_input("Upper bound (b):", value=p + se)
        z1 = (a - p) / se
        z2 = (b - p) / se
        prob = norm.cdf(z2) - norm.cdf(z1)
        st.latex(rf"""
        \text{{🧮 Step-by-step}} \\[4pt]
        Z_1 = {z1:.4f}, \quad Z_2 = {z2:.4f} \\[6pt]
        P({z1:.4f} < Z < {z2:.4f}) = {prob:.4f} \\[6pt]
        \boxed{{P({a} < \hat{{p}} < {b}) = {prob:.4f}}}
        """)

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), color="orange", alpha=0.6)
        ax.axvline(a, color="red", linestyle="--")
        ax.axvline(b, color="red", linestyle="--")
        st.pyplot(fig)


# ==========================================================
# Uniform Distribution (Text-based + interpretation tips)
# ==========================================================
def uniform_distribution(decimal):
    st.markdown("### 🎲 **Uniform Distribution**")
    st.info("📘 Parameters: a = lower bound, b = upper bound")
    st.latex(r"f(x) = \frac{1}{b - a}, \quad a \le x \le b")

    a = st.number_input("Lower bound (a):", value=0.0)
    b = st.number_input("Upper bound (b):", value=10.0)
    if b <= a:
        st.error("⚠️ Upper bound (b) must be greater than lower bound (a).")
        return

    pdf = 1 / (b - a)
    st.write(f"**Constant PDF:** f(x) = {round(pdf, decimal)} for {a} ≤ x ≤ {b}")

    calc_type = st.selectbox("Choose a calculation:", [
        "P(X < x) = P(X ≤ x)",
        "P(X = x)",
        "P(X > x) = P(X ≥ x)",
        "P(a < X < b)",
        "Inverse: Find x for given probability"
    ])

    x = np.linspace(a - (b - a)*0.2, b + (b - a)*0.2, 500)
    y = np.where((x >= a) & (x <= b), pdf, 0)

    # --- P(X < x)
    if calc_type == "P(X < x) = P(X ≤ x)":
        x_val = st.number_input("Enter x value:", value=(a + b) / 2)
        if x_val <= a:
            prob = 0.0
        elif x_val >= b:
            prob = 1.0
        else:
            prob = (x_val - a) / (b - a)
        st.markdown(f"""
        **🧮 Step-by-step:**
        1. P(X ≤ x) = (x − a) / (b − a)  
        2. P(X ≤ {x_val}) = ({x_val} − {a}) / ({b} − {a}) = {round(prob, decimal)}  
        3. **Final Answer:** P(X ≤ {x_val}) = {round(prob, decimal)}  
        """)
        st.info("📘 Interpretation Tip: This represents the proportion of outcomes where X is less than or equal to the chosen value.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x >= a) & (x <= x_val), color="skyblue", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # --- P(X = x)
    elif calc_type == "P(X = x)":
        x_val = st.number_input("Enter x value:", value=(a + b) / 2)
        st.markdown(f"""
        **🧮 Step-by-step:**
        1. In a continuous distribution, P(X = x) = 0  
        2. **Final Answer:** P(X = {x_val}) = 0  
        """)
        st.info("📘 Interpretation Tip: In continuous distributions, exact values have zero probability, but intervals have measurable probabilities.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color="blue")
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # --- P(X > x)
    elif calc_type == "P(X > x) = P(X ≥ x)":
        x_val = st.number_input("Enter x value:", value=(a + b) / 2)
        if x_val <= a:
            prob = 1.0
        elif x_val >= b:
            prob = 0.0
        else:
            prob = (b - x_val) / (b - a)
        st.markdown(f"""
        **🧮 Step-by-step:**
        1. P(X ≥ x) = (b − x) / (b − a)  
        2. P(X ≥ {x_val}) = ({b} − {x_val}) / ({b} − {a}) = {round(prob, decimal)}  
        3. **Final Answer:** P(X ≥ {x_val}) = {round(prob, decimal)}  
        """)
        st.info("📘 Interpretation Tip: This represents the proportion of outcomes where X is greater than or equal to the chosen value.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x >= x_val) & (x <= b), color="lightgreen", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # --- P(a < X < b)
    elif calc_type == "P(a < X < b)":
        low = st.number_input("Lower bound (a):", value=a)
        high = st.number_input("Upper bound (b):", value=b)
        if high <= low:
            st.error("⚠️ Upper bound must be greater than lower bound.")
            return
        if high < a or low > b:
            prob = 0.0
        else:
            lower = max(low, a)
            upper = min(high, b)
            prob = (upper - lower) / (b - a)
        st.markdown(f"""
        **🧮 Step-by-step:**
        1. P(a < X < b) = (b − a) / (B − A)  
        2. P({low} < X < {high}) = ({high} − {low}) / ({b} − {a}) = {round(prob, decimal)}  
        3. **Final Answer:** P({low} < X < {high}) = {round(prob, decimal)}  
        """)
        st.info("📘 Interpretation Tip: This gives the probability that X lies between two specific values within the uniform range.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x >= low) & (x <= high), color="orange", alpha=0.6)
        ax.axvline(low, color="red", linestyle="--")
        ax.axvline(high, color="red", linestyle="--")
        st.pyplot(fig)

    # --- Inverse: Find x
    elif calc_type == "Inverse: Find x for given probability":
        p = st.number_input("Enter probability p for P(X ≤ x) = p (0 < p < 1):",
                            min_value=0.0, max_value=1.0, value=0.5)
        x_val = a + p * (b - a)
        st.markdown(f"""
        **🧮 Step-by-step:**
        1. P(X ≤ x) = (x − a) / (b − a)  
        2. Solve for x → x = a + p(b − a)  
        3. x = {a} + {p}({b} − {a}) = {round(x_val, decimal)}  
        4. **Final Answer:** x = {round(x_val, decimal)} for P(X ≤ x) = {p}  
        """)
        st.info("📘 Interpretation Tip: This finds the cutoff x below which a given proportion (p) of values in the uniform distribution fall.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x >= a) & (x <= x_val), color="skyblue", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)


# ==========================================================
# MAIN APP
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
        "Select Distribution Type:",
        [
            "Uniform Distribution",
            "Normal Distribution",
            "Sampling Distribution of the Mean",
            "Sampling Distribution of the Proportion"
        ],
        horizontal=True
    )

    decimal = st.number_input("Decimal places for output:", min_value=0, max_value=10, value=4, step=1)

    if dist_choice == "Uniform Distribution":
        uniform_distribution(decimal)
    elif dist_choice == "Normal Distribution":
        normal_distribution(decimal)
    elif dist_choice == "Sampling Distribution of the Mean":
        sampling_mean(decimal)
    elif dist_choice == "Sampling Distribution of the Proportion":
        sampling_proportion(decimal)


# ==========================================================
# Run App
# ==========================================================
if __name__ == "__main__":
    run()
