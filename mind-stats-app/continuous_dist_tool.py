# ==========================================================
# sample_proportion_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# Updated for Universal Readability (Dark & Light Mode Safe)
# ==========================================================

import streamlit as st
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

# ==========================================================
# Universal Readability Theme (Dark + Light Mode Safe)
# ==========================================================
st.markdown("""
    <style>
        /* Make all text readable in both dark and light mode */
        body, [class^="css"], .stTextInput, .stNumberInput, .stSelectbox,
        .stMarkdown, label, .stRadio, .stCheckbox, .stSelectbox div {
            color: white !important;
        }

        /* Fix all headers */
        h1, h2, h3, h4, h5, h6 {
            color: white !important;
        }

        /* Fix LaTeX visibility */
        .katex-html {
            color: white !important;
        }

        /* Plotly background consistency */
        .js-plotly-plot .plotly {
            background-color: #2B2B2B !important;
        }

        /* Code blocks */
        pre, code {
            color: white !important;
            background-color: #2B2B2B !important;
        }
    </style>
""", unsafe_allow_html=True)

# Matplotlib white text (dark-mode safe)
plt.rcParams.update({
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.edgecolor": "white",
    "figure.facecolor": "#2B2B2B",
    "axes.facecolor": "#2B2B2B",
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

    st.latex(r"""
        f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}
    """)
    st.latex(r"Z = \frac{X - \mu}{\sigma}")

    mean = st.number_input("Population mean (μ):", value=0.0)
    sd = st.number_input("Standard deviation (σ):", min_value=0.0001, value=1.0)

    calc_type = st.selectbox(
        "Choose a calculation:",
        ["P(X < x)", "P(X > x)", "P(a < X < b)", "Inverse: Find x for given probability"]
    )

    x = np.linspace(mean - 4*sd, mean + 4*sd, 500)
    y = norm.pdf(x, mean, sd)

    # ---------- P(X < x)
    if calc_type == "P(X < x)":
        x_val = st.number_input("Enter x value:", value=0.0)
        prob = norm.cdf(x_val, mean, sd)
        z = (x_val - mean) / sd

        st.latex(rf"""
            Z = \frac{{{x_val} - {mean}}}{{{sd}}} = {z:.4f} \\
            P(Z < {z:.4f}) = {prob:.{decimal}f} \\
            \boxed{{P(X \le {x_val}) = {prob:.{decimal}f}}}
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x <= x_val), alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # ---------- P(X > x)
    elif calc_type == "P(X > x)":
        x_val = st.number_input("Enter x value:", value=0.0)
        prob = 1 - norm.cdf(x_val, mean, sd)
        z = (x_val - mean) / sd

        st.latex(rf"""
            Z = \frac{{{x_val} - {mean}}}{{{sd}}} = {z:.4f} \\
            P(Z > {z:.4f}) = {prob:.{decimal}f} \\
            \boxed{{P(X \ge {x_val}) = {prob:.{decimal}f}}}
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= x_val), alpha=0.6)
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
            Z_1 = {z1:.4f},\ Z_2 = {z2:.4f} \\
            P({z1:.4f} < Z < {z2:.4f}) = {prob:.{decimal}f} \\
            \boxed{{P({a} < X < {b}) = {prob:.{decimal}f}}}
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), alpha=0.6)
        ax.axvline(a, color="red", linestyle="--")
        ax.axvline(b, color="red", linestyle="--")
        st.pyplot(fig)

    # ---------- Inverse
    else:
        tail = st.selectbox("Select tail:", ["Left tail", "Right tail"])
        p = st.number_input("Enter probability p:", 0.0, 1.0, 0.95)

        if tail == "Left tail":
            z = norm.ppf(p)
        else:
            z = norm.ppf(1 - p)

        x_val = mean + z * sd

        st.latex(rf"""
            Z_p = {z:.4f} \\
            x = \mu + Z_p \sigma = {x_val:.{decimal}f} \\
            \boxed{{x = {x_val:.{decimal}f}}}
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        if tail == "Left tail":
            ax.fill_between(x, 0, y, where=(x <= x_val), alpha=0.6)
        else:
            ax.fill_between(x, 0, y, where=(x >= x_val), alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)


# ==========================================================
# Sampling Distribution of the Mean
# ==========================================================
def sampling_mean(decimal):
    st.markdown("### 📘 **Sampling Distribution of the Mean**")

    st.latex(r"""
        \mu_{\bar{X}} = \mu,\quad
        \sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}},\quad
        Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}}
    """)

    mu_expr = st.text_input("Population mean (μ):", value="0")
    sigma_expr = st.text_input("Population SD (σ):", value="1")
    n = st.number_input("Sample size (n):", min_value=1, value=30)

    mu = parse_expression(mu_expr)
    sigma = parse_expression(sigma_expr)
    if mu is None or sigma is None:
        return

    se = sigma / math.sqrt(n)
    st.write(f"**Standard Error (σₓ̄) = {round(se, decimal)}**")

    calc_type = st.selectbox(
        "Choose a calculation:",
        ["P(X̄ < x)", "P(X̄ > x)", "P(a < X̄ < b)", "Inverse: Find x̄"]
    )

    x = np.linspace(mu - 4*se, mu + 4*se, 500)
    y = norm.pdf(x, mu, se)

    # ------- P(X̄ < x)
    if calc_type == "P(X̄ < x)":
        x_val = st.number_input("Enter sample mean (x̄):", value=mu)
        z = (x_val - mu) / se
        prob = norm.cdf(z)

        st.latex(rf"""
            Z = {z:.4f} \\
            P(Z < {z:.4f}) = {prob:.{decimal}f} \\
            \boxed{{P(\bar{{X}} < {x_val}) = {prob:.{decimal}f}}}
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x <= x_val), alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # ------- P(X̄ > x)
    elif calc_type == "P(X̄ > x)":
        x_val = st.number_input("Enter sample mean (x̄):", value=mu)
        z = (x_val - mu) / se
        prob = 1 - norm.cdf(z)

        st.latex(rf"""
            Z = {z:.4f} \\
            P(Z > {z:.4f}) = {prob:.{decimal}f} \\
            \boxed{{P(\bar{{X}} > {x_val}) = {prob:.{decimal}f}}}
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= x_val), alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # ------- P(a < X̄ < b)
    elif calc_type == "P(a < X̄ < b)":
        a = st.number_input("Lower bound (a):", value=mu - se)
        b = st.number_input("Upper bound (b):", value=mu + se)

        z1 = (a - mu) / se
        z2 = (b - mu) / se
        prob = norm.cdf(z2) - norm.cdf(z1)

        st.latex(rf"""
            P({z1:.4f} < Z < {z2:.4f}) = {prob:.{decimal}f} \\
            \boxed{{P({a} < \bar{{X}} < {b}) = {prob:.{decimal}f}}}
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), alpha=0.6)
        ax.axvline(a, color="red", linestyle="--")
        ax.axvline(b, color="red", linestyle="--")
        st.pyplot(fig)

    # ------- Inverse
    else:
        p = st.number_input("Enter probability p:", 0.0, 1.0, 0.95)
        z = norm.ppf(p)
        x_val = mu + z * se

        st.latex(rf"""
            \bar X = {x_val:.{decimal}f} \\
            \boxed{{\bar X = {x_val:.{decimal}f}}}
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x <= x_val), alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)


# ==========================================================
# Sampling Distribution of the Proportion
# ==========================================================
def sampling_proportion(decimal):
    st.markdown("### 📗 **Sampling Distribution of the Proportion**")

    st.latex(r"""
        \mu_{\hat p} = p,\quad
        \sigma_{\hat p} = \sqrt{\frac{p(1-p)}{n}},\quad
        Z = \frac{\hat p - p}{\sigma_{\hat p}}
    """)

    p_expr = st.text_input("Population proportion (p):", value="0.5")
    n = st.number_input("Sample size (n):", min_value=1, value=30)

    p = parse_expression(p_expr)
    if p is None or not (0 < p < 1):
        st.error("p must be between 0 and 1.")
        return

    q = 1 - p
    se = math.sqrt(p * q / n)
    st.write(f"**Standard Error (σₚ̂) = {round(se, decimal)}**")

    calc_type = st.selectbox(
        "Choose a calculation:",
        ["P(p̂ < x)", "P(p̂ > x)", "P(a < p̂ < b)"]
    )

    x = np.linspace(p - 4*se, p + 4*se, 500)
    y = norm.pdf(x, p, se)

    # ------- P(p̂ < x)
    if calc_type == "P(p̂ < x)":
        x_val = st.number_input("Enter sample proportion (p̂):", value=p)
        z = (x_val - p) / se
        prob = norm.cdf(z)

        st.latex(rf"""
            P(\hat p < {x_val}) = {prob:.{decimal}f}
        """)

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x <= x_val), alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # ------- P(p̂ > x)
    elif calc_type == "P(p̂ > x)":
        x_val = st.number_input("Enter sample proportion (p̂):", value=p)
        z = (x_val - p) / se
        prob = 1 - norm.cdf(z)

        st.latex(rf"""
            P(\hat p > {x_val}) = {prob:.{decimal}f}
        """)

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= x_val), alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # ------- P(a < p̂ < b)
    else:
        a = st.number_input("Lower bound (a):", value=p - se)
        b = st.number_input("Upper bound (b):", value=p + se)

        z1 = (a - p) / se
        z2 = (b - p) / se
        prob = norm.cdf(z2) - norm.cdf(z1)

        st.latex(rf"""
            P({a} < \hat p < {b}) = {prob:.{decimal}f}
        """)

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), alpha=0.6)
        ax.axvline(a, color="red", linestyle="--")
        ax.axvline(b, color="red", linestyle="--")
        st.pyplot(fig)


# ==========================================================
# Uniform Distribution
# ==========================================================
def uniform_distribution(decimal):
    st.markdown("### 🎲 **Uniform Distribution**")

    st.latex(r"""
        f(x) = \frac{1}{b - a},\quad a \le x \le b
    """)

    st.latex(r"""
        P(X \le x) = \frac{x - a}{b - a},\quad
        P(X \ge x) = \frac{b - x}{b - a}
    """)

    a = st.number_input("Lower bound (a):", value=0.0)
    b = st.number_input("Upper bound (b):", value=10.0)

    if b <= a:
        st.error("⚠️ Upper bound (b) must be greater than lower bound (a).")
        return

    pdf = 1 / (b - a)
    st.write(f"**Constant PDF:** f(x) = {round(pdf, decimal)}")

    calc_type = st.selectbox(
        "Choose a calculation:",
        [
            "P(X < x) = P(X ≤ x)",
            "P(X = x)",
            "P(X > x) = P(X ≥ x)",
            "P(a < X < b)",
            "Inverse: Find x for given probability"
        ]
    )

    x = np.linspace(a - (b - a)*0.2, b + (b - a)*0.2, 500)
    y = np.where((x >= a) & (x <= b), pdf, 0)

    # ----- P(X < x)
    if calc_type == "P(X < x) = P(X ≤ x)":
        x_val = st.number_input("Enter x value:", value=(a + b) / 2)

        if x_val <= a:
            prob = 0.0
        elif x_val >= b:
            prob = 1.0
        else:
            prob = (x_val - a) / (b - a)

        st.markdown(f"""
            **Final Answer: P(X ≤ {x_val}) = {round(prob, decimal)}**
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x <= x_val) & (x >= a), alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # ----- P(X = x)
    elif calc_type == "P(X = x)":
        x_val = st.number_input("Enter x value:", value=(a + b) / 2)

        st.markdown(f"**P(X = {x_val}) = 0** (continuous distribution)")

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # ----- P(X > x)
    elif calc_type == "P(X > x) = P(X ≥ x)":
        x_val = st.number_input("Enter x value:", value=(a + b) / 2)

        if x_val <= a:
            prob = 1.0
        elif x_val >= b:
            prob = 0.0
        else:
            prob = (b - x_val) / (b - a)

        st.markdown(f"""
            **Final Answer: P(X ≥ {x_val}) = {round(prob, decimal)}**
        """)

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= x_val) & (x <= b), alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # ----- P(a < X < b)
    elif calc_type == "P(a < X < b)":
        low = st.number_input("Lower bound:", value=a)
        high = st.number_input("Upper bound:", value=b)

        if high <= low:
            st.error("⚠️ Upper bound must be greater.")
            return

        if high < a or low > b:
            prob = 0.0
        else:
            lower = max(low, a)
            upper = min(high, b)
            prob = (upper - lower) / (b - a)

        st.markdown(f"""
            **Final Answer: P({low} < X < {high}) = {round(prob, decimal)}**
        """)

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= low) & (x <= high), alpha=0.6)
        ax.axvline(low, color="red", linestyle="--")
        ax.axvline(high, color="red", linestyle="--")
        st.pyplot(fig)

    # ----- Inverse
    else:
        p = st.number_input("Enter probability p:", 0.0, 1.0, 0.5)
        x_val = a + p * (b - a)

        st.markdown(f"""
            **Final Answer: x = {round(x_val, decimal)}**
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x <= x_val) & (x >= a), alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)


# ==========================================================
# MAIN APP
# ==========================================================
def run():

    # 🔥 Updated header per your request
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

    dist_choice = st.selectbox(
        "Select Distribution Type:",
        [
            "Uniform Distribution",
            "Normal Distribution",
            "Sampling Distribution of the Mean",
            "Sampling Distribution of the Proportion"
        ]
    )

    decimal = st.number_input(
        "Decimal places for output:",
        min_value=0, max_value=10, value=4, step=1
    )

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
