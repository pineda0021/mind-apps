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


def _fmt(x, d):
    """Round helper that returns string with fixed decimals."""
    return f"{round(x, d)}"


# ==========================================================
# Normal Distribution
# ==========================================================
def normal_distribution(decimal):
    st.markdown("### 📈 **Normal Distribution**")
    st.latex(r"Z = \frac{X - \mu}{\sigma}")

    mean = st.number_input("Population mean (μ):", value=0.0, key="norm_mu")
    sd = st.number_input("Standard deviation (σ):", min_value=0.0001, value=1.0, key="norm_sd")

    calc_type = st.selectbox("Choose a calculation:", [
        "P(X < x)",
        "P(X > x)",
        "P(a < X < b)",
        "Inverse: Find x for given probability"
    ], key="norm_calc_type")

    x = np.linspace(mean - 4*sd, mean + 4*sd, 500)
    y = norm.pdf(x, mean, sd)

    # P(X < x)
    if calc_type == "P(X < x)":
        x_val = st.number_input("Enter x value:", value=0.0, key="norm_x_lt")
        prob = norm.cdf(x_val, mean, sd)
        z = (x_val - mean) / sd
        st.success(f"P(X ≤ {x_val}) = {_fmt(prob, decimal)}")

        st.markdown(
            rf"""
**🧮 Step-by-step:**
1. Compute \( Z = \frac{{x - \mu}}{{\sigma}} = \frac{{{x_val} - {mean}}}{{{sd}}} = {_fmt(z, decimal)} \)
2. Find \( P(Z < {_fmt(z, decimal)}) = {_fmt(prob, decimal)} \)
3. **Final answer:** \( P(X \le {x_val}) = {_fmt(prob, decimal)} \)
            """
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x <= x_val), alpha=0.6)
        ax.axvline(x_val, linestyle="--")
        st.pyplot(fig)

    # P(X > x)
    elif calc_type == "P(X > x)":
        x_val = st.number_input("Enter x value:", value=0.0, key="norm_x_gt")
        prob = 1 - norm.cdf(x_val, mean, sd)
        z = (x_val - mean) / sd
        st.success(f"P(X ≥ {x_val}) = {_fmt(prob, decimal)}")

        st.markdown(
            rf"""
**🧮 Step-by-step:**
1. Compute \( Z = \frac{{x - \mu}}{{\sigma}} = \frac{{{x_val} - {mean}}}{{{sd}}} = {_fmt(z, decimal)} \)
2. Find \( P(Z > {_fmt(z, decimal)}) = 1 - \Phi({_fmt(z, decimal)}) = {_fmt(prob, decimal)} \)
3. **Final answer:** \( P(X \ge {x_val}) = {_fmt(prob, decimal)} \)
            """
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= x_val), alpha=0.6)
        ax.axvline(x_val, linestyle="--")
        st.pyplot(fig)

    # P(a < X < b)
    elif calc_type == "P(a < X < b)":
        a = st.number_input("Lower bound (a):", value=mean - sd, key="norm_between_a")
        b = st.number_input("Upper bound (b):", value=mean + sd, key="norm_between_b")
        if b <= a:
            st.error("Upper bound must be greater than lower bound.")
            return
        prob = norm.cdf(b, mean, sd) - norm.cdf(a, mean, sd)
        z1 = (a - mean) / sd
        z2 = (b - mean) / sd
        st.success(f"P({a} < X < {b}) = {_fmt(prob, decimal)}")

        st.markdown(
            rf"""
**🧮 Step-by-step:**
1. Compute \( Z_1 = \frac{{a - \mu}}{{\sigma}} = {_fmt(z1, decimal)},\quad Z_2 = \frac{{b - \mu}}{{\sigma}} = {_fmt(z2, decimal)} \)
2. \( P(a<X<b) = \Phi({_fmt(z2, decimal)}) - \Phi({_fmt(z1, decimal)}) = {_fmt(prob, decimal)} \)
3. **Final answer:** \( P({a}<X<{b}) = {_fmt(prob, decimal)} \)
            """
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), alpha=0.6)
        ax.axvline(a, linestyle="--")
        ax.axvline(b, linestyle="--")
        st.pyplot(fig)

    # Inverse
    elif calc_type == "Inverse: Find x for given probability":
        tail = st.selectbox("Select tail:", ["Left tail", "Right tail", "Middle area"], key="norm_inv_tail")
        p = st.number_input("Enter probability (0 < p < 1):", min_value=0.0, max_value=1.0, value=0.95, key="norm_inv_p")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)

        if tail == "Left tail":
            x_val = norm.ppf(p, mean, sd)
            st.success(f"x = {_fmt(x_val, decimal)} for P(X ≤ x) = {p}")
            st.markdown(
                rf"""
**🧮 Step-by-step:**
1. Solve \( \Phi\!\left(\frac{{x-\mu}}{{\sigma}}\right) = p \Rightarrow x = \mu + \sigma\,\Phi^{{-1}}(p) \)
2. \( x = {mean} + {sd}\cdot \Phi^{{-1}}({p}) = {_fmt(x_val, decimal)} \)
3. **Final answer:** \( x = {_fmt(x_val, decimal)} \)
                """
            )
            ax.fill_between(x, 0, y, where=(x <= x_val), alpha=0.6)
            ax.axvline(x_val, linestyle="--")

        elif tail == "Right tail":
            x_val = norm.ppf(1 - p, mean, sd)
            st.success(f"x = {_fmt(x_val, decimal)} for P(X ≥ x) = {p}")
            st.markdown(
                rf"""
**🧮 Step-by-step:**
1. \( P(X\ge x)=p \Rightarrow \Phi\!\left(\frac{{x-\mu}}{{\sigma}}\right)=1-p \Rightarrow x=\mu+\sigma\,\Phi^{{-1}}(1-p) \)
2. \( x = {mean} + {sd}\cdot \Phi^{{-1}}(1-{p}) = {_fmt(x_val, decimal)} \)
3. **Final answer:** \( x = {_fmt(x_val, decimal)} \)
                """
            )
            ax.fill_between(x, 0, y, where=(x >= x_val), alpha=0.6)
            ax.axvline(x_val, linestyle="--")

        else:  # Middle area
            tail_prob = (1 - p) / 2
            a_val = norm.ppf(tail_prob, mean, sd)
            b_val = norm.ppf(1 - tail_prob, mean, sd)
            st.success(f"{p*100:.1f}% of data lies between {_fmt(a_val, decimal)} and {_fmt(b_val, decimal)}")
            st.markdown(
                rf"""
**🧮 Step-by-step:**
1. Middle area \(= p\Rightarrow\) tails each \(= \frac{{1-p}}{2}\)
2. \( a = \mu + \sigma\,\Phi^{{-1}}\!\left(\tfrac{{1-p}}{2}\right),\quad b = \mu + \sigma\,\Phi^{{-1}}\!\left(1-\tfrac{{1-p}}{2}\right) \)
3. **Final answer:** \( [{_fmt(a_val, decimal)},\, {_fmt(b_val, decimal)}] \)
                """
            )
            ax.fill_between(x, 0, y, where=(x >= a_val) & (x <= b_val), alpha=0.6)
            ax.axvline(a_val, linestyle="--")
            ax.axvline(b_val, linestyle="--")

        st.pyplot(fig)


# ==========================================================
# Sampling Distribution of the Mean
# ==========================================================
def sampling_mean(decimal):
    st.markdown("### 📘 **Sampling Distribution of the Mean**")
    st.latex(r"\mu_{\bar{X}} = \mu, \quad \sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}")
    st.latex(r"Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}}")

    mu_expr = st.text_input("Population mean (μ):", value="0", key="smean_mu_expr")
    sigma_expr = st.text_input("Population SD (σ):", value="1", key="smean_sigma_expr")
    n = st.number_input("Sample size (n):", min_value=1, step=1, value=30, key="smean_n")

    mu = parse_expression(mu_expr)
    sigma = parse_expression(sigma_expr)
    if mu is None or sigma is None:
        return

    se = sigma / math.sqrt(n)
    st.write(f"**Standard Error (σₓ̄) =** {_fmt(se, decimal)}")

    calc_type = st.selectbox("Choose a calculation:", [
        "P(X̄ < x)",
        "P(X̄ > x)",
        "P(a < X̄ < b)",
        "Inverse: Find x for given probability"
    ], key="smean_calc_type")

    x = np.linspace(mu - 4*se, mu + 4*se, 500)
    y = norm.pdf(x, mu, se)

    # P(X̄ < x)
    if calc_type == "P(X̄ < x)":
        x_val = st.number_input("Enter x value (for X̄):", value=float(mu), key="smean_x_lt")
        z = (x_val - mu) / se
        prob = norm.cdf(z)
        st.success(f"P(X̄ ≤ {x_val}) = {_fmt(prob, decimal)}")

        st.markdown(
            rf"""
**🧮 Step-by-step:**
1. Compute \( \sigma_{{\bar X}} = \frac{{\sigma}}{{\sqrt{{n}}}} = {_fmt(se, decimal)} \)
2. \( Z = \frac{{\bar x - \mu}}{{\sigma_{{\bar X}}}} = \frac{{{x_val} - {mu}}}{{{_fmt(se, decimal)}}} = {_fmt(z, decimal)} \)
3. \( P(\bar X < {x_val}) = P(Z < {_fmt(z, decimal)}) = {_fmt(prob, decimal)} \)
            """
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x <= x_val), alpha=0.6)
        ax.axvline(x_val, linestyle="--")
        st.pyplot(fig)

    # P(X̄ > x)
    elif calc_type == "P(X̄ > x)":
        x_val = st.number_input("Enter x value (for X̄):", value=float(mu), key="smean_x_gt")
        z = (x_val - mu) / se
        prob = 1 - norm.cdf(z)
        st.success(f"P(X̄ ≥ {x_val}) = {_fmt(prob, decimal)}")

        st.markdown(
            rf"""
**🧮 Step-by-step:**
1. \( \sigma_{{\bar X}} = {_fmt(se, decimal)} \)
2. \( Z = \frac{{{x_val} - {mu}}}{{{_fmt(se, decimal)}}} = {_fmt(z, decimal)} \)
3. \( P(\bar X > {x_val}) = P(Z > {_fmt(z, decimal)}) = {_fmt(prob, decimal)} \)
            """
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= x_val), alpha=0.6)
        ax.axvline(x_val, linestyle="--")
        st.pyplot(fig)

    # P(a < X̄ < b)
    elif calc_type == "P(a < X̄ < b)":
        a_val = st.number_input("Lower bound (a) for X̄:", value=float(mu - se), key="smean_between_a")
        b_val = st.number_input("Upper bound (b) for X̄:", value=float(mu + se), key="smean_between_b")
        if b_val <= a_val:
            st.error("Upper bound must be greater than lower bound.")
            return
        z1 = (a_val - mu) / se
        z2 = (b_val - mu) / se
        prob = norm.cdf(z2) - norm.cdf(z1)
        st.success(f"P({a_val} < X̄ < {b_val}) = {_fmt(prob, decimal)}")

        st.markdown(
            rf"""
**🧮 Step-by-step:**
1. \( \sigma_{{\bar X}} = {_fmt(se, decimal)} \)
2. \( Z_1 = \frac{{{a_val} - {mu}}}{{{_fmt(se, decimal)}}} = {_fmt(z1, decimal)},\quad
   Z_2 = \frac{{{b_val} - {mu}}}{{{_fmt(se, decimal)}}} = {_fmt(z2, decimal)} \)
3. \( P({a_val} < \bar X < {b_val}) = \Phi({_fmt(z2, decimal)}) - \Phi({_fmt(z1, decimal)}) = {_fmt(prob, decimal)} \)
            """
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= a_val) & (x <= b_val), alpha=0.6)
        ax.axvline(a_val, linestyle="--")
        ax.axvline(b_val, linestyle="--")
        st.pyplot(fig)

    # Inverse
    elif calc_type == "Inverse: Find x for given probability":
        p = st.number_input("Enter cumulative probability p = P(X̄ < x):",
                            min_value=0.0, max_value=1.0, value=0.95, key="smean_inv_p")
        x_val = norm.ppf(p, loc=mu, scale=se)
        st.success(f"x̄ = {_fmt(x_val, decimal)} for P(X̄ < x̄) = {p}")

        st.markdown(
            rf"""
**🧮 Step-by-step:**
1. Solve \( \Phi\!\left(\frac{{x-\mu}}{{\sigma_{{\bar X}}}}\right) = p \Rightarrow x = \mu + \sigma_{{\bar X}}\Phi^{{-1}}(p) \)
2. \( \sigma_{{\bar X}} = {_fmt(se, decimal)},\; x = {mu} + {_fmt(se, decimal)}\Phi^{{-1}}({p}) = {_fmt(x_val, decimal)} \)
3. **Final answer:** \( x = {_fmt(x_val, decimal)} \)
            """
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x <= x_val), alpha=0.6)
        ax.axvline(x_val, linestyle="--")
        st.pyplot(fig)


# ==========================================================
# Uniform Distribution (with unique keys & step-by-step)
# ==========================================================
def uniform_distribution(decimal):
    st.markdown("### 🎲 **Uniform Distribution**")
    st.latex(r"f(x) = \frac{1}{b - a}, \quad a \le x \le b")

    a = st.number_input("Lower bound (a):", value=0.0, key="ua_main")
    b = st.number_input("Upper bound (b):", value=10.0, key="ub_main")
    if b <= a:
        st.error("⚠️ The upper bound (b) must be greater than the lower bound (a).")
        return

    pdf = 1 / (b - a)
    st.write(f"**Constant PDF:** f(x) = {_fmt(pdf, decimal)} for {a} ≤ x ≤ {b}")

    calc_type = st.selectbox("Choose a calculation:", [
        "P(X < x) = P(X ≤ x)",
        "P(X = x)",
        "P(X > x) = P(X ≥ x)",
        "P(a < X < b)",
        "Inverse: Find x for given probability"
    ], key="uniform_calc_type")

    x = np.linspace(a - (b - a)*0.2, b + (b - a)*0.2, 500)
    y = np.where((x >= a) & (x <= b), pdf, 0)

    # P(X < x) = P(X ≤ x)
    if calc_type == "P(X < x) = P(X ≤ x)":
        x_val = st.number_input("Enter x value:", value=(a + b) / 2, key="ux_less")
        if x_val <= a:
            prob = 0.0
        elif x_val >= b:
            prob = 1.0
        else:
            prob = (x_val - a) / (b - a)
        st.success(f"P(X ≤ {x_val}) = {_fmt(prob, decimal)}")

        st.markdown(
            rf"""
**🧮 Step-by-step:**
1. For Uniform(a,b), \( P(X\le x) = \frac{{x-a}}{{b-a}} \) when \( a \le x \le b \)
2. \( P(X\le {x_val}) = \frac{{{x_val}-{a}}}{{{b}-{a}}} = {_fmt(prob, decimal)} \)
3. **Final answer:** \( P(X\le {x_val}) = {_fmt(prob, decimal)} \)
            """
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x <= x_val) & (x >= a), alpha=0.6)
        ax.axvline(x_val, linestyle="--")
        st.pyplot(fig)

    # P(X = x)
    elif calc_type == "P(X = x)":
        x_val = st.number_input("Enter the point x:", value=(a + b) / 2, key="ux_equal")
        st.success(f"P(X = {x_val}) = 0 (continuous distribution)")

        st.markdown(
            r"""
**🧮 Step-by-step:**
1. For continuous distributions, \(P(X = x) = 0\)
2. (A point has zero width → zero area under the PDF.)
3. **Final answer:** \(P(X = x) = 0\)
            """
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.axvline(x_val, linestyle="--")
        st.pyplot(fig)

    # P(X > x) = P(X ≥ x)
    elif calc_type == "P(X > x) = P(X ≥ x)":
        x_val = st.number_input("Enter x value:", value=(a + b) / 2, key="ux_greater")
        if x_val <= a:
            prob = 1.0
        elif x_val >= b:
            prob = 0.0
        else:
            prob = (b - x_val) / (b - a)
        st.success(f"P(X ≥ {x_val}) = {_fmt(prob, decimal)}")

        st.markdown(
            rf"""
**🧮 Step-by-step:**
1. \( P(X\ge x) = 1 - P(X\le x) = 1 - \frac{{x-a}}{{b-a}} = \frac{{b-x}}{{b-a}} \)
2. \( P(X\ge {x_val}) = \frac{{{b}-{x_val}}}{{{b}-{a}}} = {_fmt(prob, decimal)} \)
3. **Final answer:** \( P(X\ge {x_val}) = {_fmt(prob, decimal)} \)
            """
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= x_val) & (x <= b), alpha=0.6)
        ax.axvline(x_val, linestyle="--")
        st.pyplot(fig)

    # P(a < X < b)
    elif calc_type == "P(a < X < b)":
        low = st.number_input("Lower bound (a):", value=a, key="ua_inner")
        high = st.number_input("Upper bound (b):", value=b, key="ub_inner")
        if high <= low:
            st.error("Upper bound must be greater than lower bound.")
            return

        if high < a or low > b:
            prob = 0.0
            lower, upper = low, high
        else:
            lower = max(low, a)
            upper = min(high, b)
            prob = (upper - lower) / (b - a)

        st.success(f"P({low} < X < {high}) = {_fmt(prob, decimal)}")

        st.markdown(
            rf"""
**🧮 Step-by-step:**
1. Overlap with \([{a},{b}]\) is \([{_fmt(lower, decimal)},{_fmt(upper, decimal)}]\)
2. \( P({low}<X<{high}) = \frac{{\text{{length of overlap}}}}{{b-a}} = \frac{{{_fmt(upper, decimal)}-{_fmt(lower, decimal)}}}{{{b}-{a}}} = {_fmt(prob, decimal)} \)
3. **Final answer:** \( P({low}<X<{high}) = {_fmt(prob, decimal)} \)
            """
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= lower) & (x <= upper), alpha=0.6)
        ax.axvline(low, linestyle="--")
        ax.axvline(high, linestyle="--")
        st.pyplot(fig)

    # Inverse: Find x for given probability
    elif calc_type == "Inverse: Find x for given probability":
        p = st.number_input("Enter probability p for P(X ≤ x) = p (0 < p < 1):",
                            min_value=0.0, max_value=1.0, value=0.5, key="u_inverse_p")
        x_val = a + p * (b - a)
        st.success(f"x = {_fmt(x_val, decimal)} for P(X ≤ x) = {p}")

        st.markdown(
            rf"""
**🧮 Step-by-step:**
1. \( P(X\le x) = \frac{{x-a}}{{b-a}} = p \Rightarrow x = a + p(b-a) \)
2. \( x = {a} + {p}({b}-{a}) = {_fmt(x_val, decimal)} \)
3. **Final answer:** \( x = {_fmt(x_val, decimal)} \)
            """
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x <= x_val) & (x >= a), alpha=0.6)
        ax.axvline(x_val, linestyle="--")
        st.pyplot(fig)


# ==========================================================
# Sampling Distribution of the Proportion
# ==========================================================
def sampling_proportion(decimal):
    st.markdown("### 📗 **Sampling Distribution of the Proportion**")
    st.latex(r"\mu_{\hat{p}} = p, \quad \sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}")
    st.latex(r"Z = \frac{\hat{p} - p}{\sqrt{p(1-p)/n}}")

    p_expr = st.text_input("Population proportion (p):", value="0.5", key="sprop_p_expr")
    n = st.number_input("Sample size (n):", min_value=1, step=1, value=30, key="sprop_n")

    p = parse_expression(p_expr)
    if p is None or not (0 < p < 1):
        st.error("p must be between 0 and 1.")
        return

    q = 1 - p
    se = math.sqrt(p * q / n)
    st.write(f"**Standard Error (σₚ̂) =** {_fmt(se, decimal)}")

    calc_type = st.selectbox("Choose a calculation:", [
        "P(p̂ < x)",
        "P(p̂ > x)",
        "P(a < p̂ < b)",
        "Inverse: Find x for given probability"
    ], key="sprop_calc_type")

    x = np.linspace(p - 4*se, p + 4*se, 500)
    y = norm.pdf(x, p, se)

    # P(p̂ < x)
    if calc_type == "P(p̂ < x)":
        xhat = st.number_input("Enter x value (for p̂):", min_value=0.0, max_value=1.0, value=float(p), key="sprop_x_lt")
        z = (xhat - p) / se
        prob = norm.cdf(z)
        st.success(f"P(p̂ ≤ {xhat}) = {_fmt(prob, decimal)}")

        st.markdown(
            rf"""
**🧮 Step-by-step:**
1. \( \sigma_{{\hat p}} = \sqrt{{\frac{{p(1-p)}}{{n}}}} = {_fmt(se, decimal)} \)
2. \( Z = \frac{{\hat p - p}}{{\sigma_{{\hat p}}}} = \frac{{{xhat} - {p}}}{{{_fmt(se, decimal)}}} = {_fmt(z, decimal)} \)
3. \( P(\hat p < {xhat}) = P(Z < {_fmt(z, decimal)}) = {_fmt(prob, decimal)} \)
            """
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x <= xhat), alpha=0.6)
        ax.axvline(xhat, linestyle="--")
        st.pyplot(fig)

    # P(p̂ > x)
    elif calc_type == "P(p̂ > x)":
        xhat = st.number_input("Enter x value (for p̂):", min_value=0.0, max_value=1.0, value=float(p), key="sprop_x_gt")
        z = (xhat - p) / se
        prob = 1 - norm.cdf(z)
        st.success(f"P(p̂ ≥ {xhat}) = {_fmt(prob, decimal)}")

        st.markdown(
            rf"""
**🧮 Step-by-step:**
1. \( \sigma_{{\hat p}} = {_fmt(se, decimal)} \)
2. \( Z = \frac{{{xhat} - {p}}}{{{_fmt(se, decimal)}}} = {_fmt(z, decimal)} \)
3. \( P(\hat p > {xhat}) = P(Z > {_fmt(z, decimal)}) = {_fmt(prob, decimal)} \)
            """
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= xhat), alpha=0.6)
        ax.axvline(xhat, linestyle="--")
        st.pyplot(fig)

    # P(a < p̂ < b)
    elif calc_type == "P(a < p̂ < b)":
        a_val = st.number_input("Lower bound (a) for p̂:", min_value=0.0, max_value=1.0, value=max(0.0, float(p - se)), key="sprop_between_a")
        b_val = st.number_input("Upper bound (b) for p̂:", min_value=0.0, max_value=1.0, value=min(1.0, float(p + se)), key="sprop_between_b")
        if b_val <= a_val:
            st.error("Upper bound must be greater than lower bound.")
            return
        z1 = (a_val - p) / se
        z2 = (b_val - p) / se
        prob = norm.cdf(z2) - norm.cdf(z1)
        st.success(f"P({a_val} < p̂ < {b_val}) = {_fmt(prob, decimal)}")

        st.markdown(
            rf"""
**🧮 Step-by-step:**
1. \( \sigma_{{\hat p}} = {_fmt(se, decimal)} \)
2. \( Z_1 = \frac{{{a_val} - {p}}}{{{_fmt(se, decimal)}}} = {_fmt(z1, decimal)},\quad
   Z_2 = \frac{{{b_val} - {p}}}{{{_fmt(se, decimal)}}} = {_fmt(z2, decimal)} \)
3. \( P({a_val}<\hat p<{b_val}) = \Phi({_fmt(z2, decimal)}) - \Phi({_fmt(z1, decimal)}) = {_fmt(prob, decimal)} \)
            """
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x >= a_val) & (x <= b_val), alpha=0.6)
        ax.axvline(a_val, linestyle="--")
        ax.axvline(b_val, linestyle="--")
        st.pyplot(fig)

    # Inverse
    elif calc_type == "Inverse: Find x for given probability":
        p_val = st.number_input("Enter p = P(p̂ < x):", min_value=0.0, max_value=1.0, value=0.95, key="sprop_inv_p")
        xhat = norm.ppf(p_val, loc=p, scale=se)
        st.success(f"p̂ = {_fmt(xhat, decimal)} for P(p̂ < x) = {p_val}")

        st.markdown(
            rf"""
**🧮 Step-by-step:**
1. Solve \( \Phi\!\left(\frac{{x-p}}{{\sigma_{{\hat p}}}}\right)=p_0 \Rightarrow x = p + \sigma_{{\hat p}}\Phi^{{-1}}(p_0) \)
2. \( \sigma_{{\hat p}} = {_fmt(se, decimal)},\; x = {p} + {_fmt(se, decimal)}\Phi^{{-1}}({p_val}) = {_fmt(xhat, decimal)} \)
3. **Final answer:** \( x = {_fmt(xhat, decimal)} \)
            """
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.fill_between(x, 0, y, where=(x <= xhat), alpha=0.6)
        ax.axvline(xhat, linestyle="--")
        st.pyplot(fig)


# ==========================================================
# MAIN APP
# ==========================================================
def run():
    st.header("🧠 MIND: Continuous Probability Distributions")

    dist_choice = st.radio(
        "Select Distribution Type:",
        ["Uniform Distribution", "Normal Distribution", "Sampling Distribution of the Mean", "Sampling Distribution of the Proportion"],
        horizontal=True,
        key="main_dist_choice"
    )

    decimal = st.number_input("Decimal places for output:", min_value=0, max_value=10, value=4, step=1, key="global_decimals")

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
