import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import re
from scipy.stats import norm

# ==========================================================
# Safe Expression Parsing
# ==========================================================
def parse_expression(expr):
    expr = expr.strip().lower()
    try:
        if "sqrt" in expr:
            num = re.findall(r"sqrt\((.*?)\)", expr)
            if num:
                return math.sqrt(float(num[0]))
            else:
                raise ValueError("Invalid sqrt format.")
        elif "/" in expr:
            parts = expr.split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
            else:
                raise ValueError("Invalid fraction format.")
        else:
            return float(expr)
    except Exception as e:
        st.error(f"Invalid input: {e}")
        return None


# ==========================================================
# Plot Helper (Normal/Sampling)
# ==========================================================
def plot_distribution(x, y, mean, sd, a_val=None, b_val=None, x_val=None, calc_type=None, title="Distribution"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color="blue", lw=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True)

    if calc_type in ["P(X < x)", "P(p̂ < x)"] and x_val is not None:
        ax.fill_between(x, 0, y, where=(x <= x_val), color="skyblue", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
    elif calc_type in ["P(X > x)", "P(p̂ > x)"] and x_val is not None:
        ax.fill_between(x, 0, y, where=(x >= x_val), color="lightgreen", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
    elif calc_type in ["P(a < X < b)", "P(a < p̂ < b)"] and a_val is not None and b_val is not None:
        ax.fill_between(x, 0, y, where=(x >= a_val) & (x <= b_val), color="orange", alpha=0.6)
        ax.axvline(a_val, color="red", linestyle="--")
        ax.axvline(b_val, color="red", linestyle="--")

    st.pyplot(fig)


# ==========================================================
# Uniform Distribution (Final, Non-Duplicate Version)
# ==========================================================
def uniform_distribution_tool():
    st.subheader("📏 Uniform Distribution (a ≤ X ≤ b)")
    st.latex(r"f(x) = \frac{1}{b - a}, \quad a \le x \le b")
    st.latex(r"P(X < x) = P(X \le x)")

    # --- Inputs with Unique Keys ---
    a = st.number_input("Enter minimum value (a):", value=0.0, key="a_min")
    b = st.number_input("Enter maximum value (b):", value=10.0, key="b_max")
    if b <= a:
        st.error("⚠️ The upper bound (b) must be greater than the lower bound (a).")
        return

    decimal = st.number_input("Decimal places for results:", min_value=0, max_value=10, value=4, key="decimals_uni")
    rp = lambda v: round(v, decimal)

    pdf_value = 1 / (b - a)
    st.write(f"**Constant PDF:** f(x) = 1/({b} - {a}) = **{rp(pdf_value)}**")

    calc_type = st.selectbox(
        "Choose a probability calculation:",
        ["P(X ≤ x)", "P(X ≥ x)", "P(a < X < b)", "Find x for a given probability"],
        index=0,
        key="calc_type_uni"
    )
    show_steps = st.checkbox("📖 Show Step-by-Step Solution", key="steps_uni")

    x = np.linspace(a - (b - a) * 0.2, b + (b - a) * 0.2, 500)
    y = np.where((x >= a) & (x <= b), pdf_value, 0)

    # ---------- P(X ≤ x) ----------
    if calc_type == "P(X ≤ x)":
        x_val = st.number_input("Enter x value:", value=a + (b - a) / 2, key="x_val_le")
        if st.button("Calculate", key="calc_le"):
            prob = 0 if x_val <= a else 1 if x_val >= b else (x_val - a) / (b - a)
            st.success(f"P(X ≤ {x_val}) = {rp(prob)}")
            if show_steps:
                st.latex(r"P(X \le x) = \frac{x - a}{b - a}")
                st.write(f"Substitute: ({x_val} - {a}) / ({b} - {a}) = {rp(prob)}")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, y, color="blue", lw=2)
            ax.fill_between(x, 0, y, where=(x >= a) & (x <= x_val), color="skyblue", alpha=0.6)
            ax.axvline(x_val, color="red", linestyle="--")
            ax.set_title("Uniform Distribution")
            ax.set_xlabel("X")
            ax.set_ylabel("f(x)")
            st.pyplot(fig)

    # ---------- P(X ≥ x) ----------
    elif calc_type == "P(X ≥ x)":
        x_val = st.number_input("Enter x value:", value=a + (b - a) / 2, key="x_val_ge")
        if st.button("Calculate", key="calc_ge"):
            prob = 1 if x_val <= a else 0 if x_val >= b else (b - x_val) / (b - a)
            st.success(f"P(X ≥ {x_val}) = {rp(prob)}")
            if show_steps:
                st.latex(r"P(X \ge x) = \frac{b - x}{b - a}")
                st.write(f"Substitute: ({b} - {x_val}) / ({b} - {a}) = {rp(prob)}")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, y, color="blue", lw=2)
            ax.fill_between(x, 0, y, where=(x >= x_val) & (x <= b), color="lightgreen", alpha=0.6)
            ax.axvline(x_val, color="red", linestyle="--")
            ax.set_title("Uniform Distribution")
            ax.set_xlabel("X")
            ax.set_ylabel("f(x)")
            st.pyplot(fig)

    # ---------- P(a < X < b) ----------
    elif calc_type == "P(a < X < b)":
        a1 = st.number_input("Lower bound (a₁):", value=a + (b - a) / 4, key="a1")
        b1 = st.number_input("Upper bound (b₁):", value=a + (b - a) * 3 / 4, key="b1")
        if st.button("Calculate", key="calc_between"):
            a1, b1 = max(a1, a), min(b1, b)
            if a1 >= b1:
                st.error("⚠️ Lower bound must be less than upper bound.")
                return
            prob = (b1 - a1) / (b - a)
            st.success(f"P({a1} < X < {b1}) = {rp(prob)}")
            if show_steps:
                st.latex(r"P(a < X < b) = \frac{b_1 - a_1}{b - a}")
                st.write(f"Substitute: ({b1} - {a1}) / ({b} - {a}) = {rp(prob)}")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, y, color="blue", lw=2)
            ax.fill_between(x, 0, y, where=(x >= a1) & (x <= b1), color="orange", alpha=0.6)
            ax.axvline(a1, color="red", linestyle="--")
            ax.axvline(b1, color="red", linestyle="--")
            ax.set_title("Uniform Distribution")
            ax.set_xlabel("X")
            ax.set_ylabel("f(x)")
            st.pyplot(fig)

    # ---------- Inverse ----------
    elif calc_type == "Find x for a given probability":
        p = st.number_input("Enter probability:", min_value=0.0, max_value=1.0, value=0.5, key="prob_input")
        tail = st.selectbox("Tail:", ["Left tail: P(X ≤ x) = p", "Right tail: P(X ≥ x) = p"], key="tail_choice")
        if st.button("Calculate", key="calc_inverse"):
            if tail.startswith("Left"):
                x_val = a + p * (b - a)
                st.success(f"x = {rp(x_val)} for P(X ≤ x) = {p}")
            else:
                x_val = b - p * (b - a)
                st.success(f"x = {rp(x_val)} for P(X ≥ x) = {p}")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, y, color="blue", lw=2)
            if tail.startswith("Left"):
                ax.fill_between(x, 0, y, where=(x >= a) & (x <= x_val), color="skyblue", alpha=0.6)
            else:
                ax.fill_between(x, 0, y, where=(x >= x_val) & (x <= b), color="lightgreen", alpha=0.6)
            ax.axvline(x_val, color="red", linestyle="--")
            ax.set_title("Uniform Distribution")
            ax.set_xlabel("X")
            ax.set_ylabel("f(x)")
            st.pyplot(fig)


# ==========================================================
# MAIN APP
# ==========================================================
def run():
    st.header("📊 Normal, Sampling & Uniform Distribution Calculator")

    dist_type = st.selectbox(
        "Select a distribution type:",
        [
            "Regular Normal Distribution",
            "Sampling Distribution of the Mean",
            "Sampling Distribution of the Proportion",
            "Uniform Distribution"
        ],
        index=None,
        placeholder="Select to begin..."
    )

    if not dist_type:
        st.info("👆 Please select a distribution to begin.")
        return

    decimal = st.number_input("Decimal places:", min_value=0, max_value=10, value=4, key="decimals_main")
    rp = lambda v: round(v, decimal)

    # ---------- Normal Distribution ----------
    if dist_type == "Regular Normal Distribution":
        st.subheader("📈 Normal Distribution (X ~ N(μ, σ))")
        st.latex(r"Z = \frac{X - \mu}{\sigma}")
        mean = st.number_input("Population mean (μ):", value=0.0, key="mean_norm")
        sd = st.number_input("Standard deviation (σ):", min_value=0.0001, value=1.0, key="sd_norm")
        x_val = st.number_input("Enter x:", value=0.0, key="x_norm")
        z = (x_val - mean) / sd
        st.success(f"Z = ({x_val} - {mean}) / {sd} = {rp(z)}")

    # ---------- Sampling Distribution of the Mean ----------
    elif dist_type == "Sampling Distribution of the Mean":
        st.subheader("📘 Sampling Distribution of the Sample Mean")
        st.latex(r"Z = \frac{\bar{X} - \mu_{\bar{X}}}{\sigma_{\bar{X}}}, \quad \mu_{\bar{X}} = \mu, \ \sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}")
        mean_expr = st.text_input("Population mean (μ):", value="0", key="mean_xbar")
        sd_expr = st.text_input("Population SD (σ):", value="1", key="sd_xbar")
        n = st.number_input("Sample size (n):", min_value=1, step=1, value=30, key="n_xbar")
        mean = parse_expression(mean_expr)
        sd = parse_expression(sd_expr)
        if mean is None or sd is None:
            return
        se = sd / math.sqrt(n)
        st.write(f"**Std. Error:** {rp(se)}")

    # ---------- Sampling Distribution of the Proportion ----------
    elif dist_type == "Sampling Distribution of the Proportion":
        st.subheader("📘 Sampling Distribution of the Sample Proportion")
        st.latex(r"Z = \frac{\hat{p} - \mu_{\hat{p}}}{\sigma_{\hat{p}}}, \quad \mu_{\hat{p}} = p, \ \sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}")
        p_expr = st.text_input("Population proportion (p):", value="0.5", key="p_hat")
        n = st.number_input("Sample size (n):", min_value=1, step=1, value=30, key="n_phat")
        p = parse_expression(p_expr)
        if p is None or not (0 < p < 1):
            st.error("p must be between 0 and 1.")
            return
        q = 1 - p
        se = math.sqrt((p * q) / n)
        st.write(f"**Std. Error:** {rp(se)}")

    # ---------- Uniform Distribution ----------
    elif dist_type == "Uniform Distribution":
        uniform_distribution_tool()


# ==========================================================
# Run App
# ==========================================================
if __name__ == "__main__":
    run()
