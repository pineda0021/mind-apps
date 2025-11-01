import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import re
from scipy.stats import norm

# ---------- Safe Expression Parsing ----------
def parse_expression(expr):
    """Safely evaluate simple expressions like 120/200 or sqrt(25)."""
    expr = expr.strip().lower()
    try:
        if "sqrt" in expr:
            num = re.findall(r"sqrt\((.*?)\)", expr)
            if num:
                return math.sqrt(float(num[0]))
            else:
                raise ValueError("Invalid sqrt format. Use sqrt( ) properly.")
        elif "/" in expr:
            parts = expr.split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
            else:
                raise ValueError("Invalid fraction format. Use a/b format.")
        else:
            return float(expr)
    except Exception as e:
        st.error(f"Invalid input: {e}")
        return None


# ---------- Plot Helper ----------
def plot_distribution(x, y, mean, sd, a_val=None, b_val=None, x_val=None, calc_type=None, title="Distribution"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color="blue", lw=2, label="Distribution")
    ax.set_xlabel("X")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True)

    # Highlight region
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


# ---------- Main App ----------
def run():
    st.header("📊 Normal & Sampling Distribution Calculator")

    st.markdown("""
    Choose a distribution type:
    1️⃣ **Regular Normal Distribution (X ~ N(μ, σ))**  
    2️⃣ **Sampling Distribution of the Mean (μₓ̄)**  
    3️⃣ **Sampling Distribution of the Proportion (p̂)**
    """)

    dist_type = st.selectbox(
        "Select a distribution type:",
        ["Regular Normal Distribution", "Sampling Distribution of the Mean", "Sampling Distribution of the Proportion"],
        index=None,
        placeholder="Select to begin..."
    )

    if not dist_type:
        st.info("👆 Please choose a distribution type to begin.")
        return

    decimal = st.number_input("Decimal places for output:", min_value=0, max_value=10, value=4, step=1)
    rp = lambda v: round(v, decimal)

    # ----------------------------------------------------------
    # 1. REGULAR NORMAL DISTRIBUTION
    # ----------------------------------------------------------
    if dist_type == "Regular Normal Distribution":
        st.subheader("📈 Normal Distribution (X ~ N(μ, σ))")

        mean = st.number_input("Population mean (μ):", value=0.0)
        sd = st.number_input("Standard deviation (σ):", min_value=0.0001, value=1.0)

        calc_type = st.selectbox("Choose calculation:", [
            "P(X < x)",
            "P(X > x)",
            "P(a < X < b)",
            "Inverse: Find x for given probability"
        ])

        show_steps = st.checkbox("📖 Show Step-by-Step Solution")

        if calc_type == "P(X < x)":
            x_val = st.number_input("Enter x value:", value=0.0)
            if st.button("Calculate"):
                prob = norm.cdf(x_val, mean, sd)
                z = (x_val - mean) / sd
                st.success(f"P(X < {x_val}) = {rp(prob)}")
                if show_steps:
                    st.write(f"Z = ({x_val} - {mean}) / {sd} = {rp(z)}")
                x = np.linspace(mean - 4 * sd, mean + 4 * sd, 1000)
                plot_distribution(x, norm.pdf(x, mean, sd), mean, sd, x_val=x_val, calc_type="P(X < x)", title="Normal Distribution")

        elif calc_type == "P(X > x)":
            x_val = st.number_input("Enter x value:", value=0.0)
            if st.button("Calculate"):
                prob = 1 - norm.cdf(x_val, mean, sd)
                z = (x_val - mean) / sd
                st.success(f"P(X > {x_val}) = {rp(prob)}")
                if show_steps:
                    st.write(f"Z = ({x_val} - {mean}) / {sd} = {rp(z)}")
                x = np.linspace(mean - 4 * sd, mean + 4 * sd, 1000)
                plot_distribution(x, norm.pdf(x, mean, sd), mean, sd, x_val=x_val, calc_type="P(X > x)", title="Normal Distribution")

        elif calc_type == "P(a < X < b)":
            a = st.number_input("Lower bound (a):", value=-1.0)
            b = st.number_input("Upper bound (b):", value=1.0)
            if st.button("Calculate"):
                prob = norm.cdf(b, mean, sd) - norm.cdf(a, mean, sd)
                z1 = (a - mean) / sd
                z2 = (b - mean) / sd
                st.success(f"P({a} < X < {b}) = {rp(prob)}")
                if show_steps:
                    st.write(f"Z₁ = {rp(z1)}, Z₂ = {rp(z2)} → P(Z₁ < Z < Z₂) = {rp(prob)}")
                x = np.linspace(mean - 4 * sd, mean + 4 * sd, 1000)
                plot_distribution(x, norm.pdf(x, mean, sd), mean, sd, a_val=a, b_val=b, calc_type="P(a < X < b)", title="Normal Distribution")

        elif calc_type == "Inverse: Find x for given probability":
            tail = st.selectbox("Select tail:", ["Left tail", "Right tail", "Middle area"])
            p = st.number_input("Enter probability (0 < p < 1):", min_value=0.0, max_value=1.0, value=0.95)
            if st.button("Calculate"):
                if tail == "Left tail":
                    x_val = norm.ppf(p, mean, sd)
                    st.success(f"x = {rp(x_val)} for P(X < x) = {p}")
                elif tail == "Right tail":
                    x_val = norm.ppf(1 - p, mean, sd)
                    st.success(f"x = {rp(x_val)} for P(X > x) = {p}")
                else:
                    tail_prob = (1 - p) / 2
                    a = norm.ppf(tail_prob, mean, sd)
                    b = norm.ppf(1 - tail_prob, mean, sd)
                    st.success(f"{p*100:.1f}% of data lies between {rp(a)} and {rp(b)}")
                    x_val = None
                x = np.linspace(mean - 4 * sd, mean + 4 * sd, 1000)
                plot_distribution(x, norm.pdf(x, mean, sd), mean, sd, a_val=a if tail=="Middle area" else None,
                                  b_val=b if tail=="Middle area" else None, x_val=x_val,
                                  calc_type="P(a < X < b)" if tail=="Middle area" else "P(X < x)", title="Normal Distribution")

    # ----------------------------------------------------------
    # 2. SAMPLING DISTRIBUTION OF THE MEAN
    # ----------------------------------------------------------
    elif dist_type == "Sampling Distribution of the Mean":
        st.subheader("📘 Sampling Distribution of the Sample Mean")
        mean_expr = st.text_input("Population mean (μ):", value="0")
        sd_expr = st.text_input("Population standard deviation (σ):", value="1")
        n = st.number_input("Sample size (n):", min_value=1, step=1, value=30)

        mean = parse_expression(mean_expr)
        sd = parse_expression(sd_expr)
        if mean is None or sd is None:
            return

        sample_mean = mean
        sample_sd = sd / math.sqrt(n)

        st.latex(r"\mu_{\bar{x}} = \mu, \quad \sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}}")
        st.write(f"**Mean:** {rp(sample_mean)}  |  **Std. Error:** {rp(sample_sd)}")

        st.markdown("---")
        st.write("Choose a calculation:")
        option = st.selectbox("", ["P(X̄ < x)", "P(X̄ > x)", "P(a < X̄ < b)", "Inverse: Find x given probability"])
        show_steps = st.checkbox("📖 Show Step-by-Step Solution")

        if option == "P(X̄ < x)":
            x_val = st.number_input("Enter x̄ value:", value=sample_mean)
            if st.button("Calculate"):
                prob = norm.cdf(x_val, sample_mean, sample_sd)
                z = (x_val - sample_mean) / sample_sd
                st.success(f"P(X̄ < {x_val}) = {rp(prob)}")
                if show_steps: st.write(f"Z = ({x_val}-{sample_mean})/{sample_sd} = {rp(z)}")
                x = np.linspace(sample_mean - 4*sample_sd, sample_mean + 4*sample_sd, 1000)
                plot_distribution(x, norm.pdf(x, sample_mean, sample_sd), sample_mean, sample_sd,
                                  x_val=x_val, calc_type="P(X < x)", title="Sampling Distribution of the Mean")

        elif option == "P(X̄ > x)":
            x_val = st.number_input("Enter x̄ value:", value=sample_mean)
            if st.button("Calculate"):
                prob = 1 - norm.cdf(x_val, sample_mean, sample_sd)
                z = (x_val - sample_mean) / sample_sd
                st.success(f"P(X̄ > {x_val}) = {rp(prob)}")
                if show_steps: st.write(f"Z = {rp(z)}")
                x = np.linspace(sample_mean - 4*sample_sd, sample_mean + 4*sample_sd, 1000)
                plot_distribution(x, norm.pdf(x, sample_mean, sample_sd), sample_mean, sample_sd,
                                  x_val=x_val, calc_type="P(X > x)", title="Sampling Distribution of the Mean")

        elif option == "P(a < X̄ < b)":
            a = st.number_input("Lower bound (a):", value=sample_mean - sample_sd)
            b = st.number_input("Upper bound (b):", value=sample_mean + sample_sd)
            if st.button("Calculate"):
                prob = norm.cdf(b, sample_mean, sample_sd) - norm.cdf(a, sample_mean, sample_sd)
                st.success(f"P({a} < X̄ < {b}) = {rp(prob)}")
                if show_steps:
                    z1 = (a - sample_mean)/sample_sd
                    z2 = (b - sample_mean)/sample_sd
                    st.write(f"Z₁={rp(z1)}, Z₂={rp(z2)}")
                x = np.linspace(sample_mean - 4*sample_sd, sample_mean + 4*sample_sd, 1000)
                plot_distribution(x, norm.pdf(x, sample_mean, sample_sd), sample_mean, sample_sd,
                                  a_val=a, b_val=b, calc_type="P(a < X < b)", title="Sampling Distribution of the Mean")

        elif option == "Inverse: Find x given probability":
            p = st.number_input("Enter cumulative probability:", min_value=0.0, max_value=1.0, value=0.95)
            if st.button("Calculate"):
                x_val = norm.ppf(p, sample_mean, sample_sd)
                st.success(f"x̄ such that P(X̄ < x̄)={p} is {rp(x_val)}")
                x = np.linspace(sample_mean - 4*sample_sd, sample_mean + 4*sample_sd, 1000)
                plot_distribution(x, norm.pdf(x, sample_mean, sample_sd), sample_mean, sample_sd,
                                  x_val=x_val, calc_type="P(X < x)", title="Sampling Distribution of the Mean")

    # ----------------------------------------------------------
    # 3. SAMPLING DISTRIBUTION OF THE PROPORTION
    # ----------------------------------------------------------
    elif dist_type == "Sampling Distribution of the Proportion":
        st.subheader("📘 Sampling Distribution of the Sample Proportion")
        p_expr = st.text_input("Enter population proportion (p̂): e.g. 120/200 or 0.6", value="0.5")
        n = st.number_input("Sample size (n):", min_value=1, step=1, value=30)

        p_hat = parse_expression(p_expr)
        if p_hat is None:
            return
        if not (0 < p_hat < 1):
            st.error("p̂ must be between 0 and 1.")
            return

        q_hat = 1 - p_hat
        sample_mean = p_hat
        sample_sd = math.sqrt((p_hat * q_hat) / n)
        st.latex(r"\mu_{\hat{p}} = p, \quad \sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}")
        st.write(f"**Mean:** {rp(sample_mean)}  |  **Std. Error:** {rp(sample_sd)}")

        st.markdown("---")
        option = st.selectbox("", ["P(p̂ < x)", "P(p̂ > x)", "P(a < p̂ < b)", "Inverse: Find x given probability"])
        show_steps = st.checkbox("📖 Show Step-by-Step Solution")

        if option == "P(p̂ < x)":
            x_val = st.number_input("Enter p̂ value:", value=sample_mean)
            if st.button("Calculate"):
                prob = norm.cdf(x_val, sample_mean, sample_sd)
                st.success(f"P(p̂ < {x_val}) = {rp(prob)}")
                if show_steps:
                    z = (x_val - sample_mean)/sample_sd
                    st.write(f"Z = {rp(z)}")
                x = np.linspace(sample_mean - 4*sample_sd, sample_mean + 4*sample_sd, 1000)
                plot_distribution(x, norm.pdf(x, sample_mean, sample_sd), sample_mean, sample_sd,
                                  x_val=x_val, calc_type="P(p̂ < x)", title="Sampling Distribution of the Proportion")

        elif option == "P(p̂ > x)":
            x_val = st.number_input("Enter p̂ value:", value=sample_mean)
            if st.button("Calculate"):
                prob = 1 - norm.cdf(x_val, sample_mean, sample_sd)
                st.success(f"P(p̂ > {x_val}) = {rp(prob)}")
                if show_steps:
                    z = (x_val - sample_mean)/sample_sd
                    st.write(f"Z = {rp(z)}")
                x = np.linspace(sample_mean - 4*sample_sd, sample_mean + 4*sample_sd, 1000)
                plot_distribution(x, norm.pdf(x, sample_mean, sample_sd), sample_mean, sample_sd,
                                  x_val=x_val, calc_type="P(p̂ > x)", title="Sampling Distribution of the Proportion")

        elif option == "P(a < p̂ < b)":
            a = st.number_input("Lower bound (a):", value=sample_mean - sample_sd)
            b = st.number_input("Upper bound (b):", value=sample_mean + sample_sd)
            if st.button("Calculate"):
                prob = norm.cdf(b, sample_mean, sample_sd) - norm.cdf(a, sample_mean, sample_sd)
                st.success(f"P({a} < p̂ < {b}) = {rp(prob)}")
                x = np.linspace(sample_mean - 4*sample_sd, sample_mean + 4*sample_sd, 1000)
                plot_distribution(x, norm.pdf(x, sample_mean, sample_sd), sample_mean, sample_sd,
                                  a_val=a, b_val=b, calc_type="P(a < p̂ < b)", title="Sampling Distribution of the Proportion")

        elif option == "Inverse: Find x given probability":
            p = st.number_input("Enter cumulative probability:", min_value=0.0, max_value=1.0, value=0.95)
            if st.button("Calculate"):
                x_val = norm.ppf(p, sample_mean, sample_sd)
                st.success(f"p̂ such that P(p̂ < x)={p} is {rp(x_val)}")
                x = np.linspace(sample_mean - 4*sample_sd, sample_mean + 4*sample_sd, 1000)
                plot_distribution(x, norm.pdf(x, sample_mean, sample_sd), sample_mean, sample_sd,
                                  x_val=x_val, calc_type="P(p̂ < x)", title="Sampling Distribution of the Proportion")

if __name__ == "__main__":
    run()
