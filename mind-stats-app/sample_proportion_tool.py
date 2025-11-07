import streamlit as st
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

# ==========================================================
# Helper Function for Parsing Expressions
# ==========================================================
def parse_expression(expr):
    """Safely parse numbers or fractions like 120/200 or sqrt(9)."""
    expr = expr.strip().lower()
    try:
        if "sqrt" in expr:
            inside = expr[expr.find("(")+1:expr.find(")")]
            return math.sqrt(float(inside))
        elif "/" in expr:
            num, den = expr.split("/")
            return float(num) / float(den)
        else:
            return float(expr)
    except Exception:
        st.error("⚠️ Invalid input format. Use numbers, a/b, or sqrt(x).")
        return None


# ==========================================================
# Helper Plot Function (for Normal-like Distributions)
# ==========================================================
def plot_distribution(x, y, mean, sd, a=None, b=None, x_val=None, calc_type=None, title="Distribution"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color="blue", lw=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True)

    if calc_type == "P(X < x)" and x_val is not None:
        ax.fill_between(x, 0, y, where=(x <= x_val), color="skyblue", alpha=0.6)
    elif calc_type == "P(X > x)" and x_val is not None:
        ax.fill_between(x, 0, y, where=(x >= x_val), color="lightgreen", alpha=0.6)
    elif calc_type == "P(a < X < b)" and a is not None and b is not None:
        ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), color="orange", alpha=0.6)

    st.pyplot(fig)


# ==========================================================
# MAIN APP
# ==========================================================
def run():
    st.header("📘 Continuous Probability Distributions")

    dist_type = st.selectbox(
        "Choose a distribution:",
        [
            "Normal Distribution (X ~ N(μ, σ))",
            "Sampling Distribution of the Mean (x̄)",
            "Sampling Distribution of the Proportion (p̂)",
            "Uniform Distribution (a ≤ X ≤ b)"
        ],
        index=None,
        placeholder="Select a distribution to begin..."
    )

    if not dist_type:
        st.info("👆 Please select a distribution to begin.")
        return

    decimal = st.number_input("Decimal places:", min_value=0, max_value=10, value=4, step=1)
    rp = lambda v: round(v, decimal)

    # ==========================================================
    # NORMAL DISTRIBUTION
    # ==========================================================
    if "Normal" in dist_type:
        st.subheader("📈 Normal Distribution")
        st.latex(r"Z = \frac{X - \mu}{\sigma}")
        mean = st.number_input("Population mean (μ):", value=0.0)
        sd = st.number_input("Standard deviation (σ):", min_value=0.0001, value=1.0)
        x_val = st.number_input("Enter X value:", value=0.0)
        z = (x_val - mean) / sd
        st.success(f"Z = ({x_val} − {mean}) / {sd} = {rp(z)}")

        # Optional visualization
        x = np.linspace(mean - 4*sd, mean + 4*sd, 1000)
        y = norm.pdf(x, mean, sd)
        plot_distribution(x, y, mean, sd, x_val=x_val, calc_type="P(X < x)", title="Normal Distribution")

    # ==========================================================
    # SAMPLING DISTRIBUTION OF THE MEAN
    # ==========================================================
    elif "Mean" in dist_type:
        st.subheader("📘 Sampling Distribution of the Mean")
        st.latex(r"Z = \frac{\bar{X} - \mu_{\bar{X}}}{\sigma_{\bar{X}}}, \quad \mu_{\bar{X}} = \mu, \ \sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}")

        mu_expr = st.text_input("Population mean (μ):", value="0")
        sigma_expr = st.text_input("Population standard deviation (σ):", value="1")
        n = st.number_input("Sample size (n):", min_value=1, step=1, value=30)
        mu = parse_expression(mu_expr)
        sigma = parse_expression(sigma_expr)
        if mu is None or sigma is None:
            return
        se = sigma / math.sqrt(n)
        st.write(f"**Standard Error (σₓ̄) =** {rp(se)}")

        x_val = st.number_input("Enter sample mean (x̄):", value=mu)
        z = (x_val - mu) / se
        st.success(f"Z = ({x_val} − {mu}) / {rp(se)} = {rp(z)}")

        # Visualization
        x = np.linspace(mu - 4*se, mu + 4*se, 1000)
        y = norm.pdf(x, mu, se)
        plot_distribution(x, y, mu, se, x_val=x_val, calc_type="P(X < x)", title="Sampling Distribution of the Mean")

    # ==========================================================
    # SAMPLING DISTRIBUTION OF THE PROPORTION
    # ==========================================================
    elif "Proportion" in dist_type:
        st.subheader("📘 Sampling Distribution of the Proportion")
        st.latex(r"Z = \frac{\hat{p} - \mu_{\hat{p}}}{\sigma_{\hat{p}}}, \quad \mu_{\hat{p}} = p, \ \sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}")

        p_expr = st.text_input("Population proportion (p):", value="0.5")
        n = st.number_input("Sample size (n):", min_value=1, step=1, value=30)
        p = parse_expression(p_expr)
        if p is None or not (0 < p < 1):
            st.error("p must be between 0 and 1.")
            return
        q = 1 - p
        se = math.sqrt(p * q / n)
        st.write(f"**Standard Error (σₚ̂) =** {rp(se)}")

        p_hat = st.number_input("Enter sample proportion (p̂):", value=p)
        z = (p_hat - p) / se
        st.success(f"Z = ({p_hat} − {p}) / {rp(se)} = {rp(z)}")

        # Visualization
        x = np.linspace(p - 4*se, p + 4*se, 1000)
        y = norm.pdf(x, p, se)
        plot_distribution(x, y, p, se, x_val=p_hat, calc_type="P(X < x)", title="Sampling Distribution of the Proportion")

    # ==========================================================
    # UNIFORM DISTRIBUTION
    # ==========================================================
    elif "Uniform" in dist_type:
        st.subheader("📏 Uniform Distribution")
        st.latex(r"f(x) = \frac{1}{b - a}, \quad a \le x \le b")
        st.latex(r"P(X < x) = P(X \le x)")
        st.latex(r"E[X] = \frac{a + b}{2}, \quad Var[X] = \frac{(b - a)^2}{12}")

        # Inputs
        a = st.number_input("Enter minimum value (a):", value=0.0)
        b = st.number_input("Enter maximum value (b):", value=10.0)
        if b <= a:
            st.error("⚠️ The upper bound (b) must be greater than (a).")
            return

        mean = (a + b) / 2
        variance = ((b - a)**2) / 12
        pdf = 1 / (b - a)
        st.write(f"**Mean:** {rp(mean)} | **Variance:** {rp(variance)} | **f(x) =** {rp(pdf)}")

        # Select probability type
        calc_type = st.selectbox(
            "Choose a probability calculation:",
            ["P(X ≤ x)", "P(X ≥ x)", "P(a < X < b)", "Find x for a given probability"],
            index=0
        )
        show_steps = st.checkbox("📖 Show Step-by-Step Solution")

        # Base curve
        x = np.linspace(a - (b - a)*0.2, b + (b - a)*0.2, 500)
        y = np.where((x >= a) & (x <= b), pdf, 0)

        # ----- P(X ≤ x) -----
        if calc_type == "P(X ≤ x)":
            x_val = st.number_input("Enter x value:", value=a + (b - a)/2)
            if st.button("Calculate"):
                prob = 0 if x_val <= a else 1 if x_val >= b else (x_val - a) / (b - a)
                st.success(f"P(X ≤ {x_val}) = {rp(prob)}")
                if show_steps:
                    st.latex(r"P(X \le x) = \frac{x - a}{b - a}")
                    st.write(f"({x_val} − {a}) / ({b} − {a}) = {rp(prob)}")

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(x, y, color="blue", lw=2)
                ax.fill_between(x, 0, y, where=(x <= x_val) & (x >= a), color="skyblue", alpha=0.6)
                ax.axvline(x_val, color="red", linestyle="--")
                ax.set_title("Uniform Distribution")
                ax.set_xlabel("X")
                ax.set_ylabel("f(x)")
                st.pyplot(fig)

        # ----- P(X ≥ x) -----
        elif calc_type == "P(X ≥ x)":
            x_val = st.number_input("Enter x value:", value=a + (b - a)/2)
            if st.button("Calculate"):
                prob = 1 if x_val <= a else 0 if x_val >= b else (b - x_val) / (b - a)
                st.success(f"P(X ≥ {x_val}) = {rp(prob)}")
                if show_steps:
                    st.latex(r"P(X \ge x) = \frac{b - x}{b - a}")
                    st.write(f"({b} − {x_val}) / ({b} − {a}) = {rp(prob)}")

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(x, y, color="blue", lw=2)
                ax.fill_between(x, 0, y, where=(x >= x_val) & (x <= b), color="lightgreen", alpha=0.6)
                ax.axvline(x_val, color="red", linestyle="--")
                ax.set_title("Uniform Distribution")
                ax.set_xlabel("X")
                ax.set_ylabel("f(x)")
                st.pyplot(fig)

        # ----- P(a < X < b) -----
        elif calc_type == "P(a < X < b)":
            a1 = st.number_input("Lower bound (a₁):", value=a + (b - a)/4)
            b1 = st.number_input("Upper bound (b₁):", value=a + (b - a)*3/4)
            if st.button("Calculate"):
                if a1 < a: a1 = a
                if b1 > b: b1 = b
                if a1 >= b1:
                    st.error("⚠️ Lower bound must be less than upper bound.")
                    return
                prob = (b1 - a1) / (b - a)
                st.success(f"P({a1} < X < {b1}) = {rp(prob)}")
                if show_steps:
                    st.latex(r"P(a < X < b) = \frac{b_1 - a_1}{b - a}")
                    st.write(f"({b1} − {a1}) / ({b} − {a}) = {rp(prob)}")

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(x, y, color="blue", lw=2)
                ax.fill_between(x, 0, y, where=(x >= a1) & (x <= b1), color="orange", alpha=0.6)
                ax.axvline(a1, color="red", linestyle="--")
                ax.axvline(b1, color="red", linestyle="--")
                ax.set_title("Uniform Distribution")
                ax.set_xlabel("X")
                ax.set_ylabel("f(x)")
                st.pyplot(fig)

        # ----- Find x for given probability -----
        elif calc_type == "Find x for a given probability":
            p = st.number_input("Enter probability (0 < p < 1):", min_value=0.0, max_value=1.0, value=0.5)
            direction = st.selectbox("Select tail:", ["Left tail: P(X ≤ x) = p", "Right tail: P(X ≥ x) = p"])
            if st.button("Calculate"):
                if direction.startswith("Left"):
                    x_val = a + p * (b - a)
                    st.success(f"x = {rp(x_val)} for P(X ≤ x) = {p}")
                else:
                    x_val = b - p * (b - a)
                    st.success(f"x = {rp(x_val)} for P(X ≥ x) = {p}")

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(x, y, color="blue", lw=2)
                if direction.startswith("Left"):
                    ax.fill_between(x, 0, y, where=(x <= x_val) & (x >= a), color="skyblue", alpha=0.6)
                else:
                    ax.fill_between(x, 0, y, where=(x >= x_val) & (x <= b), color="lightgreen", alpha=0.6)
                ax.axvline(x_val, color="red", linestyle="--")
                ax.set_title("Uniform Distribution")
                ax.set_xlabel("X")
                ax.set_ylabel("f(x)")
                st.pyplot(fig)


# ==========================================================
# Run App
# ==========================================================
if __name__ == "__main__":
    run()
