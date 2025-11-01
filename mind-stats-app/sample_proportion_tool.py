import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import re

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

# ---------- Main App ----------
def run():
    st.header("🔔 Sampling Distribution & Standard Normal Calculator")

    st.markdown("""
    Choose the type of sampling distribution:
    - **Mean (μ)** — when population mean and standard deviation are known  
    - **Proportion (p̂)** — when working with success/failure proportions  
    """)

    calc_type = st.radio("Select distribution type:", ["Sampling Distribution of the Mean", "Sampling Distribution of the Proportion"])

    # ----------- SAMPLING DISTRIBUTION OF MEAN -----------
    if calc_type == "Sampling Distribution of the Mean":
        st.subheader("📘 Sampling Distribution of the Sample Mean")
        mean_expr = st.text_input("Enter the population mean (μ):", value="0")
        sd_expr = st.text_input("Enter the population standard deviation (σ):", value="1")
        n = st.number_input("Enter the sample size (n):", min_value=1, step=1, value=30)
        decimal = st.number_input("Decimal places for output:", min_value=0, max_value=10, value=4, step=1)

        mean = parse_expression(mean_expr)
        sd = parse_expression(sd_expr)
        if mean is None or sd is None:
            return

        sample_mean = mean
        sample_sd = sd / math.sqrt(n)

        st.markdown("---")
        st.latex(r"\mu_{\bar{x}} = \mu")
        st.latex(r"\sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}}")
        st.write(f"**Mean of Sampling Distribution (μₓ̄):** {sample_mean:.{decimal}f}")
        st.write(f"**Standard Error (σₓ̄):** {sample_sd:.{decimal}f}")

    # ----------- SAMPLING DISTRIBUTION OF PROPORTION -----------
    else:
        st.subheader("📘 Sampling Distribution of the Sample Proportion")
        p_expr = st.text_input("Enter population proportion (p̂): (can enter as fraction e.g. 120/200)", value="0.5")
        n = st.number_input("Enter the sample size (n):", min_value=1, step=1, value=30)
        decimal = st.number_input("Decimal places for output:", min_value=0, max_value=10, value=4, step=1)

        p_hat = parse_expression(p_expr)
        if p_hat is None:
            return
        if p_hat <= 0 or p_hat >= 1:
            st.error("p̂ must be between 0 and 1.")
            return

        q_hat = 1 - p_hat
        sample_mean = p_hat
        sample_sd = math.sqrt((p_hat * q_hat) / n)

        st.markdown("---")
        st.latex(r"\mu_{\hat{p}} = p")
        st.latex(r"\sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}")
        st.write(f"**Mean of Sampling Distribution (μₚ̂):** {sample_mean:.{decimal}f}")
        st.write(f"**Standard Error (σₚ̂):** {sample_sd:.{decimal}f}")

    # ----------- OPTIONS -----------
    st.markdown("---")
    st.write("### Choose a calculation:")
    option = st.radio("", [
        "1. P(X < x) or P(p̂ < x)",
        "2. P(X > x) or P(p̂ > x)",
        "3. P(a < X < b) or P(a < p̂ < b)",
        "4. Inverse: Find x for given P(X < x)",
        "5. Inverse: Find x for given P(X > x)",
        "6. Inverse: Find x₁ and x₂ for given P(x₁ < X < x₂)"
    ])

    x_val = a = b = p = p_lower = p_upper = None

    if option == "1. P(X < x) or P(p̂ < x)":
        x_val = parse_expression(st.text_input("Enter value of x or p̂:", value=f"{sample_mean:.4f}"))

    elif option == "2. P(X > x) or P(p̂ > x)":
        x_val = parse_expression(st.text_input("Enter value of x or p̂:", value=f"{sample_mean:.4f}"))

    elif option == "3. P(a < X < b) or P(a < p̂ < b)":
        a = parse_expression(st.text_input("Enter lower bound (a):", value=f"{sample_mean - sample_sd:.4f}"))
        b = parse_expression(st.text_input("Enter upper bound (b):", value=f"{sample_mean + sample_sd:.4f}"))
        if a is not None and b is not None and b < a:
            st.error("Upper bound b must be greater than or equal to lower bound a.")
            return

    elif option == "4. Inverse: Find x for given P(X < x)":
        p = st.number_input("Enter cumulative probability p (0 < p < 1):", min_value=0.0, max_value=1.0, value=0.5)

    elif option == "5. Inverse: Find x for given P(X > x)":
        p = st.number_input("Enter cumulative probability p (0 < p < 1):", min_value=0.0, max_value=1.0, value=0.5)

    elif option == "6. Inverse: Find x₁ and x₂ for given P(x₁ < X < x₂)":
        p_lower = st.number_input("Enter lower cumulative probability (0 < p_lower < 1):", min_value=0.0, max_value=1.0, value=0.2)
        p_upper = st.number_input("Enter upper cumulative probability (p_lower < p_upper < 1):", min_value=0.0, max_value=1.0, value=0.8)
        if p_upper <= p_lower:
            st.error("Upper probability must be greater than lower probability.")
            return

    show_steps = st.checkbox("📖 Show Step-by-Step Solution")
    show_z = st.checkbox("📊 Show Standard Normal (Z) Distribution Comparison")

    # ---------- CALCULATE ----------
    if st.button("🔹 Calculate"):
        x = np.linspace(sample_mean - 4 * sample_sd, sample_mean + 4 * sample_sd, 1000)
        y = stats.norm.pdf(x, sample_mean, sample_sd)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, label="Sampling Distribution", color='blue')

        def rp(v): return round(v, decimal)

        # Calculate and visualize
        if option == "1. P(X < x) or P(p̂ < x)":
            prob = stats.norm.cdf(x_val, sample_mean, sample_sd)
            z = (x_val - sample_mean) / sample_sd
            st.success(f"P(X < {x_val}) = {rp(prob)}")

            if show_steps:
                st.markdown(f"**Step 1:** Compute Z = (x - μ) / σ = ({x_val} - {sample_mean}) / {sample_sd} = **{rp(z)}**")
                st.markdown(f"**Step 2:** Find P(Z < {rp(z)}) = **{rp(prob)}**")

            ax.fill_between(x, 0, y, where=(x <= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--', label=f"x = {rp(x_val)}")

        elif option == "2. P(X > x) or P(p̂ > x)":
            prob = 1 - stats.norm.cdf(x_val, sample_mean, sample_sd)
            z = (x_val - sample_mean) / sample_sd
            st.success(f"P(X > {x_val}) = {rp(prob)}")

            if show_steps:
                st.markdown(f"**Step 1:** Compute Z = (x - μ) / σ = ({x_val} - {sample_mean}) / {sample_sd} = **{rp(z)}**")
                st.markdown(f"**Step 2:** Find P(Z > {rp(z)}) = **{rp(prob)}**")

            ax.fill_between(x, 0, y, where=(x >= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--', label=f"x = {rp(x_val)}")

        elif option == "3. P(a < X < b) or P(a < p̂ < b)":
            prob = stats.norm.cdf(b, sample_mean, sample_sd) - stats.norm.cdf(a, sample_mean, sample_sd)
            z1 = (a - sample_mean) / sample_sd
            z2 = (b - sample_mean) / sample_sd
            st.success(f"P({a} < X < {b}) = {rp(prob)}")

            if show_steps:
                st.markdown(f"**Z₁ = ({a} - {sample_mean}) / {sample_sd} = {rp(z1)}**")
                st.markdown(f"**Z₂ = ({b} - {sample_mean}) / {sample_sd} = {rp(z2)}**")
                st.markdown(f"P(Z₁ < Z < Z₂) = P({rp(z1)} < Z < {rp(z2)}) = **{rp(prob)}**")

            ax.fill_between(x, 0, y, where=(x >= a) & (x <= b), color='lightblue')
            ax.axvline(a, color='red', linestyle='--')
            ax.axvline(b, color='red', linestyle='--')

        elif option == "4. Inverse: Find x for given P(X < x)":
            x_val = stats.norm.ppf(p, sample_mean, sample_sd)
            z = stats.norm.ppf(p)
            st.success(f"x such that P(X < x) = {p} is {rp(x_val)}")

            if show_steps:
                st.markdown(f"**Z = invNorm({p}) = {rp(z)}**")
                st.markdown(f"**x = μ + Z·σ = {sample_mean} + ({rp(z)}·{sample_sd}) = {rp(x_val)}**")

            ax.fill_between(x, 0, y, where=(x <= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        elif option == "5. Inverse: Find x for given P(X > x)":
            x_val = stats.norm.ppf(1 - p, sample_mean, sample_sd)
            z = stats.norm.ppf(1 - p)
            st.success(f"x such that P(X > x) = {p} is {rp(x_val)}")

            if show_steps:
                st.markdown(f"**Z = invNorm(1 - {p}) = {rp(z)}**")
                st.markdown(f"**x = μ + Z·σ = {sample_mean} + ({rp(z)}·{sample_sd}) = {rp(x_val)}**")

            ax.fill_between(x, 0, y, where=(x >= x_val), color='lightblue')
            ax.axvline(x_val, color='red', linestyle='--')

        elif option == "6. Inverse: Find x₁ and x₂ for given P(x₁ < X < x₂)":
            x1 = stats.norm.ppf(p_lower, sample_mean, sample_sd)
            x2 = stats.norm.ppf(p_upper, sample_mean, sample_sd)
            prob_between = p_upper - p_lower
            st.success(f"x₁ = {rp(x1)}, x₂ = {rp(x2)}, P({rp(x1)} < X < {rp(x2)}) = {rp(prob_between)}")

            if show_steps:
                st.markdown(f"**Z₁ = invNorm({p_lower}) = {rp(stats.norm.ppf(p_lower))}**")
                st.markdown(f"**Z₂ = invNorm({p_upper}) = {rp(stats.norm.ppf(p_upper))}**")
                st.markdown(f"**x₁ = μ + Z₁·σ = {rp(x1)}**  and  **x₂ = μ + Z₂·σ = {rp(x2)}**")

            ax.fill_between(x, 0, y, where=(x >= x1) & (x <= x2), color='lightblue')
            ax.axvline(x1, color='red', linestyle='--')
            ax.axvline(x2, color='red', linestyle='--')

        ax.set_xlabel("Sample Mean (x̄) or Sample Proportion (p̂)")
        ax.set_ylabel("Density")
        ax.set_title("Sampling Distribution")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # ---------- OPTIONAL STANDARD NORMAL ----------
        if show_z:
            z_values = np.linspace(-4, 4, 1000)
            z_pdf = stats.norm.pdf(z_values)
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(z_values, z_pdf, color='purple', label="Standard Normal (Z)")
            ax2.fill_between(z_values, 0, z_pdf, color='lavender')
            ax2.axvline(0, color='black', linestyle='--')
            ax2.set_title("Standard Normal Distribution (Z ~ N(0,1))")
            ax2.set_xlabel("Z-Score")
            ax2.set_ylabel("Density")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)

if __name__ == "__main__":
    run()
