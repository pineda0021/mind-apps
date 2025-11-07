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
    st.markdown("### 📈 **Normal Distribution**  [🔗](#)")
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
    st.markdown("### 📘 **Sampling Distribution of the Mean**  [🔗](#)")
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
# Sampling Distribution of the Proportion
# ==========================================================
def sampling_proportion(decimal):
    st.markdown("### 📗 **Sampling Distribution of the Proportion**  [🔗](#)")
    st.latex(r"Z = \frac{\hat{p} - p}{\sqrt{p(1-p)/n}}")

    st.write("Used when working with proportions (percentages), such as approval ratings or survey results.")

    p_expr = st.text_input("Population proportion (p):", value="0.5")
    n = st.number_input("Sample size (n):", min_value=1, step=1, value=30)

    p = parse_expression(p_expr)
    if p is None or not (0 < p < 1):
        st.error("p must be between 0 and 1.")
        return

    q = 1 - p
    se = math.sqrt(p*q/n)
    st.write(f"**Standard Error (σₚ̂) =** {round(se, decimal)}")

    p_hat = st.number_input("Enter sample proportion (p̂):", value=p)
    z = (p_hat - p) / se
    st.success(f"Z = ({p_hat} − {p}) / {round(se, decimal)} = {round(z, decimal)}")

    x = np.linspace(p - 4*se, p + 4*se, 500)
    y = norm.pdf(x, p, se)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, y, lw=2, color="blue")
    ax.fill_between(x, 0, y, where=(x <= p_hat), color="skyblue", alpha=0.6)
    ax.axvline(p_hat, color="red", linestyle="--")
    ax.set_xlabel("p̂")
    ax.set_ylabel("Density")
    ax.set_title("Sampling Distribution of the Proportion")
    st.pyplot(fig)


# ==========================================================
# Uniform Distribution
# ==========================================================
def uniform_distribution(decimal):
    st.markdown("### 🎲 **Uniform Distribution**  [🔗](#)")
    st.latex(r"f(x) = \frac{1}{b - a}, \quad a \le x \le b")
    st.latex(r"P(X < x) = P(X \le x)")
    st.latex(r"E[X] = \frac{a + b}{2}, \quad Var[X] = \frac{(b - a)^2}{12}")

    st.write("Models data where all values within an interval are equally likely — like rolling a fair die.")

    a = st.number_input("Lower bound (a):", value=0.0)
    b = st.number_input("Upper bound (b):", value=10.0)
    if b <= a:
        st.error("⚠️ The upper bound (b) must be greater than (a).")
        return

    pdf = 1 / (b - a)
    mean = (a + b)/2
    var = ((b - a)**2)/12
    st.write(f"**Mean = {round(mean, decimal)} | Variance = {round(var, decimal)} | f(x) = {round(pdf, decimal)}**")

    calc_type = st.selectbox("Select probability type:", 
                             ["P(X ≤ x)", "P(X ≥ x)", "P(a < X < b)", "Find x for a given probability"])

    x = np.linspace(a - (b - a)*0.2, b + (b - a)*0.2, 500)
    y = np.where((x >= a) & (x <= b), pdf, 0)

    if calc_type == "P(X ≤ x)":
        x_val = st.number_input("Enter x value:", value=a + (b - a)/2)
        prob = 0 if x_val <= a else 1 if x_val >= b else (x_val - a)/(b - a)
        st.success(f"P(X ≤ {x_val}) = {round(prob, decimal)}")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x <= x_val) & (x >= a), color="skyblue", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    elif calc_type == "P(X ≥ x)":
        x_val = st.number_input("Enter x value:", value=a + (b - a)/2)
        prob = 1 if x_val <= a else 0 if x_val >= b else (b - x_val)/(b - a)
        st.success(f"P(X ≥ {x_val}) = {round(prob, decimal)}")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x >= x_val) & (x <= b), color="lightgreen", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    elif calc_type == "P(a < X < b)":
        a1 = st.number_input("Lower bound (a₁):", value=a + (b - a)/4)
        b1 = st.number_input("Upper bound (b₁):", value=a + (b - a)*3/4)
        if a1 >= b1:
            st.error("⚠️ Lower bound must be less than upper bound.")
            return
        prob = (b1 - a1)/(b - a)
        st.success(f"P({a1} < X < {b1}) = {round(prob, decimal)}")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x >= a1) & (x <= b1), color="orange", alpha=0.6)
        ax.axvline(a1, color="red", linestyle="--")
        ax.axvline(b1, color="red", linestyle="--")
        st.pyplot(fig)

    elif calc_type == "Find x for a given probability":
        p = st.number_input("Enter probability (0 < p < 1):", min_value=0.0, max_value=1.0, value=0.5)
        tail = st.selectbox("Tail:", ["Left tail: P(X ≤ x) = p", "Right tail: P(X ≥ x) = p"])
        x_val = a + p*(b - a) if tail.startswith("Left") else b - p*(b - a)
        st.success(f"x = {round(x_val, decimal)} for {tail}")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y, color="blue")
        if tail.startswith("Left"):
            ax.fill_between(x, 0, y, where=(x <= x_val) & (x >= a), color="skyblue", alpha=0.6)
        else:
            ax.fill_between(x, 0, y, where=(x >= x_val) & (x <= b), color="lightgreen", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)


# ==========================================================
# MAIN APP
# ==========================================================
def run():
    st.header("🧠 MIND: Continuous Probability Distributions")

    dist_choice = st.radio(
        "Select Distribution Type:",
        ["Uniform Distribution", "Normal Distribution",
         "Sampling Distribution of the Mean", "Sampling Distribution of the Proportion"],
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

