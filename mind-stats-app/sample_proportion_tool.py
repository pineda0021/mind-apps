import streamlit as st
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

# ==========================================================
# Safe numeric parser
# ==========================================================
def parse_expression(expr):
    expr = expr.strip().lower()
    try:
        if "sqrt" in expr:
            inner = expr[expr.find("(")+1:expr.find(")")]
            return math.sqrt(float(inner))
        elif "/" in expr:
            a, b = expr.split("/")
            return float(a) / float(b)
        else:
            return float(expr)
    except Exception:
        st.error("⚠️ Invalid input format. Use plain numbers, fractions (a/b), or sqrt(x).")
        return None

# ==========================================================
# Normal Distribution Calculator
# ==========================================================
def normal_distribution():
    st.subheader("📈 Normal Distribution")
    st.latex(r"Z = \frac{X - \mu}{\sigma}")

    μ = st.number_input("Population mean (μ):", value=0.0)
    σ = st.number_input("Standard deviation (σ):", min_value=0.0001, value=1.0)
    X = st.number_input("Enter value of X:", value=0.0)
    Z = (X - μ) / σ
    st.success(f"Z = ({X} − {μ}) / {σ} = {round(Z,4)}")

    x = np.linspace(μ - 4*σ, μ + 4*σ, 500)
    y = norm.pdf(x, μ, σ)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, y, lw=2, color="blue")
    ax.fill_between(x, 0, y, where=(x <= X), color="skyblue", alpha=0.6)
    ax.axvline(X, color="red", linestyle="--")
    ax.set_xlabel("X")
    ax.set_ylabel("Density")
    ax.set_title("Normal Distribution")
    st.pyplot(fig)

# ==========================================================
# Sampling Distribution of the Mean
# ==========================================================
def sampling_mean_distribution():
    st.subheader("📘 Sampling Distribution of the Mean")
    st.latex(r"Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}}")

    μ_expr = st.text_input("Population mean (μ):", value="0")
    σ_expr = st.text_input("Population SD (σ):", value="1")
    n = st.number_input("Sample size (n):", min_value=1, step=1, value=30)

    μ = parse_expression(μ_expr)
    σ = parse_expression(σ_expr)
    if μ is None or σ is None:
        return

    SE = σ / math.sqrt(n)
    st.write(f"**Standard Error (σₓ̄) =** {round(SE,4)}")

    X̄ = st.number_input("Enter sample mean (x̄):", value=μ)
    Z = (X̄ - μ) / SE
    st.success(f"Z = ({X̄} − {μ}) / {round(SE,4)} = {round(Z,4)}")

    x = np.linspace(μ - 4*SE, μ + 4*SE, 500)
    y = norm.pdf(x, μ, SE)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, y, lw=2, color="blue")
    ax.fill_between(x, 0, y, where=(x <= X̄), color="skyblue", alpha=0.6)
    ax.axvline(X̄, color="red", linestyle="--")
    ax.set_xlabel("x̄")
    ax.set_ylabel("Density")
    ax.set_title("Sampling Distribution of the Mean")
    st.pyplot(fig)

# ==========================================================
# Sampling Distribution of the Proportion
# ==========================================================
def sampling_proportion_distribution():
    st.subheader("📘 Sampling Distribution of the Proportion")
    st.latex(r"Z = \frac{\hat{p} - p}{\sqrt{p(1-p)/n}}")

    p_expr = st.text_input("Population proportion (p):", value="0.5")
    n = st.number_input("Sample size (n):", min_value=1, step=1, value=30)

    p = parse_expression(p_expr)
    if p is None or not (0 < p < 1):
        st.error("p must be between 0 and 1.")
        return

    q = 1 - p
    SE = math.sqrt(p*q/n)
    st.write(f"**Standard Error (σₚ̂) =** {round(SE,4)}")

    p_hat = st.number_input("Enter sample proportion (p̂):", value=p)
    Z = (p_hat - p) / SE
    st.success(f"Z = ({p_hat} − {p}) / {round(SE,4)} = {round(Z,4)}")

    x = np.linspace(p - 4*SE, p + 4*SE, 500)
    y = norm.pdf(x, p, SE)
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
def uniform_distribution():
    st.subheader("📏 Uniform Distribution")
    st.latex(r"f(x) = \frac{1}{b - a}, \quad a \le x \le b")
    st.latex(r"P(X < x) = P(X \le x)")
    st.latex(r"E[X] = \frac{a + b}{2}, \quad Var[X] = \frac{(b - a)^2}{12}")

    a = st.number_input("Lower bound (a):", value=0.0)
    b = st.number_input("Upper bound (b):", value=10.0)
    if b <= a:
        st.error("⚠️ The upper bound (b) must be greater than the lower bound (a).")
        return

    pdf = 1 / (b - a)
    mean = (a + b)/2
    var = ((b - a)**2)/12
    st.write(f"**Mean = {round(mean,4)} | Variance = {round(var,4)} | f(x) = {round(pdf,4)}**")

    calc_type = st.selectbox("Select probability type:", 
                             ["P(X ≤ x)", "P(X ≥ x)", "P(a < X < b)", "Find x for a given probability"])

    x = np.linspace(a - (b - a)*0.2, b + (b - a)*0.2, 500)
    y = np.where((x >= a) & (x <= b), pdf, 0)

    # CASES
    if calc_type == "P(X ≤ x)":
        x_val = st.number_input("Enter x value:", value=a + (b - a)/2)
        prob = 0 if x_val <= a else 1 if x_val >= b else (x_val - a)/(b - a)
        st.success(f"P(X ≤ {x_val}) = {round(prob,4)}")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x <= x_val) & (x >= a), color="skyblue", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    elif calc_type == "P(X ≥ x)":
        x_val = st.number_input("Enter x value:", value=a + (b - a)/2)
        prob = 1 if x_val <= a else 0 if x_val >= b else (b - x_val)/(b - a)
        st.success(f"P(X ≥ {x_val}) = {round(prob,4)}")
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
        st.success(f"P({a1} < X < {b1}) = {round(prob,4)}")
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
        st.success(f"x = {round(x_val,4)} for {tail}")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, y, color="blue")
        if tail.startswith("Left"):
            ax.fill_between(x, 0, y, where=(x <= x_val) & (x >= a), color="skyblue", alpha=0.6)
        else:
            ax.fill_between(x, 0, y, where=(x >= x_val) & (x <= b), color="lightgreen", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

# ==========================================================
# MAIN APP CONTROLLER
# ==========================================================
def run():
    st.header("📊 Continuous Probability Distributions")

    choice = st.selectbox(
        "Select a distribution to explore:",
        ["Normal Distribution", "Sampling Distribution of the Mean", 
         "Sampling Distribution of the Proportion", "Uniform Distribution"],
        index=None,
        placeholder="Choose one to begin..."
    )

    if not choice:
        st.info("👆 Please select a distribution to begin.")
        return

    if choice == "Normal Distribution":
        normal_distribution()
    elif choice == "Sampling Distribution of the Mean":
        sampling_mean_distribution()
    elif choice == "Sampling Distribution of the Proportion":
        sampling_proportion_distribution()
    elif choice == "Uniform Distribution":
        uniform_distribution()

# ==========================================================
# Run App
# ==========================================================
if __name__ == "__main__":
    run()
