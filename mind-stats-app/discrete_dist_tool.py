import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import binom, poisson
from fractions import Fraction

# ==========================================================
# Helper Function
# ==========================================================

def parse_fraction(p_str):
    """Parses fraction or decimal input safely."""
    try:
        return float(Fraction(p_str.strip()))
    except Exception:
        st.error("❌ Please enter a valid decimal or fraction (e.g., 0.5 or 1/2).")
        return None

# ==========================================================
# Discrete Distribution
# ==========================================================

def discrete_distribution_tool():
    st.subheader("🎲 Custom Discrete Probability Distribution")

    st.markdown("""
    Enter values of a discrete random variable **X** and their corresponding probabilities **P(X)**.
    The calculator will verify that probabilities sum to 1 and compute all key descriptive measures.
    """)

    x_input = st.text_input("Values of X (comma-separated):", "0,1,2,3")
    p_input = st.text_input("Probabilities P(X) (comma-separated, decimals or fractions):", "1/8,3/8,3/8,1/8")

    if st.button("📊 Calculate Discrete Distribution"):
        if not x_input or not p_input:
            st.info("Please enter both X values and probabilities.")
            return

        try:
            X = np.array([float(x.strip()) for x in x_input.split(",")])
            P = np.array([float(Fraction(p.strip())) for p in p_input.split(",")])
        except Exception as e:
            st.error(f"Error parsing input: {e}")
            return

        if len(X) != len(P):
            st.error("⚠️ X and P must have the same length.")
            return

        if not np.isclose(np.sum(P), 1):
            st.error(f"⚠️ Probabilities must sum to 1. Current sum: {np.sum(P):.4f}")
            return

        μ = np.sum(X * P)
        σ2 = np.sum(P * (X - μ) ** 2)
        σ = np.sqrt(σ2)

        st.markdown("### 📋 Distribution Table")
        df = pd.DataFrame({
            "x": X,
            "P(X = x)": np.round(P, 5),
            "x·P(X)": np.round(X * P, 5)
        }).sort_values("x").reset_index(drop=True)
        st.dataframe(df, use_container_width=True)

        st.markdown("### 🧮 Summary Statistics")
        st.markdown(f"""
        - **Mean (Expected Value):**  μ = Σ[x·P(X=x)] = **{μ:.5f}**  
        - **Variance:**  σ² = Σ[(x − μ)²·P(X=x)] = **{σ2:.5f}**  
        - **Standard Deviation:**  σ = √σ² = **{σ:.5f}**
        """)

        fig = go.Figure(go.Bar(x=X, y=P, marker_color='mediumseagreen'))
        fig.update_layout(
            title="Probability Distribution of X",
            xaxis_title="x",
            yaxis_title="P(X = x)",
            template="simple_white",
            yaxis=dict(range=[0, max(P)*1.2])
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Binomial Distribution
# ==========================================================

def binomial_distribution_tool():
    st.subheader("🎯 Binomial Distribution")

    st.markdown("""
    A binomial experiment consists of **n independent trials**, each with probability **p** of success.  
    Formula:  
    \\[
    P(X = x) = \\binom{n}{x} p^x (1-p)^{n-x}
    \\]
    """)

    n = st.number_input("Number of trials (n)", min_value=1, step=1)
    p_str = st.text_input("Probability of success (p)", value="1/2")
    p = parse_fraction(p_str)

    if p is None or not (0 <= p <= 1):
        st.warning("⚠️ Please enter a valid probability between 0 and 1.")
        return

    x_vals = np.arange(0, n + 1)
    pmf_vals = binom.pmf(x_vals, n, p)

    μ = n * p
    σ2 = n * p * (1 - p)
    σ = np.sqrt(σ2)

    st.markdown("### 🧮 Distribution Parameters")
    st.markdown(f"""
    - **Mean:** μ = n·p = **{μ:.5f}**  
    - **Variance:** σ² = n·p·(1−p) = **{σ2:.5f}**  
    - **Standard Deviation:** σ = √σ² = **{σ:.5f}**
    """)

    calc_type = st.selectbox(
        "Choose a probability calculation:",
        ["Exactly: P(X = x)", "At most: P(X ≤ x)", "Less than: P(X < x)",
         "At least: P(X ≥ x)", "Greater than: P(X > x)", "Between: P(a ≤ X ≤ b)",
         "Show table and graph"]
    )

    if "Between" in calc_type:
        a = st.number_input("Enter lower bound (a):", min_value=0, max_value=int(n), step=1)
        b = st.number_input("Enter upper bound (b):", min_value=0, max_value=int(n), step=1)
        if a > b:
            st.warning("⚠️ Lower bound (a) must be ≤ upper bound (b).")
            return
    elif calc_type != "Show table and graph":
        x = st.number_input("Enter x value:", min_value=0, max_value=int(n), step=1)
    else:
        x = a = b = None

    if st.button("📊 Calculate Binomial"):
        if calc_type == "Exactly: P(X = x)":
            prob = binom.pmf(x, n, p)
            st.success(f"P(X = {x}) = **{prob:.5f}**")
        elif calc_type == "At most: P(X ≤ x)":
            prob = binom.cdf(x, n, p)
            st.success(f"P(X ≤ {x}) = **{prob:.5f}**")
        elif calc_type == "Less than: P(X < x)":
            prob = binom.cdf(x - 1, n, p) if x > 0 else 0
            st.success(f"P(X < {x}) = **{prob:.5f}**")
        elif calc_type == "At least: P(X ≥ x)":
            prob = 1 - binom.cdf(x - 1, n, p)
            st.success(f"P(X ≥ {x}) = **{prob:.5f}**")
        elif calc_type == "Greater than: P(X > x)":
            prob = 1 - binom.cdf(x, n, p)
            st.success(f"P(X > {x}) = **{prob:.5f}**")
        elif calc_type == "Between: P(a ≤ X ≤ b)":
            prob = binom.cdf(b, n, p) - (binom.cdf(a - 1, n, p) if a > 0 else 0)
            st.success(f"P({a} ≤ X ≤ {b}) = **{prob:.5f}**")
        elif calc_type == "Show table and graph":
            df = pd.DataFrame({"x": x_vals, "P(X = x)": np.round(pmf_vals, 5)})
            st.dataframe(df, use_container_width=True)

            fig = go.Figure(go.Bar(x=x_vals, y=pmf_vals, marker_color='dodgerblue'))
            fig.update_layout(
                title=f'Binomial Distribution (n={n}, p={p})',
                xaxis_title='x',
                yaxis_title='P(X = x)',
                template='simple_white'
            )
            st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Poisson Distribution
# ==========================================================

def poisson_distribution_tool():
    st.subheader("💡 Poisson Distribution")

    st.markdown("""
    The Poisson distribution models the number of occurrences in a fixed interval when events happen independently.  
    Formula:  
    \\[
    P(X = x) = \\frac{e^{-λ} λ^x}{x!}
    \\]
    """)

    lam = st.number_input("Mean number of occurrences (λ)", min_value=0.0, format="%.4f")
    x_max = st.number_input("Maximum number of occurrences to display", min_value=1, step=1)

    if lam <= 0:
        st.warning("⚠️ λ must be greater than 0.")
        return

    x_vals = np.arange(0, int(x_max) + 1)
    pmf_vals = poisson.pmf(x_vals, lam)

    μ = lam
    σ2 = lam
    σ = np.sqrt(σ2)

    st.markdown("### 🧮 Distribution Parameters")
    st.markdown(f"""
    - **Mean:** μ = λ = **{μ:.5f}**  
    - **Variance:** σ² = λ = **{σ2:.5f}**  
    - **Standard Deviation:** σ = √λ = **{σ:.5f}**
    """)

    calc_type = st.selectbox(
        "Choose a probability calculation:",
        ["Exactly: P(X = x)", "At most: P(X ≤ x)", "Less than: P(X < x)",
         "At least: P(X ≥ x)", "Greater than: P(X > x)", "Between: P(a ≤ X ≤ b)",
         "Show table and graph"]
    )

    if "Between" in calc_type:
        a = st.number_input("Enter lower bound (a):", min_value=0, max_value=int(x_max), step=1)
        b = st.number_input("Enter upper bound (b):", min_value=0, max_value=int(x_max), step=1)
        if a > b:
            st.warning("⚠️ Lower bound (a) must be ≤ upper bound (b).")
            return
    elif calc_type != "Show table and graph":
        x = st.number_input("Enter x value:", min_value=0, max_value=int(x_max), step=1)
    else:
        x = a = b = None

    if st.button("📊 Calculate Poisson"):
        if calc_type == "Exactly: P(X = x)":
            prob = poisson.pmf(x, lam)
            st.success(f"P(X = {x}) = **{prob:.5f}**")
        elif calc_type == "At most: P(X ≤ x)":
            prob = poisson.cdf(x, lam)
            st.success(f"P(X ≤ {x}) = **{prob:.5f}**")
        elif calc_type == "Less than: P(X < x)":
            prob = poisson.cdf(x - 1, lam)
            st.success(f"P(X < {x}) = **{prob:.5f}**")
        elif calc_type == "At least: P(X ≥ x)":
            prob = 1 - poisson.cdf(x - 1, lam)
            st.success(f"P(X ≥ {x}) = **{prob:.5f}**")
        elif calc_type == "Greater than: P(X > x)":
            prob = 1 - poisson.cdf(x, lam)
            st.success(f"P(X > {x}) = **{prob:.5f}**")
        elif calc_type == "Between: P(a ≤ X ≤ b)":
            prob = poisson.cdf(b, lam) - (poisson.cdf(a - 1, lam) if a > 0 else 0)
            st.success(f"P({a} ≤ X ≤ {b}) = **{prob:.5f}**")
        elif calc_type == "Show table and graph":
            df = pd.DataFrame({"x": x_vals, "P(X = x)": np.round(pmf_vals, 5)})
            st.dataframe(df, use_container_width=True)

            fig = go.Figure(go.Bar(x=x_vals, y=pmf_vals, marker_color='indianred'))
            fig.update_layout(
                title=f'Poisson Distribution (λ={lam})',
                xaxis_title='x',
                yaxis_title='P(X = x)',
                template='simple_white'
            )
            st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Main App
# ==========================================================

def run():
    st.header("🎰 Discrete Probability Distribution Calculator")

    categories = [
        "Custom Discrete Distribution",
        "Binomial Distribution",
        "Poisson Distribution"
    ]

    choice = st.selectbox(
        "Choose a distribution type:",
        categories,
        index=None,
        placeholder="Select a distribution..."
    )

    if not choice:
        st.info("👆 Please choose a distribution to begin.")
        return

    if choice == "Custom Discrete Distribution":
        discrete_distribution_tool()
    elif choice == "Binomial Distribution":
        binomial_distribution_tool()
    elif choice == "Poisson Distribution":
        poisson_distribution_tool()

if __name__ == "__main__":
    run()
