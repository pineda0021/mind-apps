import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
from fractions import Fraction

# ---------- Helper Function ----------
def parse_fraction(p_str):
    try:
        return float(Fraction(p_str.strip()))
    except Exception:
        st.error("❌ Please enter a valid decimal or fraction (e.g., 0.5 or 1/2).")
        return None

# ---------- Discrete Distribution ----------
def discrete_distribution_tool():
    st.subheader("🎲 Discrete Probability Distribution")

    st.write("Enter values of discrete random variable X and their probabilities P(X).")
    x_input = st.text_input("Values of X (comma-separated):")
    p_input = st.text_input("Probabilities P(X) (comma-separated, decimals or fractions):")

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
            st.error("X and P must be the same length.")
            return

        if not np.isclose(np.sum(P), 1):
            st.error(f"Probabilities must sum to 1. Current sum: {np.sum(P):.4f}")
            return

        mean = np.sum(X * P)
        variance = np.sum(P * (X - mean) ** 2)
        std_dev = np.sqrt(variance)

        st.markdown("### 📋 Distribution Summary")
        summary_df = pd.DataFrame({
            "x": X,
            "P(X = x)": np.round(P, 5)
        }).sort_values("x").reset_index(drop=True)
        st.dataframe(summary_df, use_container_width=True)

        st.markdown(
            f"**Mean:** {mean:.5f}  \n"
            f"**Variance:** {variance:.5f}  \n"
            f"**Standard Deviation:** {std_dev:.5f}"
        )

        fig = go.Figure(go.Bar(x=X, y=P, marker_color='green'))
        fig.update_layout(
            title="Probability Distribution",
            xaxis_title="X",
            yaxis_title="P(X)",
            template="simple_white",
            yaxis=dict(range=[0, max(P)*1.2])
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------- Binomial Distribution ----------
def binomial_distribution_tool():
    st.subheader("🎲 Binomial Distribution")

    n = st.number_input("Number of trials (n)", min_value=1, step=1)
    p_str = st.text_input("Probability of success (p)", value="1/2")
    p = parse_fraction(p_str)

    if p is None:
        return
    if not (0 <= p <= 1):
        st.warning("⚠️ Probability must be between 0 and 1.")
        return

    x_vals = np.arange(0, n + 1)
    pmf_vals = binom.pmf(x_vals, n, p)

    mean = n * p
    variance = n * p * (1 - p)
    std_dev = np.sqrt(variance)

    st.markdown("### 📋 Binomial Summary")
    st.markdown(
        f"**Mean:** {mean:.5f}  \n"
        f"**Variance:** {variance:.5f}  \n"
        f"**Standard Deviation:** {std_dev:.5f}"
    )

    calc_type = st.selectbox(
        "Choose a probability calculation:",
        ["Exactly: P(X = x)", "At most: P(X ≤ x)", "Less than: P(X < x)", 
         "At least: P(X ≥ x)", "Greater than: P(X > x)", "Between: P(a ≤ X ≤ b)", 
         "Show table and graph"]
    )

    x = a = b = None
    if "Between" in calc_type:
        a = st.number_input("Enter lower bound (a):", min_value=0, max_value=int(n), step=1)
        b = st.number_input("Enter upper bound (b):", min_value=0, max_value=int(n), step=1)
        if a > b:
            st.warning("⚠️ Lower bound (a) must be ≤ upper bound (b).")
            return
    elif calc_type != "Show table and graph":
        x = st.number_input("Enter x value:", min_value=0, max_value=int(n), step=1)

    if st.button("📊 Calculate Binomial"):
        if calc_type == "Exactly: P(X = x)":
            prob = binom.pmf(x, n, p)
            st.success(f"P(X = {x}) = {prob:.5f}")
        elif calc_type == "P(X ≤ x)":
            prob = binom.cdf(x, n, p)
            st.success(f"P(X ≤ {x}) = {prob:.5f}")
        elif calc_type == "P(X < x)":
            prob = binom.cdf(x - 1, n, p) if x > 0 else 0.0
            st.success(f"P(X < {x}) = {prob:.5f}")
        elif calc_type == "P(X ≥ x)":
            prob = 1 - binom.cdf(x - 1, n, p) if x > 0 else 1.0
            st.success(f"P(X ≥ {x}) = {prob:.5f}")
        elif calc_type == "P(X > x)":
            prob = 1 - binom.cdf(x, n, p)
            st.success(f"P(X > {x}) = {prob:.5f}")
        elif calc_type == "Between: P(a ≤ X ≤ b)":
            prob = binom.cdf(b, n, p) - binom.cdf(a - 1, n, p) if a > 0 else binom.cdf(b, n, p)
            st.success(f"P({a} ≤ X ≤ {b}) = {prob:.5f}")
        elif calc_type == "Show table and graph":
            summary_df = pd.DataFrame({
                "x": x_vals,
                "P(X = x)": np.round(pmf_vals, 5)
            })
            st.dataframe(summary_df, use_container_width=True)

            fig = go.Figure(go.Bar(x=x_vals, y=pmf_vals, marker_color='skyblue'))
            fig.update_layout(
                title=f'Binomial Distribution (n={n}, p={p})',
                xaxis_title='x',
                yaxis_title='P(X = x)',
                template='simple_white'
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------- Poisson Distribution ----------
def poisson_distribution_tool():
    st.subheader("🎲 Poisson Distribution")

    lam = st.number_input("Mean number of occurrences (λ)", min_value=0.0, format="%.4f")
    x_max = st.number_input("Max number of occurrences to consider (x max)", min_value=1, step=1)

    if lam <= 0 or x_max < 1:
        st.warning("⚠️ Please enter λ > 0 and x max ≥ 1.")
        return

    x_vals = np.arange(0, int(x_max) + 1)
    pmf_vals = poisson.pmf(x_vals, lam)

    mean = lam
    variance = lam
    std_dev = np.sqrt(variance)

    st.markdown("### 📋 Poisson Summary")
    st.markdown(
        f"**Mean:** {mean:.5f}  \n"
        f"**Variance:** {variance:.5f}  \n"
        f"**Standard Deviation:** {std_dev:.5f}"
    )

    calc_type = st.selectbox(
        "Choose a probability calculation:",
        ["Exactly: P(X = x)", "At most: P(X ≤ x)", "Less than: P(X < x)",
         "At least: P(X ≥ x)", "Greater than: P(X > x)", "Between: P(a ≤ X ≤ b)",
         "Show table and graph"]
    )

    x = a = b = None
    if "Between" in calc_type:
        a = st.number_input("Enter lower bound (a):", min_value=0, max_value=int(x_max), step=1)
        b = st.number_input("Enter upper bound (b):", min_value=0, max_value=int(x_max), step=1)
        if a > b:
            st.warning("⚠️ Lower bound (a) must be ≤ upper bound (b).")
            return
    elif calc_type != "Show table and graph":
        x = st.number_input("Enter x value:", min_value=0, max_value=int(x_max), step=1)

    if st.button("📊 Calculate Poisson"):
        if calc_type == "Exactly: P(X = x)":
            prob = poisson.pmf(x, lam)
            st.success(f"P(X = {x}) = {prob:.5f}")
        elif calc_type == "P(X ≤ x)":
            prob = poisson.cdf(x, lam)
            st.success(f"P(X ≤ {x}) = {prob:.5f}")
        elif calc_type == "P(X < x)":
            prob = poisson.cdf(x - 1, lam) if x > 0 else 0.0
            st.success(f"P(X < {x}) = {prob:.5f}")
        elif calc_type == "P(X ≥ x)":
            prob = 1 - poisson.cdf(x - 1, lam) if x > 0 else 1.0
            st.success(f"P(X ≥ {x}) = {prob:.5f}")
        elif calc_type == "P(X > x)":
            prob = 1 - poisson.cdf(x, lam)
            st.success(f"P(X > {x}) = {prob:.5f}")
        elif calc_type == "Between: P(a ≤ X ≤ b)":
            prob = poisson.cdf(b, lam) - poisson.cdf(a - 1, lam) if a > 0 else poisson.cdf(b, lam)
            st.success(f"P({a} ≤ X ≤ {b}) = {prob:.5f}")
        elif calc_type == "Show table and graph":
            summary_df = pd.DataFrame({
                "x": x_vals,
                "P(X = x)": np.round(pmf_vals, 5)
            })
            st.dataframe(summary_df, use_container_width=True)

            fig = go.Figure(go.Bar(x=x_vals, y=pmf_vals, marker_color='salmon'))
            fig.update_layout(
                title=f'Poisson Distribution (λ={lam})',
                xaxis_title='x',
                yaxis_title='P(X = x)',
                template='simple_white'
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------- Main App ----------
def run():
    st.header("🎰 Probability Distribution Calculator")

    categories = [
        "Discrete Distribution",
        "Binomial Distribution",
        "Poisson Distribution"
    ]

    choice = st.selectbox(
        "Choose a category:",
        categories,
        index=None,
        placeholder="Select a distribution type..."
    )

    if not choice:
        st.info("👆 Please choose a category to begin.")
        return

    if choice == "Discrete Distribution":
        discrete_distribution_tool()
    elif choice == "Binomial Distribution":
        binomial_distribution_tool()
    elif choice == "Poisson Distribution":
        poisson_distribution_tool()

if __name__ == "__main__":
    run()
