import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from fractions import Fraction
from scipy.stats import binom, poisson

st.set_page_config(page_title="MIND: Discrete Distributions Suite", layout="centered")

st.title("🎓 MIND: Making Inference Digestible")
st.header("Discrete Probability Distributions")

def parse_fraction(p_str):
    try:
        return float(Fraction(p_str.strip()))
    except Exception:
        st.error("❌ Please enter a valid decimal or fraction (e.g., 0.5 or 1/2).")
        return None

def discrete_distribution_tool():
    st.subheader("🎲 Discrete Probability Distribution Tool")
    st.write("Enter values of discrete random variable X and their probabilities P(X).")
    
    x_input = st.text_input("Values of X (comma-separated):")
    p_input = st.text_input("Probabilities P(X) (comma-separated, decimals or fractions):")

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
    cdf = np.cumsum(P)

    st.subheader("Discrete Distribution Summary")
    st.dataframe({
        "X": X,
        "P(X)": P.round(4),
        "Cumulative P(X ≤ x)": cdf.round(4)
    })

    st.markdown(f"**Mean:** {mean:.4f}  \n**Variance:** {variance:.4f}  \n**Standard Deviation:** {std_dev:.4f}")

    fig = go.Figure(go.Bar(x=X, y=P, marker_color='green'))
    fig.update_layout(title="Probability Distribution", xaxis_title="X", yaxis_title="P(X)", template="simple_white")
    st.plotly_chart(fig)

def binomial_distribution_tool():
    st.subheader("🎯 Binomial Probability Calculator")

    n = st.number_input("Number of trials (n)", min_value=1, step=1)
    p_str = st.text_input("Probability of success (p)", value="1/2")
    p = parse_fraction(p_str)
    if p is None or not (0 <= p <= 1):
        if p is not None:
            st.warning("⚠️ Probability must be between 0 and 1.")
        return

    calc_type = st.selectbox("Choose a probability calculation:", 
                             ["P(X = x)", "P(X ≤ x)", "P(X < x)", "P(X ≥ x)", "P(X > x)", "Show table and graph"])

    if calc_type != "Show table and graph":
        x = st.number_input("Enter x value:", min_value=0, max_value=int(n), step=1)

    if st.button("Calculate Binomial Probability"):
        if calc_type == "P(X = x)":
            st.success(f"P(X = {x}) = {binom.pmf(x, n, p):.5f}")
        elif calc_type == "P(X ≤ x)":
            st.success(f"P(X ≤ {x}) = {binom.cdf(x, n, p):.5f}")
        elif calc_type == "P(X < x)":
            st.success(f"P(X < {x}) = {binom.cdf(x - 1, n, p):.5f}")
        elif calc_type == "P(X ≥ x)":
            st.success(f"P(X ≥ {x}) = {1 - binom.cdf(x - 1, n, p):.5f}")
        elif calc_type == "P(X > x)":
            st.success(f"P(X > {x}) = {1 - binom.cdf(x, n, p):.5f}")
        elif calc_type == "Show table and graph":
            x_vals = np.arange(0, n + 1)
            pmf_vals = binom.pmf(x_vals, n, p)

            st.subheader("Binomial Probability Table")
            table_data = {f'x={i}': [f'{val:.5f}'] for i, val in zip(x_vals, pmf_vals)}
            st.table(table_data)

            st.subheader("Binomial Distribution Plot")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x_vals, pmf_vals, color='skyblue', edgecolor='black')
            ax.set_title(f'Binomial Distribution (n={n}, p={p})')
            ax.set_xlabel('Number of Successes')
            ax.set_ylabel('Probability')
            ax.grid(axis='y')
            st.pyplot(fig)

def poisson_distribution_tool():
    st.subheader("🐑 Poisson Probability Calculator")

    lam_str = st.text_input("Enter λ (average rate):", value="3")
    try:
        lam = float(lam_str)
        if lam <= 0:
            st.warning("⚠️ λ must be a positive number.")
            return
    except Exception:
        st.error("❌ Please enter a valid number for λ.")
        return

    calc_type = st.selectbox("Choose a probability calculation:", 
                             ["P(X = x)", "P(X ≤ x)", "P(X < x)", "P(X ≥ x)", "P(X > x)", "Show table and graph"])

    if calc_type != "Show table and graph":
        x = st.number_input("Enter x value:", min_value=0, step=1)

    if st.button("Calculate Poisson Probability"):
        if calc_type == "P(X = x)":
            st.success(f"P(X = {x}) = {poisson.pmf(x, lam):.5f}")
        elif calc_type == "P(X ≤ x)":
            st.success(f"P(X ≤ {x}) = {poisson.cdf(x, lam):.5f}")
        elif calc_type == "P(X < x)":
            st.success(f"P(X < {x}) = {poisson.cdf(x - 1, lam):.5f}")
        elif calc_type == "P(X ≥ x)":
            st.success(f"P(X ≥ {x}) = {1 - poisson.cdf(x - 1, lam):.5f}")
        elif calc_type == "P(X > x)":
            st.success(f"P(X > {x}) = {1 - poisson.cdf(x, lam):.5f}")
        elif calc_type == "Show table and graph":
            max_x = max(15, int(lam * 3))
            x_vals = np.arange(0, max_x + 1)
            pmf_vals = poisson.pmf(x_vals, lam)

            st.subheader("Poisson Probability Table")
            table_data = {f'x={i}': [f'{val:.5f}'] for i, val in zip(x_vals, pmf_vals)}
            st.table(table_data)

            st.subheader("Poisson Distribution Plot")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x_vals, pmf_vals, color='lightcoral', edgecolor='black')
            ax.set_title(f'Poisson Distribution (λ={lam})')
            ax.set_xlabel('Number of Events')
            ax.set_ylabel('Probability')
            ax.grid(axis='y')
            st.pyplot(fig)

# Main routing logic
choice = st.sidebar.selectbox("Choose Distribution", 
                              ["Discrete Distribution", "Binomial Distribution", "Poisson Distribution"])

if choice == "Discrete Distribution":
    discrete_distribution_tool()
elif choice == "Binomial Distribution":
    binomial_distribution_tool()
elif choice == "Poisson Distribution":
    poisson_distribution_tool()
