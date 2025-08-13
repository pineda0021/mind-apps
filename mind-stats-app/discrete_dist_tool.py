import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
from fractions import Fraction
from math import exp, factorial

def parse_fraction(p_str):
    try:
        return float(Fraction(p_str.strip()))
    except Exception:
        st.error("❌ Please enter a valid decimal or fraction (e.g., 0.5 or 1/2).")
        return None

def discrete_distribution_tool():
    st.header("🎲 Discrete Probability Distribution Tool")

    st.write("Enter values of discrete random variable X and their probabilities P(X).")
    x_input = st.text_input("Values of X (comma-separated):")
    p_input = st.text_input("Probabilities P(X) (comma-separated, decimals or fractions):")

    if st.button("📊 Calculate"):
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
            "P(X)": np.round(P, 4),
            "Cumulative P(X ≤ x)": np.round(cdf, 4)
        })

        st.markdown(f"**Mean:** {mean:.4f}  \n**Variance:** {variance:.4f}  \n**Standard Deviation:** {std_dev:.4f}")

        fig = go.Figure(go.Bar(x=X, y=P, marker_color='green'))
        fig.update_layout(title="Probability Distribution", xaxis_title="X", yaxis_title="P(X)", template="simple_white")
        st.plotly_chart(fig)

def binomial_distribution_tool():
    st.header("🎲 Binomial Distribution Tool")

    n = st.number_input("Number of trials (n)", min_value=1, step=1)
    p_str = st.text_input("Probability of success (p)", value="1/2")
    p = parse_fraction(p_str)

    if p is not None and 0 <= p <= 1:
        x_vals = np.arange(0, n + 1)
        pmf_vals = binom.pmf(x_vals, n, p)
        mean = n * p
        variance = n * p * (1 - p)
        std_dev = np.sqrt(variance)
        cdf_vals = binom.cdf(x_vals, n, p)

        st.subheader("Binomial Distribution Summary")
        st.markdown(f"**Mean:** {mean:.4f}  \n**Variance:** {variance:.4f}  \n**Standard Deviation:** {std_dev:.4f}")

        calc_type = st.selectbox("Choose a probability calculation:", 
                                 ["P(X = x)", "P(X ≤ x)", "P(X < x)", "P(X ≥ x)", "P(X > x)", "Show table and graph"])

        if calc_type != "Show table and graph":
            x = st.number_input("Enter x value:", min_value=0, max_value=int(n), step=1)

        if st.button("Calculate"):
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
    else:
        if p is not None:
            st.warning("⚠️ Probability must be between 0 and 1.")

def poisson_distribution_tool():
    st.header("🎲 Poisson Distribution Tool")

    lam = st.number_input("Mean number of occurrences (λ)", min_value=0.0, format="%.4f")
    x_max = st.number_input("Max number of occurrences to consider (x max)", min_value=1, step=1)

    if lam > 0 and x_max >= 1:
        x_vals = np.arange(0, int(x_max) + 1)
        pmf_vals = poisson.pmf(x_vals, lam)
        mean = lam
        variance = lam
        std_dev = np.sqrt(variance)
        cdf_vals = poisson.cdf(x_vals, lam)

        st.subheader("Poisson Distribution Summary")
        st.markdown(f"**Mean:** {mean:.4f}  \n**Variance:** {variance:.4f}  \n**Standard Deviation:** {std_dev:.4f}")

        calc_type = st.selectbox("Choose a probability calculation:", 
                                 ["P(X = x)", "P(X ≤ x)", "P(X < x)", "P(X ≥ x)", "P(X > x)", "Show table and graph"])

        if calc_type != "Show table and graph":
            x = st.number_input("Enter x value:", min_value=0, max_value=int(x_max), step=1)

        if st.button("Calculate"):
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
                st.subheader("Poisson Probability Table")
                table_data = {f'x={i}': [f'{val:.5f}'] for i, val in zip(x_vals, pmf_vals)}
                st.table(table_data)

                st.subheader("Poisson Distribution Plot")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(x_vals, pmf_vals, color='lightcoral', edgecolor='black')
                ax.set_title(f'Poisson Distribution (λ={lam})')
                ax.set_xlabel('Number of Occurrences')
                ax.set_ylabel('Probability')
                ax.grid(axis='y')
                st.pyplot(fig)
    else:
        st.warning("⚠️ Please enter λ > 0 and x max ≥ 1.")

def run():
    st.sidebar.title("Choose a Distribution")
    choice = st.sidebar.radio("", ["Discrete Distribution", "Binomial Distribution", "Poisson Distribution"])

    if choice == "Discrete Distribution":
        discrete_distribution_tool()
    elif choice == "Binomial Distribution":
        binomial_distribution_tool()
    elif choice == "Poisson Distribution":
        poisson_distribution_tool()

