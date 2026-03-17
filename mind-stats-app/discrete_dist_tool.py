# ==========================================================
# Discrete Distributions Tool
# Clean Version (No Dark Mode Styling)
# Created by Professor Edward Pineda-Castro
# MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import binom, poisson
from fractions import Fraction


# ---------- Fraction Parser ----------
def parse_fraction(p_str):
    try:
        return float(Fraction(p_str.strip()))
    except Exception:
        st.error("❌ Please enter a valid decimal or fraction (e.g., 0.5 or 1/2).")
        return None


# ==========================================================
# Main App
# ==========================================================
def run():

    st.title("🎲 Discrete Probability Distributions")

    # ---------- GLOBAL DECIMAL SELECTOR ----------
    decimals = st.number_input(
        "Decimal places for output:",
        min_value=0,
        max_value=10,
        value=4,
        step=1
    )

    # ---------- Distribution Type ----------
    dist_type = st.radio(
        "Select Distribution Type:",
        ["Discrete", "Binomial", "Poisson"],
        horizontal=True
    )

    # ======================================================
    # 1. DISCRETE DISTRIBUTION
    # ======================================================
    if dist_type == "Discrete":

        st.subheader("🧮 Discrete Distribution")

        st.latex(r"\mu = \sum x_i p_i")
        st.latex(r"\sigma^2 = \sum p_i (x_i - \mu)^2")
        st.latex(r"\sigma = \sqrt{\sigma^2}")

        x_input = st.text_input("Values of X (comma-separated):", "0,1,2,3")
        p_input = st.text_input("Probabilities P(X) (comma-separated):", "1/8,3/8,3/8,1/8")

        if st.button("📊 Calculate Discrete Distribution"):

            try:
                X = np.array([float(x.strip()) for x in x_input.split(",")])
                P = np.array([float(Fraction(p.strip())) for p in p_input.split(",")])
            except:
                st.error("❌ Error parsing numbers.")
                return

            if len(X) != len(P):
                st.error("⚠️ X and P(X) must have the same length.")
                return

            if not np.isclose(P.sum(), 1):
                st.error(f"⚠️ Probabilities must sum to 1. Current sum = {P.sum():.5f}")
                return

            mu = np.sum(X * P)
            variance = np.sum(P * (X - mu) ** 2)
            sigma = np.sqrt(variance)

            df = pd.DataFrame({
                "x": X,
                "P(X = x)": np.round(P, decimals),
                "x·P(X)": np.round(X * P, decimals)
            }).sort_values("x").reset_index(drop=True)

            st.dataframe(df, use_container_width=True)

            st.markdown(f"""
            **Mean (μ)** = {round(mu, decimals)}  
            **Variance (σ²)** = {round(variance, decimals)}  
            **Standard Deviation (σ)** = {round(sigma, decimals)}
            """)

            fig = go.Figure()
            fig.add_bar(x=X, y=P)

            fig.update_layout(
                title="Discrete Probability Distribution",
                xaxis_title="x",
                yaxis_title="P(X = x)",
                template="plotly"
            )

            st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # 2. BINOMIAL DISTRIBUTION (FIXED)
    # ======================================================
    elif dist_type == "Binomial":

        st.subheader("🎯 Binomial Distribution")
        st.latex(r"P(X = x) = \binom{n}{x} p^x (1-p)^{n-x}")

        n = st.number_input("Number of trials (n):", min_value=1, step=1)
        p_str = st.text_input("Probability of success (p):", "1/2")

        p = parse_fraction(p_str)
        if p is None:
            return

        if not (0 <= p <= 1):
            st.error("⚠️ p must be between 0 and 1.")
            return

        x_vals = np.arange(0, n + 1)
        pmf_vals = binom.pmf(x_vals, n, p)

        μ = n * p
        σ = np.sqrt(n * p * (1 - p))

        st.markdown(f"""
        **Mean (μ)** = {round(μ, decimals)}  
        **Standard Deviation (σ)** = {round(σ, decimals)}
        """)

        calc = st.selectbox(
            "Select Probability Type:",
            [
                "Exactly: P(X = x)",
                "At most: P(X ≤ x)",
                "Less than: P(X < x)",
                "At least: P(X ≥ x)",
                "Greater than: P(X > x)",
                "Between: P(a ≤ X ≤ b)",
                "Show Full Table & Graph"
            ]
        )

        if "Between" in calc:
            a = st.number_input("Lower bound (a):", min_value=0, max_value=int(n))
            b = st.number_input("Upper bound (b):", min_value=0, max_value=int(n))
        elif "Full Table" not in calc:
            x = st.number_input("Enter x:", min_value=0, max_value=int(n))

        if st.button("📊 Calculate Binomial"):

            if calc == "Exactly: P(X = x)":
                prob = binom.pmf(x, n, p)

            elif calc == "At most: P(X ≤ x)":
                prob = binom.cdf(x, n, p)

            elif calc == "Less than: P(X < x)":
                prob = 0.0 if x <= 0 else binom.cdf(x - 1, n, p)
                st.caption(f"Using: P(X < {x}) = P(X ≤ {x-1})")

            elif calc == "At least: P(X ≥ x)":
                prob = 1.0 if x <= 0 else 1 - binom.cdf(x - 1, n, p)

            elif calc == "Greater than: P(X > x)":
                prob = 0.0 if x >= n else 1 - binom.cdf(x, n, p)

            elif calc == "Between: P(a ≤ X ≤ b)":
                if a > b:
                    st.error("⚠️ Lower bound must be ≤ upper bound.")
                    return
                lower = binom.cdf(a - 1, n, p) if a > 0 else 0.0
                prob = binom.cdf(b, n, p) - lower
                st.success(f"P({a} ≤ X ≤ {b}) = {round(prob, decimals)}")
                return

            elif "Full Table" in calc:

                df = pd.DataFrame({
                    "x": x_vals,
                    "P(X = x)": np.round(pmf_vals, decimals)
                })

                st.dataframe(df, use_container_width=True)

                fig = go.Figure()
                fig.add_bar(x=x_vals, y=pmf_vals)

                fig.update_layout(
                    title=f"Binomial Distribution (n={n}, p={p})",
                    xaxis_title="x",
                    yaxis_title="P(X = x)",
                    template="plotly"
                )

                st.plotly_chart(fig, use_container_width=True)
                return

            st.success(f"{calc.split(':')[1].replace('x', str(x))} = {round(prob, decimals)}")

    # ======================================================
    # 3. POISSON DISTRIBUTION (UNCHANGED)
    # ======================================================
    elif dist_type == "Poisson":

        st.subheader("💡 Poisson Distribution")
        st.latex(r"P(X = x) = \frac{e^{-λ} λ^x}{x!}")

        lam = st.number_input("Mean occurrences (λ):", min_value=0.01, format="%.4f")
        x_max = st.number_input("Maximum x to display:", min_value=1, step=1)

        x_vals = np.arange(0, int(x_max) + 1)
        pmf_vals = poisson.pmf(x_vals, lam)

        σ = np.sqrt(lam)

        st.markdown(f"""
        **Mean (μ)** = {round(lam, decimals)}  
        **Standard Deviation (σ)** = {round(σ, decimals)}
        """)

        calc = st.selectbox(
            "Select Probability Type:",
            [
                "Exactly: P(X = x)",
                "At most: P(X ≤ x)",
                "Less than: P(X < x)",
                "At least: P(X ≥ x)",
                "Greater than: P(X > x)",
                "Between: P(a ≤ X ≤ b)",
                "Show Full Table & Graph"
            ]
        )

        if "Between" in calc:
            a = st.number_input("Lower bound (a):", min_value=0, max_value=int(x_max))
            b = st.number_input("Upper bound (b):", min_value=0, max_value=int(x_max))
        elif "Full Table" not in calc:
            x = st.number_input("Enter x:", min_value=0, max_value=int(x_max))

        if st.button("📊 Calculate Poisson"):

            if calc == "Exactly: P(X = x)":
                st.success(f"P(X = {x}) = {round(poisson.pmf(x, lam), decimals)}")

            elif calc == "At most: P(X ≤ x)":
                st.success(f"P(X ≤ {x}) = {round(poisson.cdf(x, lam), decimals)}")

            elif calc == "Less than: P(X < x)":
                prob = poisson.cdf(x - 1, lam) if x > 0 else 0
                st.success(f"P(X < {x}) = {round(prob, decimals)}")

            elif calc == "At least: P(X ≥ x)":
                st.success(f"P(X ≥ {x}) = {round(1 - poisson.cdf(x - 1, lam), decimals)}")

            elif calc == "Greater than: P(X > x)":
                st.success(f"P(X > {x}) = {round(1 - poisson.cdf(x, lam), decimals)}")

            elif calc == "Between: P(a ≤ X ≤ b)":
                prob = poisson.cdf(b, lam) - (poisson.cdf(a - 1, lam) if a > 0 else 0)
                st.success(f"P({a} ≤ X ≤ {b}) = {round(prob, decimals)}")

            elif "Full Table" in calc:

                df = pd.DataFrame({
                    "x": x_vals,
                    "P(X = x)": np.round(pmf_vals, decimals)
                })

                st.dataframe(df, use_container_width=True)

                fig = go.Figure()
                fig.add_bar(x=x_vals, y=pmf_vals)

                fig.update_layout(
                    title=f"Poisson Distribution (λ={lam})",
                    xaxis_title="x",
                    yaxis_title="P(X = x)",
                    template="plotly"
                )

                st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# Run App
# ==========================================================
if __name__ == "__main__":
    run()
