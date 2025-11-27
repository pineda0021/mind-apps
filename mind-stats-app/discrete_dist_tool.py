# ==========================================================
# Discrete Distributions Tool
# Updated for Universal Readability (Dark & Light Mode Safe)
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import binom, poisson
from fractions import Fraction

# ---------- Universal Readability Colors ----------
BACKGROUND = "#2B2B2B"
TEXT = "white"
ACCENT = "#4da3ff"


# ---------- Helper Step Box ----------
def step_box(text):
    st.markdown(
        f"""
        <div style="
            background-color:{BACKGROUND};
            padding:12px;
            border-radius:10px;
            border-left:6px solid {ACCENT};
            margin-bottom:12px;">
            <p style="color:{TEXT};margin:0;font-weight:bold;">{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    st.markdown(f"<h1 style='color:{TEXT};'>🎲 Discrete Probability Distributions</h1>", unsafe_allow_html=True)

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
        st.markdown(f"<h2 style='color:{TEXT};'>🧮 Discrete Distribution</h2>", unsafe_allow_html=True)

        st.latex(r"P(X = x_i) = p_i,\quad \sum p_i = 1")
        st.latex(r"\mu = \sum x_i p_i")
        st.latex(r"\sigma = \sqrt{\sum p_i (x_i - \mu)^2}")

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

            # Mean
            μ = np.sum(X * P)

            # Standard deviation
            σ = np.sqrt(np.sum(P * (X - μ) ** 2))

            # Table
            df = pd.DataFrame({
                "x": X,
                "P(X = x)": np.round(P, decimals),
                "x·P(X)": np.round(X * P, decimals)
            }).sort_values("x").reset_index(drop=True)

            st.dataframe(df, use_container_width=True)

            st.markdown(
                f"""
                <p style='color:{TEXT};'>
                • <b>Mean:</b> μ = {round(μ, decimals)}<br>
                • <b>Standard Deviation:</b> σ = {round(σ, decimals)}
                </p>
                """,
                unsafe_allow_html=True
            )

            # Plot
            fig = go.Figure()
            fig.add_bar(x=X, y=P, marker=dict(color=ACCENT))

            fig.update_layout(
                title=dict(text="Discrete Probability Distribution", font=dict(color=TEXT)),
                xaxis=dict(title="x", color=TEXT, showgrid=False),
                yaxis=dict(title="P(X = x)", color=TEXT, showgrid=False),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor=BACKGROUND,
                font=dict(color=TEXT)
            )

            st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # 2. BINOMIAL DISTRIBUTION
    # ======================================================
    elif dist_type == "Binomial":
        st.markdown(f"<h2 style='color:{TEXT};'>🎯 Binomial Distribution</h2>", unsafe_allow_html=True)

        st.latex(r"P(X = x) = \binom{n}{x} p^x (1-p)^{n-x}")

        n = st.number_input("Number of trials (n):", min_value=1, step=1)
        p_str = st.text_input("Probability of success (p):", "1/2")

        p = parse_fraction(p_str)
        if p is None:
            return

        x_vals = np.arange(0, n + 1)
        pmf_vals = binom.pmf(x_vals, n, p)

        μ = n * p
        σ = np.sqrt(n * p * (1 - p))

        st.markdown(
            f"""
            <p style='color:{TEXT};'>
            • <b>Mean:</b> μ = {round(μ, decimals)}<br>
            • <b>Standard Deviation:</b> σ = {round(σ, decimals)}
            </p>
            """,
            unsafe_allow_html=True
        )

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
                st.success(f"P(X = {x}) = {round(binom.pmf(x, n, p), decimals)}")

            elif calc == "At most: P(X ≤ x)":
                st.success(f"P(X ≤ {x}) = {round(binom.cdf(x, n, p), decimals)}")

            elif calc == "Less than: P(X < x)":
                prob = binom.cdf(x - 1, n, p) if x > 0 else 0
                st.success(f"P(X < {x}) = {round(prob, decimals)}")

            elif calc == "At least: P(X ≥ x)":
                st.success(f"P(X ≥ {x}) = {round(1 - binom.cdf(x - 1, n, p), decimals)}")

            elif calc == "Greater than: P(X > x)":
                st.success(f"P(X > {x}) = {round(1 - binom.cdf(x, n, p), decimals)}")

            elif calc == "Between: P(a ≤ X ≤ b)":
                prob = binom.cdf(b, n, p) - (binom.cdf(a - 1, n, p) if a > 0 else 0)
                st.success(f"P({a} ≤ X ≤ {b}) = {round(prob, decimals)}")

            elif "Full Table" in calc:
                df = pd.DataFrame({"x": x_vals, "P(X = x)": np.round(pmf_vals, decimals)})
                st.dataframe(df, use_container_width=True)

                fig = go.Figure()
                fig.add_bar(x=x_vals, y=pmf_vals, marker=dict(color=ACCENT))

                fig.update_layout(
                    title=dict(text=f"Binomial Distribution (n={n}, p={p})", font=dict(color=TEXT)),
                    xaxis=dict(title="x", color=TEXT, showgrid=False),
                    yaxis=dict(title="P(X = x)", color=TEXT, showgrid=False),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor=BACKGROUND,
                    font=dict(color=TEXT)
                )

                st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # 3. POISSON DISTRIBUTION
    # ======================================================
    elif dist_type == "Poisson":
        st.markdown(f"<h2 style='color:{TEXT};'>💡 Poisson Distribution</h2>", unsafe_allow_html=True)

        st.latex(r"P(X = x) = \frac{e^{-λ} λ^x}{x!}")

        lam = st.number_input("Mean occurrences (λ):", min_value=0.01, format="%.4f")
        x_max = st.number_input("Maximum x to display:", min_value=1, step=1)

        x_vals = np.arange(0, int(x_max) + 1)
        pmf_vals = poisson.pmf(x_vals, lam)

        σ = np.sqrt(lam)

        st.markdown(
            f"""
            <p style='color:{TEXT};'>
            • <b>Mean:</b> μ = {round(lam, decimals)}<br>
            • <b>Standard Deviation:</b> σ = {round(σ, decimals)}
            </p>
            """,
            unsafe_allow_html=True
        )

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
                df = pd.DataFrame({"x": x_vals, "P(X = x)": np.round(pmf_vals, decimals)})
                st.dataframe(df, use_container_width=True)

                fig = go.Figure()
                fig.add_bar(x=x_vals, y=pmf_vals, marker=dict(color=ACCENT))

                fig.update_layout(
                    title=dict(text=f"Poisson Distribution (λ={lam})", font=dict(color=TEXT)),
                    xaxis=dict(title="x", color=TEXT, showgrid=False),
                    yaxis=dict(title="P(X = x)", color=TEXT, showgrid=False),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor=BACKGROUND,
                    font=dict(color=TEXT)
                )

                st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# Run App
# ==========================================================
if __name__ == "__main__":
    run()
