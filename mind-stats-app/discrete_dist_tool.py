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
BACKGROUND = "#2B2B2B"     # Dark-neutral background
TEXT = "white"             # Universal light text
ACCENT = "#4da3ff"         # Blue accent


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


# ---------- Fraction parser ----------
def parse_fraction(p_str):
    try:
        return float(Fraction(p_str.strip()))
    except Exception:
        st.error("❌ Please enter a valid decimal or fraction (e.g., 0.5 or 1/2).")
        return None


# ==========================================================
# Main Discrete Probability Distribution Tool
# ==========================================================
def run():

    st.markdown(f"<h1 style='color:{TEXT};'>🎲 Discrete Probability Distributions</h1>", unsafe_allow_html=True)

    dist_type = st.selectbox(
        "Choose a distribution type:",
        ["Custom Discrete", "Binomial", "Poisson"],
        index=None
    )

    # 👆 Friendly instruction
    if not dist_type:
        st.info("👆 Please select a category to begin.")
        return

    # ======================================================
    # 1. Custom Discrete Distribution
    # ======================================================
    if dist_type == "Custom Discrete":
        st.markdown(f"<h2 style='color:{TEXT};'>🧮 Custom Discrete Distribution</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{TEXT};'>Enter values of X and their probabilities P(X).</p>", unsafe_allow_html=True)

        x_input = st.text_input("Values of X (comma-separated):", "0,1,2,3")
        p_input = st.text_input("Probabilities P(X) (comma-separated):", "1/8,3/8,3/8,1/8")

        if st.button("📊 Calculate Distribution"):
            if not x_input or not p_input:
                st.warning("Please enter both X and P(X) values.")
                return

            try:
                X = np.array([float(x.strip()) for x in x_input.split(",")])
                P = np.array([float(Fraction(p.strip())) for p in p_input.split(",")])
            except Exception as e:
                st.error(f"Error parsing input: {e}")
                return

            if len(X) != len(P):
                st.error("⚠️ X and P(X) must have the same length.")
                return

            if not np.isclose(P.sum(), 1):
                st.error(f"⚠️ Probabilities must sum to 1. Current sum = {P.sum():.4f}")
                return

            μ = np.sum(X * P)
            σ2 = np.sum(P * (X - μ) ** 2)
            σ = np.sqrt(σ2)

            df = pd.DataFrame({
                "x": X,
                "P(X = x)": np.round(P, 5),
                "x·P(X)": np.round(X * P, 5)
            }).sort_values("x").reset_index(drop=True)

            st.dataframe(df, use_container_width=True)

            st.markdown(f"""
            <p style='color:{TEXT};'>
            • <b>Mean:</b> μ = {μ:.5f}<br>
            • <b>Variance:</b> σ² = {σ2:.5f}<br>
            • <b>Standard Deviation:</b> σ = {σ:.5f}
            </p>
            """, unsafe_allow_html=True)

            # --- Plot (Dark/Light Safe) ---
            fig = go.Figure()
            fig.add_bar(x=X, y=P, marker=dict(color=ACCENT))

            fig.update_layout(
                title=dict(text="Custom Discrete Distribution", font=dict(color=TEXT)),
                xaxis=dict(title="x", color=TEXT, showgrid=False),
                yaxis=dict(title="P(X = x)", color=TEXT, showgrid=False),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor=BACKGROUND,
                font=dict(color=TEXT)
            )

            st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # 2. Binomial Distribution
    # ======================================================
    elif dist_type == "Binomial":
        st.markdown(f"<h2 style='color:{TEXT};'>🎯 Binomial Distribution</h2>", unsafe_allow_html=True)
        st.write("A binomial experiment consists of n independent trials with probability p of success.")
        st.latex(r"P(X = x) = \binom{n}{x} p^x (1-p)^{n-x}")

        n = st.number_input("Number of trials (n):", min_value=1, step=1)
        p_str = st.text_input("Probability of success (p):", "1/2")
        p = parse_fraction(p_str)

        if p is None or not (0 <= p <= 1):
            return

        x_vals = np.arange(0, n + 1)
        pmf_vals = binom.pmf(x_vals, n, p)

        μ = n * p
        σ2 = n * p * (1 - p)
        σ = np.sqrt(σ2)

        st.markdown(f"""
        <p style='color:{TEXT};'>
        • <b>Mean:</b> μ = {μ:.5f}<br>
        • <b>Variance:</b> σ² = {σ2:.5f}<br>
        • <b>Standard Deviation:</b> σ = {σ:.5f}
        </p>
        """, unsafe_allow_html=True)

        calc = st.selectbox("Select Probability Type:", [
            "Exactly: P(X = x)",
            "At most: P(X ≤ x)",
            "Less than: P(X < x)",
            "At least: P(X ≥ x)",
            "Greater than: P(X > x)",
            "Between: P(a ≤ X ≤ b)",
            "Show Full Table & Graph"
        ])

        if "Between" in calc:
            a = st.number_input("Lower bound (a):", min_value=0, max_value=int(n))
            b = st.number_input("Upper bound (b):", min_value=0, max_value=int(n))
        elif "Full Table" not in calc:
            x = st.number_input("Enter x value:", min_value=0, max_value=int(n))
        else:
            x = a = b = None

        if st.button("📊 Calculate Binomial"):

            if calc == "Exactly: P(X = x)":
                st.success(f"P(X = {x}) = {binom.pmf(x, n, p):.5f}")

            elif calc == "At most: P(X ≤ x)":
                st.success(f"P(X ≤ {x}) = {binom.cdf(x, n, p):.5f}")

            elif calc == "Less than: P(X < x)":
                prob = binom.cdf(x - 1, n, p) if x > 0 else 0
                st.success(f"P(X < {x}) = {prob:.5f}")

            elif calc == "At least: P(X ≥ x)":
                st.success(f"P(X ≥ {x}) = {1 - binom.cdf(x - 1, n, p):.5f}")

            elif calc == "Greater than: P(X > x)":
                st.success(f"P(X > {x}) = {1 - binom.cdf(x, n, p):.5f}")

            elif calc == "Between: P(a ≤ X ≤ b)":
                prob = binom.cdf(b, n, p) - (binom.cdf(a - 1, n, p) if a > 0 else 0)
                st.success(f"P({a} ≤ X ≤ {b}) = {prob:.5f}")

            elif "Full Table" in calc:
                df = pd.DataFrame({"x": x_vals, "P(X = x)": np.round(pmf_vals, 5)})
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
    # 3. Poisson Distribution
    # ======================================================
    elif dist_type == "Poisson":
        st.markdown(f"<h2 style='color:{TEXT};'>💡 Poisson Distribution</h2>", unsafe_allow_html=True)
        st.write("The Poisson distribution models occurrences within a fixed interval.")
        st.latex(r"P(X = x) = \frac{e^{-λ} λ^x}{x!}")

        lam = st.number_input("Mean occurrences (λ):", min_value=0.0, format="%.4f")
        x_max = st.number_input("Maximum x to display:", min_value=1, step=1)

        if lam <= 0:
            return

        x_vals = np.arange(0, int(x_max) + 1)
        pmf_vals = poisson.pmf(x_vals, lam)

        st.markdown(f"""
        <p style='color:{TEXT};'>
        • <b>Mean:</b> μ = {lam:.5f}<br>
        • <b>Variance:</b> σ² = {lam:.5f}<br>
        • <b>Standard Deviation:</b> σ = {np.sqrt(lam):.5f}
        </p>
        """, unsafe_allow_html=True)

        calc = st.selectbox("Select Probability Type:", [
            "Exactly: P(X = x)",
            "At most: P(X ≤ x)",
            "Less than: P(X < x)",
            "At least: P(X ≥ x)",
            "Greater than: P(X > x)",
            "Between: P(a ≤ X ≤ b)",
            "Show Full Table & Graph"
        ])

        if "Between" in calc:
            a = st.number_input("Lower bound (a):", min_value=0, max_value=int(x_max))
            b = st.number_input("Upper bound (b):", min_value=0, max_value=int(x_max))
        elif "Full Table" not in calc:
            x = st.number_input("Enter x:", min_value=0, max_value=int(x_max))
        else:
            x = a = b = None

        if st.button("📊 Calculate Poisson"):

            if calc == "Exactly: P(X = x)":
                st.success(f"P(X = {x}) = {poisson.pmf(x, lam):.5f}")

            elif calc == "At most: P(X ≤ x)":
                st.success(f"P(X ≤ {x}) = {poisson.cdf(x, lam):.5f}")

            elif calc == "Less than: P(X < x)":
                prob = poisson.cdf(x - 1, lam) if x > 0 else 0
                st.success(f"P(X < {x}) = {prob:.5f}")

            elif calc == "At least: P(X ≥ x)":
                st.success(f"P(X ≥ {x}) = {1 - poisson.cdf(x - 1, lam):.5f}")

            elif calc == "Greater than: P(X > x)":
                st.success(f"P(X > {x}) = {1 - poisson.cdf(x, lam):.5f}")

            elif calc == "Between: P(a ≤ X ≤ b)":
                prob = poisson.cdf(b, lam) - (poisson.cdf(a - 1, lam) if a > 0 else 0)
                st.success(f"P({a} ≤ X ≤ {b}) = {prob:.5f}")

            elif "Full Table" in calc:
                df = pd.DataFrame({"x": x_vals, "P(X = x)": np.round(pmf_vals, 5)})
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
