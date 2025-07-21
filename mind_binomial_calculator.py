
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from fractions import Fraction

st.set_page_config(page_title="MIND: Binomial Calculator", layout="centered")

st.title("🎓 MIND: Making Inference Digestible")
st.header("Binomial Probability Calculator")

def parse_fraction(p_str):
    try:
        return float(Fraction(p_str.strip()))
    except Exception:
        st.error("❌ Please enter a valid decimal or fraction (e.g., 0.5 or 1/2).")
        return None

n = st.number_input("🔢 Number of trials (n)", min_value=1, step=1)
p_str = st.text_input("🎯 Probability of success (p)", value="1/2")

p = parse_fraction(p_str)

if p is not None and 0 <= p <= 1:
    calc_type = st.selectbox("📘 Choose a probability calculation:", 
                             ["P(X = x)", "P(X ≤ x)", "P(X < x)", "P(X ≥ x)", "P(X > x)", "Show table and graph"])

    if calc_type != "Show table and graph":
        x = st.number_input("📍 Enter x value:", min_value=0, max_value=int(n), step=1)

    if st.button("📊 Calculate"):
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

            st.subheader("📋 Binomial Probability Table")
            st.table({f'x={i}': [f'{val:.5f}'] for i, val in zip(x_vals, pmf_vals)})

            st.subheader("📈 Binomial Distribution Plot")
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

# Footer
st.markdown("---")
st.markdown("📘 Created by **Professor Edward Pineda-Castro**, Los Angeles City College")
st.markdown("_This tool was created with the students in **MIND** — Making Inference Digestible._")
