import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def run():
    st.header("Distribution of the Sample Proportion")

    p = st.number_input("Population Proportion (p)", min_value=0.0, max_value=1.0, value=0.5)
    n = st.number_input("Sample size (n)", min_value=1, step=1, value=30)

    se = np.sqrt(p * (1 - p) / n)  # standard error
    x = np.linspace(max(0, p - 4 * se), min(1, p + 4 * se), 500)
    pdf = norm.pdf(x, p, se)

    st.subheader("Sampling Distribution of the Sample Proportion (Normal Approximation)")
    fig, ax = plt.subplots()
    ax.plot(x, pdf, label="PDF")
    ax.set_title(f"Sample Proportion Distribution (n={n})")
    ax.grid(True)
    st.pyplot(fig)

    st.markdown(f"**Mean (p):** {p:.5f}")
    st.markdown(f"**Standard Error:** {se:.5f}")
    st.markdown(f"**Variance:** {se**2:.5f}")
