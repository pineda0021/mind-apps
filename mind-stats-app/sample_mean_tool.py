import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def run():
    st.header("Distribution of the Sample Mean")

    mu = st.number_input("Population Mean (μ)", value=0.0)
    sigma = st.number_input("Population Standard Deviation (σ)", min_value=0.0001, value=1.0)
    n = st.number_input("Sample size (n)", min_value=1, step=1, value=30)

    se = sigma / np.sqrt(n)  # standard error
    x = np.linspace(mu - 4 * se, mu + 4 * se, 500)
    pdf = norm.pdf(x, mu, se)

    st.subheader("Sampling Distribution of the Sample Mean (Normal Approximation)")
    fig, ax = plt.subplots()
    ax.plot(x, pdf, label="PDF")
    ax.set_title(f"Sample Mean Distribution (n={n})")
    ax.grid(True)
    st.pyplot(fig)

    st.markdown(f"**Mean:** {mu:.5f}")
    st.markdown(f"**Standard Error:** {se:.5f}")
    st.markdown(f"**Variance:** {se**2:.5f}")
