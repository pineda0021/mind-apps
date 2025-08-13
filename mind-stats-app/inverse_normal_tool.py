import streamlit as st
import numpy as np
from scipy.stats import norm

def run():
    st.header("Inverse Normal Distribution (Quantile Function)")

    mu = st.number_input("Mean (μ)", value=0.0)
    sigma = st.number_input("Standard deviation (σ)", min_value=0.0001, value=1.0)
    p = st.slider("Probability p", min_value=0.0, max_value=1.0, value=0.5)

    if p == 0.0:
        st.error("Probability p cannot be 0 for inverse normal.")
        return
    elif p == 1.0:
        st.error("Probability p cannot be 1 for inverse normal.")
        return

    quantile = norm.ppf(p, mu, sigma)

    st.markdown(f"Quantile (x) where P(X ≤ x) = {p:.4f} is: **{quantile:.5f}**")

    # Optional: Show standard normal curve and mark quantile
    import matplotlib.pyplot as plt
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
    pdf = norm.pdf(x, mu, sigma)
    fig, ax = plt.subplots()
    ax.plot(x, pdf, label="PDF")
    ax.axvline(quantile, color='red', linestyle='--', label=f'Quantile at p={p:.2f}')
    ax.legend()
    ax.set_title("Normal Distribution PDF with Quantile")
    st.pyplot(fig)
