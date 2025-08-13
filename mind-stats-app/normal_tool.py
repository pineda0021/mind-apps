import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def run():
    st.header("Normal Distribution")

    mu = st.number_input("Mean (μ)", value=0.0)
    sigma = st.number_input("Standard deviation (σ)", min_value=0.0001, value=1.0)

    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
    pdf = norm.pdf(x, mu, sigma)
    cdf = norm.cdf(x, mu, sigma)

    st.subheader("PDF")
    fig, ax = plt.subplots()
    ax.plot(x, pdf, label="PDF", color="blue")
    ax.fill_between(x, 0, pdf, alpha=0.3)
    ax.set_title(f"Normal Distribution PDF (μ={mu}, σ={sigma})")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("CDF")
    fig2, ax2 = plt.subplots()
    ax2.plot(x, cdf, label="CDF", color="green")
    ax2.set_title(f"Normal Distribution CDF (μ={mu}, σ={sigma})")
    ax2.grid(True)
    st.pyplot(fig2)

    st.markdown(f"**Mean:** {mu:.5f}")
    st.markdown(f"**Variance:** {(sigma**2):.5f}")
