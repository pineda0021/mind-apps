import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

def run():
    st.header("Uniform Distribution")

    a = st.number_input("Lower bound (a)", value=0.0)
    b = st.number_input("Upper bound (b)", value=1.0)

    if b <= a:
        st.error("Upper bound b must be greater than lower bound a.")
        return

    x = np.linspace(a - (b - a) * 0.1, b + (b - a) * 0.1, 500)
    pdf = uniform.pdf(x, loc=a, scale=b - a)
    cdf = uniform.cdf(x, loc=a, scale=b - a)

    st.subheader("PDF")
    fig, ax = plt.subplots()
    ax.plot(x, pdf, label="PDF")
    ax.fill_between(x, 0, pdf, alpha=0.3)
    ax.set_title(f"Uniform Distribution PDF on [{a}, {b}]")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("CDF")
    fig2, ax2 = plt.subplots()
    ax2.plot(x, cdf, label="CDF", color="orange")
    ax2.set_title(f"Uniform Distribution CDF on [{a}, {b}]")
    ax2.grid(True)
    st.pyplot(fig2)

    mean = uniform.mean(loc=a, scale=b - a)
    var = uniform.var(loc=a, scale=b - a)
    st.markdown(f"**Mean:** {mean:.5f}")
    st.markdown(f"**Variance:** {var:.5f}")
