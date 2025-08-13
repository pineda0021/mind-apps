import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

def run():
    st.header("📦 Poisson Distribution Calculator")

    lam = st.number_input("Enter the mean rate λ (lambda)", min_value=0.0, step=0.1, value=3.0)

    option = st.selectbox("Select probability calculation:",
                          ["P(X = x)", "P(X ≤ x)", "P(X ≥ x)", "P(a ≤ X ≤ b)"])

    if option == "P(X = x)":
        x = st.number_input("Number of events x", min_value=0, step=1)
        prob = poisson.pmf(x, lam)
        st.write(f"P(X = {x}) = {prob:.5f}")

    elif option == "P(X ≤ x)":
        x = st.number_input("Number of events x", min_value=0, step=1)
        prob = poisson.cdf(x, lam)
        st.write(f"P(X ≤ {x}) = {prob:.5f}")

    elif option == "P(X ≥ x)":
        x = st.number_input("Number of events x", min_value=0, step=1)
        prob = 1 - poisson.cdf(x - 1, lam)
        st.write(f"P(X ≥ {x}) = {prob:.5f}")

    elif option == "P(a ≤ X ≤ b)":
        a = st.number_input("Lower bound a", min_value=0, step=1)
        b = st.number_input("Upper bound b", min_value=0, step=1)
        if a > b:
            st.error("Lower bound must be ≤ upper bound.")
            return
        prob = poisson.cdf(b, lam) - poisson.cdf(a - 1, lam)
        st.write(f"P({a} ≤ X ≤ {b}) = {prob:.5f}")

    # Plot distribution
    max_x = int(lam + 4 * np.sqrt(lam))  # cover most of the distribution
    x_vals = np.arange(0, max_x + 1)
    y_vals = poisson.pmf(x_vals, lam)

    fig, ax = plt.subplots()
    ax.bar(x_vals, y_vals, color='lightcoral', edgecolor='black')
    ax.set_title(f'Poisson Distribution (λ={lam:.2f})')
    ax.set_xlabel('Number of Events')
    ax.set_ylabel('Probability')
    st.pyplot(fig)
