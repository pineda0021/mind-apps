import streamlit as st
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from fractions import Fraction

def run():
    st.header("🕒 Poisson Distribution Calculator")

    lam_input = st.text_input("Enter the mean (λ):", "1.0")

    try:
        lam = float(Fraction(lam_input.strip()))
        if lam <= 0:
            st.error("λ must be a positive number.")
            return
    except:
        st.error("Invalid input for λ.")
        return

    st.write("Choose a probability query:")
    option = st.selectbox("", [
        "P(X = x) Exactly x events",
        "P(X ≤ x) At most x events",
        "P(X ≥ x) At least x events",
        "P(a ≤ X ≤ b) Between a and b events",
        "View full distribution table and plot"
    ])

    max_x = max(15, int(lam + 4 * np.sqrt(lam)))  # heuristic range for plot

    if option == "View full distribution table and plot":
        x_vals = np.arange(0, max_x + 1)
        probs = poisson.pmf(x_vals, lam)
        df = { "x": x_vals, "P(X=x)": np.round(probs, 5) }
        st.table(df)

        fig, ax = plt.subplots()
        ax.bar(x_vals, probs, color='lightgreen', edgecolor='black')
        ax.set_title(f'Poisson Distribution (λ={lam})')
        ax.set_xlabel('Number of Events')
        ax.set_ylabel('Probability')
        st.pyplot(fig)
        return

    if option == "P(X = x) Exactly x events":
        x = st.number_input("x (number of events):", min_value=0, step=1)
        prob = poisson.pmf(x, lam)
        st.write(f"P(X = {x}) = {prob:.5f}")

    elif option == "P(X ≤ x) At most x events":
        x = st.number_input("x (number of events):", min_value=0, step=1)
        prob = poisson.cdf(x, lam)
        st.write(f"P(X ≤ {x}) = {prob:.5f}")

    elif option == "P(X ≥ x) At least x events":
        x = st.number_input("x (number of events):", min_value=0, step=1)
        prob = 1 - poisson.cdf(x - 1, lam) if x > 0 else 1
        st.write(f"P(X ≥ {x}) = {prob:.5f}")

    elif option == "P(a ≤ X ≤ b) Between a and b events":
        a = st.number_input("a (lower bound):", min_value=0, step=1)
        b = st.number_input("b (upper bound):", min_value=0, step=1)
        if b < a:
            st.error("Upper bound b must be ≥ lower bound a.")
        else:
            prob = poisson.cdf(b, lam) - poisson.cdf(a - 1, lam) if a > 0 else poisson.cdf(b, lam)
            st.write(f"P({a} ≤ X ≤ {b}) = {prob:.5f}")
