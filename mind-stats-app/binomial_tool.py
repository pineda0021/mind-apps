import streamlit as st
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
from fractions import Fraction

def run():
    st.header("🎲 Binomial Distribution Calculator")

    n = st.number_input("Number of trials (n):", min_value=1, step=1)
    p_input = st.text_input("Probability of success (p) (decimal or fraction):", "0.5")

    try:
        p = float(Fraction(p_input.strip()))
        if not 0 <= p <= 1:
            st.error("Probability p must be between 0 and 1.")
            return
    except:
        st.error("Invalid input for probability p.")
        return

    st.write("Choose a probability query:")
    option = st.selectbox("", [
        "P(X = x) Exactly x successes",
        "P(X ≤ x) At most x successes",
        "P(X ≥ x) At least x successes",
        "P(a ≤ X ≤ b) Between a and b successes",
        "View full distribution table and plot"
    ])

    if option == "View full distribution table and plot":
        x_vals = np.arange(0, n+1)
        probs = binom.pmf(x_vals, n, p)
        df = { "x": x_vals, "P(X=x)": np.round(probs, 5) }
        st.table(df)

        fig, ax = plt.subplots()
        ax.bar(x_vals, probs, color='skyblue', edgecolor='black')
        ax.set_title(f'Binomial Distribution (n={n}, p={p})')
        ax.set_xlabel('Number of Successes')
        ax.set_ylabel('Probability')
        st.pyplot(fig)
        return

    if option == "P(X = x) Exactly x successes":
        x = st.number_input("x (number of successes):", min_value=0, max_value=n, step=1)
        prob = binom.pmf(x, n, p)
        st.write(f"P(X = {x}) = {prob:.5f}")

    elif option == "P(X ≤ x) At most x successes":
        x = st.number_input("x (number of successes):", min_value=0, max_value=n, step=1)
        prob = binom.cdf(x, n, p)
        st.write(f"P(X ≤ {x}) = {prob:.5f}")

    elif option == "P(X ≥ x) At least x successes":
        x = st.number_input("x (number of successes):", min_value=0, max_value=n, step=1)
        prob = 1 - binom.cdf(x-1, n, p) if x > 0 else 1
        st.write(f"P(X ≥ {x}) = {prob:.5f}")

    elif option == "P(a ≤ X ≤ b) Between a and b successes":
        a = st.number_input("a (lower bound):", min_value=0, max_value=n, step=1)
        b = st.number_input("b (upper bound):", min_value=0, max_value=n, step=1)
        if b < a:
            st.error("Upper bound b must be ≥ lower bound a.")
        else:
            prob = binom.cdf(b, n, p) - binom.cdf(a-1, n, p) if a > 0 else binom.cdf(b, n, p)
            st.write(f"P({a} ≤ X ≤ {b}) = {prob:.5f}")
