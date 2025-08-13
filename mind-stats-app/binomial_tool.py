import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from fractions import Fraction

def parse_fraction(p_str):
    try:
        return float(Fraction(p_str.strip()))
    except ValueError:
        return None

def display_binomial_table(n, p):
    table_data = {
        "x": list(range(n+1)),
        "P(X = x)": [binom.pmf(x, n, p) for x in range(n+1)]
    }
    st.table(table_data)

def display_binomial_plot(n, p):
    x = np.arange(0, n + 1)
    y = binom.pmf(x, n, p)
    fig, ax = plt.subplots()
    ax.bar(x, y, color='skyblue', edgecolor='black')
    ax.set_title(f'Binomial Distribution (n={n}, p={p})')
    ax.set_xlabel('Number of Successes')
    ax.set_ylabel('Probability')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

def run():
    st.header("🎲 Binomial Probability Calculator")

    n = st.number_input("Enter the number of trials (n):", min_value=1, value=5)
    p_input = st.text_input("Enter the probability of success (p — decimals or fractions):", "1/2")
    p = parse_fraction(p_input)

    if p is None or not (0 <= p <= 1):
        st.error("❌ Probability must be between 0 and 1.")
        return

    option = st.selectbox("Choose probability calculation:", [
        "Exactly x successes",
        "At most x successes",
        "At least x successes",
        "Between x and y successes",
        "More than x successes",
        "Fewer than x successes",
        "View Probability Table and Graph"
    ])

    if option == "Exactly x successes":
        x = st.number_input("Enter x:", min_value=0, max_value=n, value=2)
        st.write(f"P(X = {x}) = {binom.pmf(x, n, p):.5f}")

    elif option == "At most x successes":
        x = st.number_input("Enter x:", min_value=0, max_value=n, value=2)
        st.write(f"P(X ≤ {x}) = {binom.cdf(x, n, p):.5f}")

    elif option == "At least x successes":
        x = st.number_input("Enter x:", min_value=0, max_value=n, value=2)
        st.write(f"P(X ≥ {x}) = {1 - binom.cdf(x - 1, n, p):.5f}")

    elif option == "Between x and y successes":
        a = st.number_input("Enter lower bound (a):", min_value=0, max_value=n, value=1)
        b = st.number_input("Enter upper bound (b):", min_value=a, max_value=n, value=3)
        st.write(f"P({a} ≤ X ≤ {b}) = {binom.cdf(b, n, p) - binom.cdf(a - 1, n, p):.5f}")

    elif option == "More than x successes":
        x = st.number_input("Enter x:", min_value=0, max_value=n, value=2)
        st.write(f"P(X > {x}) = {1 - binom.cdf(x, n, p):.5f}")

    elif option == "Fewer than x successes":
        x = st.number_input("Enter x:", min_value=0, max_value=n, value=2)
        st.write(f"P(X < {x}) = {binom.cdf(x - 1, n, p):.5f}")

    elif option == "View Probability Table and Graph":
        display_binomial_table(n, p)
        display_binomial_plot(n, p)
