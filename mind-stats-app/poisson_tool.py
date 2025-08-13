import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

def display_poisson_table(lmbda):
    table_data = {
        "x": list(range(0, int(lmbda*3) + 1)),
        "P(X = x)": [poisson.pmf(x, lmbda) for x in range(0, int(lmbda*3) + 1)]
    }
    st.table(table_data)

def display_poisson_plot(lmbda):
    x = np.arange(0, int(lmbda*3) + 1)
    y = poisson.pmf(x, lmbda)
    fig, ax = plt.subplots()
    ax.bar(x, y, color='salmon', edgecolor='black')
    ax.set_title(f'Poisson Distribution (λ={lmbda})')
    ax.set_xlabel('Number of Events')
    ax.set_ylabel('Probability')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

def run():
    st.header("📦 Poisson Probability Calculator")

    lmbda = st.number_input("Enter the average rate (λ):", min_value=0.01, value=2.0)

    option = st.selectbox("Choose probability calculation:", [
        "Exactly x events",
        "At most x events",
        "At least x events",
        "Between x and y events",
        "More than x events",
        "Fewer than x events",
        "View Probability Table and Graph"
    ])

    if option == "Exactly x events":
        x = st.number_input("Enter x:", min_value=0, value=2)
        st.write(f"P(X = {x}) = {poisson.pmf(x, lmbda):.5f}")

    elif option == "At most x events":
        x = st.number_input("Enter x:", min_value=0, value=2)
        st.write(f"P(X ≤ {x}) = {poisson.cdf(x, lmbda):.5f}")

    elif option == "At least x events":
        x = st.number_input("Enter x:", min_value=0, value=2)
        st.write(f"P(X ≥ {x}) = {1 - poisson.cdf(x - 1, lmbda):.5f}")

    elif option == "Between x and y events":
        a = st.number_input("Enter lower bound (a):", min_value=0, value=1)
        b = st.number_input("Enter upper bound (b):", min_value=a, value=3)
        st.write(f"P({a} ≤ X ≤ {b}) = {poisson.cdf(b, lmbda) - poisson.cdf(a - 1, lmbda):.5f}")

    elif option == "More than x events":
        x = st.number_input("Enter x:", min_value=0, value=2)
        st.write(f"P(X > {x}) = {1 - poisson.cdf(x, lmbda):.5f}")

    elif option == "Fewer than x events":
        x = st.number_input("Enter x:", min_value=0, value=2)
        st.write(f"P(X < {x}) = {poisson.cdf(x - 1, lmbda):.5f}")

    elif option == "View Probability Table and Graph":
        display_poisson_table(lmbda)
        display_poisson_plot(lmbda)
