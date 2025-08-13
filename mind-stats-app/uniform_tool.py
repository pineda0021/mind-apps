import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from fractions import Fraction

def fraction_str(value):
    try:
        return str(Fraction(value).limit_denominator())
    except Exception:
        return f"{value:.5f}"

def run():
    st.header("🎲 Continuous Uniform Distribution")

    # Inputs for uniform support
    a = st.number_input("Lower bound (a)", value=0.0)
    b = st.number_input("Upper bound (b)", value=1.0)

    if b <= a:
        st.error("⚠️ Upper bound b must be greater than lower bound a.")
        return

    width = b - a
    prob_per_unit = 1 / width

    st.markdown(f"**Each interval of length 1 has probability:** {fraction_str(prob_per_unit)} (i.e., 1 / {width})")

    # Show rectangles for unit intervals if integers
    if float(a).is_integer() and float(b).is_integer():
        ints = np.arange(int(a), int(b))
        st.markdown("### Probabilities for intervals of length 1:")
        for i in ints:
            st.write(f"P({i} ≤ X < {i+1}) = {fraction_str(prob_per_unit)}")

    # Single value query
    st.markdown("---")
    st.subheader("Single value or tail probability queries")

    x = st.number_input("Enter a value x to query probabilities", value=(a + b) / 2)

    query_type = st.selectbox("Choose a probability query:",
                              ["P(X = x)", "P(X ≥ x)", "P(X > x)", "P(X ≤ x)", "P(X < x)"])

    if st.button("📊 Calculate Probability for x"):
        if query_type == "P(X = x)":
            if a <= x <= b:
                st.info("For continuous uniform, P(X = x) = 0 for any single point.")
            else:
                st.info("x is outside the distribution support, so P(X = x) = 0.")

        elif query_type in ["P(X ≥ x)", "P(X > x)"]:
            if x < a:
                p = 1.0
            elif x > b:
                p = 0.0
            else:
                p = (b - x) / width
            st.markdown(f"**{query_type} = {p:.5f}**")

        elif query_type in ["P(X ≤ x)", "P(X < x)"]:
            if x < a:
                p = 0.0
            elif x > b:
                p = 1.0
            else:
                p = (x - a) / width
            st.markdown(f"**{query_type} = {p:.5f}**")

    # Interval query
    st.markdown("---")
    st.subheader("Interval probability query")

    lower_query = st.number_input(f"Lower bound of interval (≥ {a})", value=a, min_value=a, max_value=b, key="lower_query")
    upper_query = st.number_input(f"Upper bound of interval (≤ {b})", value=b, min_value=a, max_value=b, key="upper_query")

    if upper_query < lower_query:
        st.error("Upper bound must be greater than or equal to lower bound.")
    else:
        interval_length = max(0, upper_query - lower_query)
        interval_prob = interval_length / width
        st.markdown(f"**P({lower_query} < X < {upper_query}) = {interval_prob:.5f}**")

    # Plot PDF + shading
    xx = np.linspace(a - width * 0.2, b + width * 0.2, 500)
    pdf = uniform.pdf(xx, loc=a, scale=width)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xx, pdf, label="PDF")

    # Shade entire uniform range lightly
    ax.fill_between(xx, 0, pdf, color="lightgrey", alpha=0.3)

    # Shade queried interval
    if upper_query >= lower_query:
        ax.fill_between(xx, 0, pdf, where=(xx >= lower_query) & (xx <= upper_query), color="orange", alpha=0.6,
                        label=f"P({lower_query} < X < {upper_query})")

    # Shade single value tail queries if valid
    if 'Calculate Probability for x' in st.session_state and st.session_state.get('Calculate Probability for x'):
        if query_type in ["P(X ≥ x)", "P(X > x)"] and a <= x <= b:
            ax.fill_between(xx, 0, pdf, where=(xx >= x), color="skyblue", alpha=0.5, label=f"{query_type}")
        elif query_type in ["P(X ≤ x)", "P(X < x)"] and a <= x <= b:
            ax.fill_between(xx, 0, pdf, where=(xx <= x), color="lightgreen", alpha=0.5, label=f"{query_type}")

    # Draw rectangles for unit intervals if integer bounds
    if float(a).is_integer() and float(b).is_integer():
        ints = np.arange(int(a), int(b))
        for i in ints:
            rect = plt.Rectangle((i, 0), 1, prob_per_unit, alpha=0.3, color='blue')
            ax.add_patch(rect)
            ax.text(i + 0.5, prob_per_unit + 0.02, f"1/{int(width)}", ha='center', color='blue')

    ax.set_title(f"Uniform Distribution PDF on [{a}, {b}]")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

