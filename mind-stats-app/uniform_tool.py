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
    st.header("Uniform Distribution")

    a = st.number_input("Lower bound (a)", value=0.0)
    b = st.number_input("Upper bound (b)", value=1.0)

    if b <= a:
        st.error("Upper bound b must be greater than lower bound a.")
        return

    width = b - a
    prob_per_unit = 1 / width

    st.markdown(f"**Each interval of length 1 has probability:** {fraction_str(prob_per_unit)} (i.e., 1 / {width})")

    # Show rectangles for integers in [a,b]
    if a.is_integer() and b.is_integer():
        ints = np.arange(int(a), int(b))
        st.markdown("### Probabilities for intervals of length 1:")
        for i in ints:
            st.write(f"P({i} ≤ X < {i+1}) = {fraction_str(prob_per_unit)}")

    x = st.number_input("Enter a value x to query probabilities", value=(a + b) / 2)

    query_type = st.selectbox("Choose a probability query:",
                              ["P(X = x)", "P(X ≥ x)", "P(X > x)", "P(X ≤ x)", "P(X < x)", "P(a < X < b)"])

    if st.button("Calculate Probability"):
        # For continuous uniform, P(X = x) = 0 always
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
                # P(X >= x) = (b - x) / (b - a)
                p = (b - x) / width
            st.markdown(f"**{query_type} = {p:.5f}**")

        elif query_type in ["P(X ≤ x)", "P(X < x)"]:
            if x < a:
                p = 0.0
            elif x > b:
                p = 1.0
            else:
                # P(X <= x) = (x - a) / (b - a)
                p = (x - a) / width
            st.markdown(f"**{query_type} = {p:.5f}**")

        elif query_type == "P(a < X < b)":
            st.markdown(f"Since the uniform distribution is defined over [{a}, {b}],")
            st.markdown(f"**P({a} < X < {b}) = 1**")

    # Plot PDF and shade probability area for P(X > x) or P(X < x)
    import matplotlib.pyplot as plt

    xx = np.linspace(a - width*0.2, b + width*0.2, 500)
    pdf = uniform.pdf(xx, loc=a, scale=width)

    fig, ax = plt.subplots()
    ax.plot(xx, pdf, label="PDF")

    # Shade area based on query
    if query_type in ["P(X ≥ x)", "P(X > x)"] and a <= x <= b:
        ax.fill_between(xx, 0, pdf, where=(xx >= x), color="skyblue", alpha=0.5, label=f"P(X ≥ {x})")
    elif query_type in ["P(X ≤ x)", "P(X < x)"] and a <= x <= b:
        ax.fill_between(xx, 0, pdf, where=(xx <= x), color="lightgreen", alpha=0.5, label=f"P(X ≤ {x})")
    else:
        # No shading if out of bounds or single point
        pass

    ax.set_title(f"Uniform Distribution PDF on [{a}, {b}]")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
