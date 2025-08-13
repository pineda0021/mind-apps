import streamlit as st
import numpy as np
import plotly.graph_objects as go
from fractions import Fraction

def run():
    st.header("🎲 Discrete Probability Distribution Tool")

    st.write("Enter values of discrete random variable X and their probabilities P(X).")

    x_input = st.text_input("Values of X (comma-separated):")
    p_input = st.text_input("Probabilities P(X) (comma-separated, decimals or fractions):")

    if not x_input or not p_input:
        st.info("Please enter both X values and probabilities.")
        return

    try:
        X = np.array([float(x.strip()) for x in x_input.split(",")])
        P = np.array([float(Fraction(p.strip())) for p in p_input.split(",")])
    except Exception as e:
        st.error(f"Error parsing input: {e}")
        return

    if len(X) != len(P):
        st.error("X and P must be the same length.")
        return

    if not np.isclose(np.sum(P), 1):
        st.error(f"Probabilities must sum to 1. Current sum: {np.sum(P):.4f}")
        return

    # Summary
    mean = np.sum(X * P)
    variance = np.sum(P * (X - mean) ** 2)
    std_dev = np.sqrt(variance)
    cdf = np.cumsum(P)

    st.subheader("Discrete Distribution Summary")
    summary_df = st.dataframe({
        "X": X,
        "P(X)": P.round(4),
        "Cumulative P(X ≤ x)": cdf.round(4)
    })

    st.markdown(f"**Mean:** {mean:.4f}  \n**Variance:** {variance:.4f}  \n**Standard Deviation:** {std_dev:.4f}")

    # Plot with Plotly
    fig = go.Figure(go.Bar(x=X, y=P, marker_color='green'))
    fig.update_layout(title="Probability Distribution", xaxis_title="X", yaxis_title="P(X)", template="simple_white")
    st.plotly_chart(fig)
