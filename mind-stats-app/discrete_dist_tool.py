import streamlit as st
import numpy as np
import plotly.graph_objects as go
from fractions import Fraction

def parse_input(input_string, allow_fraction=False):
    """Parses comma-separated numbers or fractions into a NumPy array."""
    try:
        if allow_fraction:
            return np.array([float(Fraction(x.strip())) for x in input_string.split(',') if x.strip() != ""])
        else:
            return np.array([float(x.strip()) for x in input_string.split(',') if x.strip() != ""])
    except ValueError:
        return None

def validate_probabilities(px):
    return np.isclose(np.sum(px), 1.0)

def display_discrete_summary(X, P):
    mean = np.round(np.sum(X * P), 4)
    variance = np.round(np.sum(((X - mean) ** 2) * P), 4)
    std_dev = np.round(np.sqrt(variance), 4)
    cumulative = np.round(np.cumsum(P), 4)

    st.subheader("📊 Discrete Probability Distribution Summary")
    table_data = {
        "X": X,
        "P(X)": P,
        "Cumulative P(X ≤ x)": cumulative
    }
    st.table(table_data)

    st.write("### 🧮 Descriptive Statistics")
    st.write(f"**Mean (μ):** {mean}")
    st.write(f"**Variance (σ²):** {variance}")
    st.write(f"**Std Dev (σ):** {std_dev}")

def display_probability_bar(X, P):
    fig = go.Figure(data=[go.Bar(x=X, y=P, marker_color='green')])
    fig.update_layout(
        title="Probability Distribution of X",
        xaxis_title="X",
        yaxis_title="P(X)",
        template="simple_white"
    )
    st.plotly_chart(fig)

def run():
    st.header("🎯 Discrete Probability Distribution Tool")
    x_input = st.text_input("Enter values of random variable X (comma-separated):", "1, 2, 3")
    p_input = st.text_input("Enter corresponding probabilities P(X) (comma-separated, fractions allowed):", "1/6, 1/6, 4/6")

    if x_input and p_input:
        X = parse_input(x_input)
        P = parse_input(p_input, allow_fraction=True)

        if X is None or P is None:
            st.error("❌ Please enter valid numbers or fractions.")
            return

        if len(X) != len(P):
            st.error("❌ X and P(X) must have the same length.")
            return

        if not validate_probabilities(P):
            st.error(f"❌ Probabilities must sum to 1. Your sum was {np.sum(P):.4f}")
            return

        display_discrete_summary(X, P)
        display_probability_bar(X, P)
