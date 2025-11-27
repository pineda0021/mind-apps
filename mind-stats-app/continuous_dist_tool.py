import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm


# ---------------------------------------------------------
#  DARK/LIGHT MODE–SAFE GLOBAL STYLE
# ---------------------------------------------------------
STYLE = """
<style>
    body {
        background-color: transparent !important;
    }
    .stPlotlyChart {
        background-color: transparent !important;
    }
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: inherit !important;
    }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# ---------------------------------------------------------
#  PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Sampling Simulator", layout="wide")

st.title("📊 Sampling Distributions & Probability Explorer")

# ---------------------------------------------------------
#  DROPDOWN MENU
# ---------------------------------------------------------
mode = st.selectbox(
    "Choose a module:",
    [
        "Uniform Distribution",
        "Normal Distribution",
        "Sampling Distribution of the Mean",
        "Sampling Distribution of a Proportion"
    ]
)

# ---------------------------------------------------------
#  HELPER FUNCTION: DARK/LIGHT MODE–SAFE PLOT
# ---------------------------------------------------------
def plot_distribution(x, y, title, xlabel):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        fill="tozeroy",
        line=dict(width=3),
        hoverinfo="x+y"
    ))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title="Density",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white" if st.get_option('theme.primaryColor') else "black"
    )

    return fig


# ---------------------------------------------------------
#  UNIFORM DISTRIBUTION
# ---------------------------------------------------------
if mode == "Uniform Distribution":
    st.header("🎲 Uniform Distribution")

    a = st.number_input("Lower bound a:", value=0.0)
    b = st.number_input("Upper bound b:", value=10.0)

    if b <= a:
        st.error("Upper bound must be greater than lower bound.")
    else:
        x = np.linspace(a, b, 400)
        y = [1 / (b - a)] * len(x)

        st.plotly_chart(plot_distribution(x, y, "Uniform Density", "x"), use_container_width=True)

        st.subheader("Step-by-step solution")

        st.markdown(f"""
        - Support: [{a}, {b}]  
        - PDF: \( f(x) = 1 / (b - a) = 1 / ({b} - {a}) = {1/(b-a):.4f} \)  
        - Mean: \( \\mu = (a + b) / 2 = {(a+b)/2:.4f} \)  
        - Variance: \( \\sigma^2 = (b - a)^2 / 12 = {((b-a)**2)/12:.4f} \)
        """)


# ---------------------------------------------------------
#  NORMAL DISTRIBUTION
# ---------------------------------------------------------
if mode == "Normal Distribution":
    st.header("📈 Normal Distribution")

    mean = st.number_input("Mean μ:", value=0.0)
    sd = st.number_input("Standard deviation σ:", value=1.0)

    x = np.linspace(mean - 4*sd, mean + 4*sd, 400)
    y = norm.pdf(x, mean, sd)

    st.plotly_chart(plot_distribution(x, y, "Normal Distribution", "x"), use_container_width=True)

    st.subheader("Step-by-step solution")

    st.markdown(f"""
    - PDF: \( f(x) = \\frac{{1}}{{\\sqrt{{2\\pi}}\\sigma}} e^{{-(x-\\mu)^2/(2\\sigma^2)}} \)  
    - Mean: \( \\mu = {mean} \)  
    - Standard deviation: \( \\sigma = {sd} \)
    """)


# ---------------------------------------------------------
#  SAMPLING DISTRIBUTION OF THE MEAN
# ---------------------------------------------------------
if mode == "Sampling Distribution of the Mean":
    st.header("📘 Sampling Distribution of the Mean")

    mu = st.number_input("Population mean μ:", value=50.0)
    sigma = st.number_input("Population SD σ:", value=10.0)
    n = st.number_input("Sample size n:", value=30, min_value=1)

    se = sigma / np.sqrt(n)

    x = np.linspace(mu - 4*se, mu + 4*se, 400)
    y = norm.pdf(x, mu, se)

    st.plotly_chart(plot_distribution(x, y, "Sampling Distribution of the Mean", "Sample Mean x̄"), use_container_width=True)

    st.subheader("Step-by-step solution")

    st.markdown(f"""
    - Standard error:  
      \( SE = \\sigma / \\sqrt{{n}} = {sigma} / \\sqrt{{{n}}} = {se:.4f} \)

    - Sampling distribution:  
      \( x̄ \\sim N(\\mu, SE) \)  
      \( x̄ \\sim N({mu}, {se:.4f}) \)
    """)


# ---------------------------------------------------------
#  SAMPLING DISTRIBUTION OF A PROPORTION
# ---------------------------------------------------------
if mode == "Sampling Distribution of a Proportion":
    st.header("📗 Sampling Distribution of a Proportion")

    p = st.number_input("Population proportion p:", value=0.5, min_value=0.0, max_value=1.0)
    n = st.number_input("Sample size n:", value=40, min_value=1)

    se = np.sqrt(p*(1-p)/n)

    x = np.linspace(p - 4*se, p + 4*se, 400)
    y = norm.pdf(x, p, se)

    st.plotly_chart(plot_distribution(x, y, "Sampling Distribution of p̂", "p̂"), use_container_width=True)

    st.subheader("Step-by-step solution")

    st.markdown(f"""
    - Standard error:  
      \( SE = \\sqrt{{ p(1-p)/n }} = \\sqrt{{ {p}({1-p})/{n} }} = {se:.4f} \)

    - Sampling distribution:  
      \( \\hat p \\sim N(p, SE) \)  
      \( \\hat p \\sim N({p}, {se:.4f}) \)
    """)

st.success("Ready! Choose another module from the dropdown.")
