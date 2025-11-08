# ==========================================================
# sample_proportion_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
import math
import pandas as pd
from scipy.stats import norm, t, chi2
import matplotlib.pyplot as plt

# ==========================================================
# Helper Functions
# ==========================================================
def parse_expression(expr):
    expr = expr.strip().lower()
    try:
        if "sqrt" in expr:
            return math.sqrt(float(expr[5:-1]))
        elif "/" in expr:
            a, b = expr.split("/")
            return float(a) / float(b)
        else:
            return float(expr)
    except Exception:
        st.error("⚠️ Invalid input format. Use a number, a/b, or sqrt(x).")
        return None


def load_data_upload():
    """Uploads and extracts numeric column data"""
    uploaded_file = st.file_uploader("📂 Upload CSV or Excel file (single column of numeric data)", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    return df[col].dropna().to_numpy()
            st.error("No numeric column found in file.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    return None


# ==========================================================
# Confidence Interval for Mean (σ unknown, given s or data)
# ==========================================================
def confidence_interval_mean(decimal):
    st.markdown("### 📏 **Confidence Interval for the Mean (σ unknown)**")
    st.info("Uses Student's *t*-distribution when population σ is unknown. Formula:")
    st.latex(r"\bar{X} \pm t_{\alpha/2,\,n-1} \left(\frac{s}{\sqrt{n}}\right)")

    input_mode = st.radio("Select input method:", ["Enter summary statistics", "Upload raw data"], horizontal=True)

    if input_mode == "Enter summary statistics":
        xbar = st.number_input("Sample mean (x̄):", value=50.0)
        s = st.number_input("Sample SD (s):", value=10.0, min_value=0.0001)
        n = st.number_input("Sample size (n):", min_value=2, value=30)
    else:
        data = load_data_upload()
        if data is not None:
            xbar = np.mean(data)
            s = np.std(data, ddof=1)
            n = len(data)
            st.success(f"✅ Data loaded: n={n}, x̄={xbar:.{decimal}f}, s={s:.{decimal}f}")
        else:
            st.stop()

    conf_level = st.slider("Select confidence level:", 0.80, 0.99, 0.95, 0.01)
    alpha = 1 - conf_level
    df = n - 1
    t_crit = t.ppf(1 - alpha/2, df)
    E = t_crit * s / math.sqrt(n)
    lower = xbar - E
    upper = xbar + E

    st.latex(rf"""
    \text{{🧮 Step-by-step}} \\[6pt]
    t_{{\alpha/2,\,{df}}} = {t_crit:.4f} \\[4pt]
    E = t_{{\alpha/2}} \cdot \frac{{s}}{{\sqrt{{n}}}} = {t_crit:.4f} \cdot \frac{{{s:.4f}}}{{\sqrt{{{n}}}}} = {E:.4f} \\[6pt]
    \boxed{{{conf_level*100:.0f}\% \text{{ CI: }} ({lower:.4f},\, {upper:.4f})}}
    """)


# ==========================================================
# Confidence Interval for Variance / Standard Deviation (Chi-Square)
# ==========================================================
def confidence_interval_chi2(decimal):
    st.markdown("### 📊 **Confidence Interval for Variance / Standard Deviation (χ²)**")
    st.info("Uses the Chi-Square distribution:")
    st.latex(r"""
    \text{Variance CI: } \left(\frac{(n-1)s^2}{\chi^2_R}, \frac{(n-1)s^2}{\chi^2_L}\right)
    \quad \text{and} \quad
    \text{SD CI: } \left(\sqrt{\frac{(n-1)s^2}{\chi^2_R}}, \sqrt{\frac{(n-1)s^2}{\chi^2_L}}\right)
    """)

    input_mode = st.radio("Select input method:", ["Enter summary statistics", "Upload raw data"], horizontal=True)

    if input_mode == "Enter summary statistics":
        s = st.number_input("Sample SD (s):", value=10.0, min_value=0.0001)
        n = st.number_input("Sample size (n):", min_value=2, value=30)
    else:
        data = load_data_upload()
        if data is not None:
            s = np.std(data, ddof=1)
            n = len(data)
            st.success(f"✅ Data loaded: n={n}, s={s:.{decimal}f}")
        else:
            st.stop()

    conf_level = st.slider("Select confidence level:", 0.80, 0.99, 0.95, 0.01)
    alpha = 1 - conf_level
    df = n - 1
    chi2_L = chi2.ppf(alpha/2, df)
    chi2_R = chi2.ppf(1 - alpha/2, df)

    var_lower = (df * s**2) / chi2_R
    var_upper = (df * s**2) / chi2_L
    sd_lower = math.sqrt(var_lower)
    sd_upper = math.sqrt(var_upper)

    st.latex(rf"""
    \text{{🧮 Step-by-step}} \\[6pt]
    \chi^2_L = {chi2_L:.4f}, \quad \chi^2_R = {chi2_R:.4f}, \quad df = {df} \\[6pt]
    \text{{Variance CI}} = \left(\frac{{({df})({s:.4f})^2}}{{{chi2_R:.4f}}},\; 
    \frac{{({df})({s:.4f})^2}}{{{chi2_L:.4f}}}\right)
    = ({var_lower:.4f},\, {var_upper:.4f}) \\[6pt]
    \text{{SD CI}} = \left(\sqrt{{{var_lower:.4f}}},\, \sqrt{{{var_upper:.4f}}}\right)
    = ({sd_lower:.4f},\, {sd_upper:.4f}) \\[6pt]
    \boxed{{{conf_level*100:.0f}\% \text{{ CI for σ: }} ({sd_lower:.4f},\, {sd_upper:.4f})}}
    """)


# ==========================================================
# Uniform Distribution and Normal/Proportion functions
# (Keep your previous ones here if needed)
# ==========================================================


# ==========================================================
# MAIN APP
# ==========================================================
def run():
    st.header("🧠 MIND: Continuous Probability & Confidence Intervals")

    st.markdown("""
    ---
    ### 📘 Quick Reference:
    - $\\bar{X}$ : sample mean  $s$ : sample SD  $\\sigma$ : population SD  
    - $\\hat{p}$ : sample proportion  $E$ : margin of error  $n$ : sample size  
    - $\\chi^2$ : chi-square critical values for variance/SD intervals  
    ---
    """)

    tool = st.radio(
        "Select a concept:",
        [
            "Confidence Interval for Mean (σ unknown, given s or data)",
            "Confidence Interval for Variance / Standard Deviation (χ²)"
        ],
        horizontal=True
    )

    decimal = st.number_input("Decimal places for output:", min_value=0, max_value=6, value=4, step=1)

    if tool.startswith("Confidence Interval for Mean"):
        confidence_interval_mean(decimal)
    elif tool.startswith("Confidence Interval for Variance"):
        confidence_interval_chi2(decimal)


# ==========================================================
# Run App
# ==========================================================
if __name__ == "__main__":
    run()
