# ==========================================================
# confidence_intervals_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import math
from scipy.stats import t, chi2, norm

# ==========================================================
# Helper Functions
# ==========================================================
def load_data_upload(label="📂 Upload CSV or Excel file (single column of numeric data)"):
    """Uploads and extracts numeric column data"""
    uploaded_file = st.file_uploader(label, type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    return df[col].dropna().to_numpy()
            st.error("⚠️ No numeric column found in file.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    return None


# ==========================================================
# Confidence Interval for Mean (σ unknown, uses t)
# ==========================================================
def confidence_interval_mean(decimal):
    st.markdown("### 📏 **Confidence Interval for the Mean (σ unknown)**")
    st.info("Uses Student's *t*-distribution when population σ is unknown.")
    st.latex(r"\bar{X} \pm t_{\alpha/2,\,n-1}\left(\frac{s}{\sqrt{n}}\right)")

    input_mode = st.radio(
        "Select input method:",
        ["Enter summary statistics", "Upload raw data"],
        horizontal=True
    )

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
    t_crit = t.ppf(1 - alpha / 2, df)
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
# Confidence Interval for Variance / SD (uses χ²)
# ==========================================================
def confidence_interval_chi2(decimal):
    st.markdown("### 📊 **Confidence Interval for Variance / Standard Deviation (χ²)**")
    st.info("Uses the Chi-Square distribution:")
    st.latex(r"""
    \text{Variance CI: } 
    \left(\frac{(n-1)s^2}{\chi^2_R}, \frac{(n-1)s^2}{\chi^2_L}\right),
    \quad
    \text{SD CI: } 
    \left(\sqrt{\frac{(n-1)s^2}{\chi^2_R}}, \sqrt{\frac{(n-1)s^2}{\chi^2_L}}\right)
    """)

    input_mode = st.radio(
        "Select input method:",
        ["Enter summary statistics", "Upload raw data"],
        horizontal=True
    )

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
    chi2_L = chi2.ppf(alpha / 2, df)
    chi2_R = chi2.ppf(1 - alpha / 2, df)

    var_lower = (df * s**2) / chi2_R
    var_upper = (df * s**2) / chi2_L
    sd_lower = math.sqrt(var_lower)
    sd_upper = math.sqrt(var_upper)

    st.latex(rf"""
    \text{{🧮 Step-by-step}} \\[6pt]
    \chi^2_L = {chi2_L:.4f}, \quad \chi^2_R = {chi2_R:.4f}, \quad df = {df} \\[6pt]
    \text{{Variance CI}} = ({var_lower:.4f},\, {var_upper:.4f}) \\[6pt]
    \text{{SD CI}} = ({sd_lower:.4f},\, {sd_upper:.4f}) \\[6pt]
    \boxed{{{conf_level*100:.0f}\% \text{{ CI for σ: }} ({sd_lower:.4f},\, {sd_upper:.4f})}}
    """)


# ==========================================================
# Confidence Interval for Proportion (uses z)
# ==========================================================
def confidence_interval_proportion(decimal):
    st.markdown("### 📈 **Confidence Interval for a Population Proportion (z)**")
    st.info("Formula:")
    st.latex(r"\hat{p} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}")

    x = st.number_input("Number of successes (x):", min_value=0, value=50)
    n = st.number_input("Sample size (n):", min_value=1, value=100)
    p_hat = x / n
    conf_level = st.slider("Select confidence level:", 0.80, 0.99, 0.95, 0.01)
    alpha = 1 - conf_level
    z_crit = norm.ppf(1 - alpha / 2)
    E = z_crit * math.sqrt(p_hat * (1 - p_hat) / n)
    lower = p_hat - E
    upper = p_hat + E

    st.latex(rf"""
    \text{{🧮 Step-by-step}} \\[6pt]
    \hat{{p}} = \frac{{x}}{{n}} = \frac{{{x}}}{{{n}}} = {p_hat:.4f} \\[6pt]
    E = z_{{\alpha/2}} \sqrt{{\frac{{\hat{{p}}(1-\hat{{p}})}}{{n}}}} 
      = {z_crit:.4f} \sqrt{{\frac{{{p_hat:.4f}(1-{p_hat:.4f})}}{{{n}}}}}
      = {E:.4f} \\[6pt]
    \boxed{{{conf_level*100:.0f}\% \text{{ CI: }} ({lower:.4f},\, {upper:.4f})}}
    """)


# ==========================================================
# Confidence Interval for Two Means (independent samples)
# ==========================================================
def confidence_interval_two_means(decimal):
    st.markdown("### ⚖️ **Confidence Interval for Difference of Two Means (Independent Samples)**")
    st.info("Uses Welch’s *t*-interval (unequal variances assumed):")
    st.latex(r"(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2,\,df}\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}")

    input_mode = st.radio(
        "Select input method:",
        ["Enter summary statistics", "Upload two raw datasets"],
        horizontal=True
    )

    if input_mode == "Enter summary statistics":
        x1 = st.number_input("Sample mean 1 (x̄₁):", value=50.0)
        s1 = st.number_input("Sample SD 1 (s₁):", value=10.0, min_value=0.0001)
        n1 = st.number_input("Sample size 1 (n₁):", min_value=2, value=30)
        x2 = st.number_input("Sample mean 2 (x̄₂):", value=45.0)
        s2 = st.number_input("Sample SD 2 (s₂):", value=9.0, min_value=0.0001)
        n2 = st.number_input("Sample size 2 (n₂):", min_value=2, value=25)
    else:
        st.write("Upload two datasets below:")
        col1, col2 = st.columns(2)
        with col1:
            data1 = load_data_upload("📂 Upload Dataset 1")
        with col2:
            data2 = load_data_upload("📂 Upload Dataset 2")
        if data1 is not None and data2 is not None:
            x1, s1, n1 = np.mean(data1), np.std(data1, ddof=1), len(data1)
            x2, s2, n2 = np.mean(data2), np.std(data2, ddof=1), len(data2)
            st.success(f"✅ Data 1: n₁={n1}, x̄₁={x1:.{decimal}f}, s₁={s1:.{decimal}f}")
            st.success(f"✅ Data 2: n₂={n2}, x̄₂={x2:.{decimal}f}, s₂={s2:.{decimal}f}")
        else:
            st.stop()

    conf_level = st.slider("Select confidence level:", 0.80, 0.99, 0.95, 0.01)
    alpha = 1 - conf_level

    # Welch–Satterthwaite df
    se1, se2 = s1**2 / n1, s2**2 / n2
    se = math.sqrt(se1 + se2)
    df = (se1 + se2)**2 / ((se1**2) / (n1 - 1) + (se2**2) / (n2 - 1))
    t_crit = t.ppf(1 - alpha / 2, df)
    diff = x1 - x2
    E = t_crit * se
    lower = diff - E
    upper = diff + E

    st.latex(rf"""
    \text{{🧮 Step-by-step}} \\[6pt]
    df = {df:.4f}, \quad t_{{\alpha/2,df}} = {t_crit:.4f} \\[4pt]
    E = t_{{\alpha/2}} \sqrt{{\frac{{s_1^2}}{{n_1}} + \frac{{s_2^2}}{{n_2}}}} 
      = {t_crit:.4f} \sqrt{{\frac{{{s1:.4f}^2}}{{{n1}}} + \frac{{{s2:.4f}^2}}{{{n2}}}}}
      = {E:.4f} \\[6pt]
    \boxed{{{conf_level*100:.0f}\% \text{{ CI for }} (\mu_1 - \mu_2): ({lower:.4f},\, {upper:.4f})}}
    """)


# ==========================================================
# MAIN APP
# ==========================================================
def run():
    st.header("🧠 MIND: Confidence Interval Tools")

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
            "Confidence Interval for Variance / Standard Deviation (χ²)",
            "Confidence Interval for Proportion (z)",
            "Confidence Interval for Difference of Two Means (t)"
        ],
        horizontal=False
    )

    decimal = st.number_input("Decimal places for output:", min_value=0, max_value=6, value=4, step=1)

    if tool.startswith("Confidence Interval for Mean"):
        confidence_interval_mean(decimal)
    elif tool.startswith("Confidence Interval for Variance"):
        confidence_interval_chi2(decimal)
    elif tool.startswith("Confidence Interval for Proportion"):
        confidence_interval_proportion(decimal)
    elif tool.startswith("Confidence Interval for Difference"):
        confidence_interval_two_means(decimal)


# ==========================================================
# Run directly
# ==========================================================
if __name__ == "__main__":
    run()
