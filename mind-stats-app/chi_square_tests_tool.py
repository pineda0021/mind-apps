import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2, chisquare, chi2_contingency

def load_uploaded_data():
    uploaded_file = st.file_uploader(
        "📂 Upload CSV or Excel file",
        type=["csv", "xlsx"]
    )
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error reading file: {e}")
    return None

def run_chi_square_tool():
    st.header("📊 Chi-Squared Hypothesis Tests")

    test_choice = st.selectbox(
        "Choose a Chi-Squared Test:",
        [
            "Goodness-of-Fit Test (expected percentages)",
            "Goodness-of-Fit Test (uniform distribution)",
            "Chi-Square Test of Independence / Homogeneity"
        ]
    )

    alpha = st.number_input("Significance level (e.g., 0.05)", value=0.05, min_value=0.0001, max_value=0.5, step=0.01)
    decimal = st.number_input("Decimal places for output", min_value=1, max_value=10, value=4)

    # ------------------- 1. Goodness-of-Fit (expected percentages) -------------------
    if test_choice == "Goodness-of-Fit Test (expected percentages)":
        st.write("Provide observed counts and expected percentages (must sum to 100).")
        raw_obs = st.text_area("Observed counts (comma-separated)")
        raw_exp_pct = st.text_area("Expected percentages (comma-separated)")

        if st.button("👨‍💻 Calculate"):
            try:
                obs = np.array([float(x.strip()) for x in raw_obs.split(",")])
                exp_pct = np.array([float(x.strip()) for x in raw_exp_pct.split(",")])
                if len(obs) != len(exp_pct):
                    st.error("Observed and expected arrays must be the same length")
                    return
                exp = np.sum(obs) * exp_pct / 100
                chi2_stat = np.sum((obs - exp)**2 / exp)
                df = len(obs) - 1
                p_val = 1 - chi2.cdf(chi2_stat, df)
                reject = p_val < alpha

                st.text(f"""
=====================
Goodness-of-Fit Test (Expected Percentages)
=====================
Observed counts = {obs}
Expected counts = {np.round(exp, decimal)}
Chi-squared statistic = {chi2_stat:.{decimal}f}
Degrees of freedom = {df}
P-value = {p_val:.{decimal}f}
Conclusion: {'Reject' if reject else 'Do not reject'} the null hypothesis
""")
            except:
                st.error("Invalid input")

    # ------------------- 2. Goodness-of-Fit (uniform distribution) -------------------
    elif test_choice == "Goodness-of-Fit Test (uniform distribution)":
        st.write("Enter observed counts for each category.")
        raw_obs = st.text_area("Observed counts (comma-separated)")

        if st.button("👨‍💻 Calculate"):
            try:
                obs = np.array([float(x.strip()) for x in raw_obs.split(",")])
                k = len(obs)
                exp = np.full(k, np.mean(obs))  # uniform expected counts
                chi2_stat = np.sum((obs - exp)**2 / exp)
                df = k - 1
                p_val = 1 - chi2.cdf(chi2_stat, df)
                reject = p_val < alpha

                st.text(f"""
=====================
Goodness-of-Fit Test (Uniform)
=====================
Observed counts = {obs}
Expected counts = {np.round(exp, decimal)}
Chi-squared statistic = {chi2_stat:.{decimal}f}
Degrees of freedom = {df}
P-value = {p_val:.{decimal}f}
Conclusion: {'Reject' if reject else 'Do not reject'} the null hypothesis
""")
            except:
                st.error("Invalid input")

    # ------------------- 3. Chi-Square Test of Independence / Homogeneity -------------------
    elif test_choice == "Chi-Square Test of Independence / Homogeneity":
        st.write("Provide a contingency table as CSV/Excel or manual entry.")
        file = load_uploaded_data()
        raw_data = st.text_area("Or enter data as rows of comma-separated values, one row per line")

        if file is not None:
            data = file.to_numpy()
        elif raw_data:
            try:
                data = np.array([[float(x) for x in row.split(",")] for row in raw_data.split("\n") if row.strip() != ""])
            except:
                st.error("Invalid data input")
                data = None
        else:
            data = None

        if st.button("👨‍💻 Calculate") and data is not None:
            chi2_stat, p_val, df, expected = chi2_contingency(data)
            reject = p_val < alpha

            st.text(f"""
=====================
Chi-Square Test of Independence / Homogeneity
=====================
Observed Table:
{data}
Expected Table:
{np.round(expected, decimal)}
Chi-squared statistic = {chi2_stat:.{decimal}f}
Degrees of freedom = {df}
P-value = {p_val:.{decimal}f}
Conclusion: {'Reject' if reject else 'Do not reject'} the null hypothesis
""")

if __name__ == "__main__":
    run_chi_square_tool()
