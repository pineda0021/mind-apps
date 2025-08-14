import streamlit as st
import math
import numpy as np
import pandas as pd
from scipy.stats import norm, t, chi2, binom

def load_uploaded_data():
    uploaded_file = st.file_uploader(
        "📂 Upload CSV or Excel file with a single column of numeric data",
        type=["csv", "xlsx"]
    )
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            # Take the first numeric column
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    return df[col].dropna().to_numpy()
            st.error("No numeric column found in file.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    return None

def run_hypothesis_tool():
    st.header("🔎 Inferences on One Sample")

    test_choice = st.selectbox(
        "Choose a hypothesis test:",
        [
            "Proportion test (large sample)",
            "Proportion test (small sample, binomial)",
            "t-test for population mean (summary stats)",
            "t-test for population mean (raw data)",
            "Chi-squared test for std dev (summary stats)",
            "Chi-squared test for std dev (raw data)"
        ]
    )

    alpha = st.number_input("Significance level α", value=0.05, min_value=0.001, max_value=0.5, step=0.01)

    tails = st.selectbox("Tails", ["two", "left", "right"])

    # ------------------- PROPORTION TESTS -------------------
    if test_choice in ["Proportion test (large sample)", "Proportion test (small sample, binomial)"]:
        x = st.number_input("Number of successes", min_value=0, step=1)
        n = st.number_input("Sample size", min_value=1, step=1)
        p0 = st.number_input("Null proportion (p0)", min_value=0.0, max_value=1.0, format="%.10f")

        if st.button("👨‍💻 Calculate"):
            p_hat = x / n
            report = f"""
=====================
{test_choice}
=====================
Sample successes = {x}
Sample size = {n}
Sample proportion = {p_hat:.4f}
Null proportion p0 = {p0:.4f}
"""
            if test_choice == "Proportion test (large sample)":
                se = math.sqrt(p0*(1-p0)/n)
                z_stat = (p_hat - p0)/se
                if tails == "left":
                    z_crit = -abs(norm.ppf(alpha))
                    p_val = norm.cdf(z_stat)
                    reject = z_stat < z_crit
                    crit_str = f"{z_crit:.4f}"
                elif tails == "right":
                    z_crit = abs(norm.ppf(1-alpha))
                    p_val = 1 - norm.cdf(z_stat)
                    reject = z_stat > z_crit
                    crit_str = f"{z_crit:.4f}"
                else:
                    z_crit_left = -abs(norm.ppf(alpha/2))
                    z_crit_right = abs(norm.ppf(alpha/2))
                    p_val = 2*(1 - norm.cdf(abs(z_stat)))
                    reject = abs(z_stat) > z_crit_right
                    crit_str = f"{z_crit_left:.4f}, {z_crit_right:.4f}"
                report += f"Z = {z_stat:.4f}\nCritical Value(s) = {crit_str}\nP-value = {p_val:.4f}\nDecision = {'Reject' if reject else 'Fail to reject'}\n"

            else:  # small sample binomial
                if tails == "left":
                    p_val = binom.cdf(x, n, p0)
                elif tails == "right":
                    p_val = 1 - binom.cdf(x-1, n, p0)
                else:
                    p_val = 2 * min(binom.cdf(x, n, p0), 1 - binom.cdf(x-1, n, p0))
                reject = p_val < alpha
                report += f"P-value = {p_val:.4f}\nDecision = {'Reject' if reject else 'Fail to reject'}\n"

            st.text(report)

    # ------------------- T-TESTS -------------------
    elif test_choice in ["t-test for population mean (summary stats)", "t-test for population mean (raw data)"]:
        if test_choice == "t-test for population mean (summary stats)":
            mean = st.number_input("Sample mean", format="%.10f")
            sd = st.number_input("Sample standard deviation", format="%.10f")
            n = st.number_input("Sample size", min_value=2, step=1)
        else:
            st.write("Option 1: Upload CSV or Excel")
            uploaded_data = load_uploaded_data()
            st.write("Option 2: Enter comma-separated values")
            raw_input = st.text_area("Data", placeholder="1.2, 2.3, 3.1")
        mu0 = st.number_input("Null hypothesis mean", format="%.10f")

        if st.button("👨‍💻 Calculate"):
            if test_choice == "t-test for population mean (raw data)":
                if uploaded_data is not None:
                    data = uploaded_data
                elif raw_input:
                    try:
                        data = np.array([float(i.strip()) for i in raw_input.split(",")])
                    except:
                        st.error("Invalid data")
                        return
                else:
                    st.error("Provide data")
                    return
                mean = np.mean(data)
                sd = np.std(data, ddof=1)
                n = len(data)

            se = sd / math.sqrt(n)
            t_stat = (mean - mu0)/se
            df = n - 1

            if tails == "left":
                t_crit = -abs(t.ppf(alpha, df))
                p_val = t.cdf(t_stat, df)
                reject = t_stat < t_crit
                crit_str = f"{t_crit:.4f}"
            elif tails == "right":
                t_crit = abs(t.ppf(1-alpha, df))
                p_val = 1 - t.cdf(t_stat, df)
                reject = t_stat > t_crit
                crit_str = f"{t_crit:.4f}"
            else:
                t_crit_left = -abs(t.ppf(alpha/2, df))
                t_crit_right = abs(t.ppf(alpha/2, df))
                p_val = 2*(1 - t.cdf(abs(t_stat), df))
                reject = abs(t_stat) > t_crit_right
                crit_str = f"{t_crit_left:.4f}, {t_crit_right:.4f}"

            report = f"""
=====================
{test_choice}
=====================
Sample mean = {mean:.4f}
Sample SD = {sd:.4f}
Sample size = {n}
Null hypothesis mean = {mu0:.4f}
t = {t_stat:.4f}
Critical Value(s) = {crit_str}
P-value = {p_val:.4f}
Decision = {'Reject' if reject else 'Fail to reject'}
"""
            st.text(report)

    # ------------------- CHI-SQUARED TESTS -------------------
    elif test_choice in ["Chi-squared test for std dev (summary stats)", "Chi-squared test for std dev (raw data)"]:
        if test_choice == "Chi-squared test for std dev (summary stats)":
            sd = st.number_input("Sample standard deviation", format="%.10f")
            n = st.number_input("Sample size", min_value=2, step=1)
        else:
            st.write("Option 1: Upload CSV or Excel")
            uploaded_data = load_uploaded_data()
            st.write("Option 2: Enter comma-separated values")
            raw_input = st.text_area("Data", placeholder="1.2, 2.3, 3.1")

        sigma0 = st.number_input("Population standard deviation (null hypothesis)", format="%.10f")

        if st.button("👨‍💻 Calculate"):
            if test_choice == "Chi-squared test for std dev (raw data)":
                if uploaded_data is not None:
                    data = uploaded_data
                elif raw_input:
                    try:
                        data = np.array([float(i.strip()) for i in raw_input.split(",")])
                    except:
                        st.error("Invalid data")
                        return
                else:
                    st.error("Provide data")
                    return
                sd = np.std(data, ddof=1)
                n = len(data)

            df = n - 1
            chi2_stat = (df * sd**2) / sigma0**2

            if tails == "left":
                chi2_crit = chi2.ppf(alpha, df)
                p_val = chi2.cdf(chi2_stat, df)
                reject = chi2_stat < chi2_crit
                crit_str = f"{chi2_crit:.4f}"
            elif tails == "right":
                chi2_crit = chi2.ppf(1-alpha, df)
                p_val = 1 - chi2.cdf(chi2_stat, df)
                reject = chi2_stat > chi2_crit
                crit_str = f"{chi2_crit:.4f}"
            else:
                chi2_crit_left = chi2.ppf(alpha/2, df)
                chi2_crit_right = chi2.ppf(1-alpha/2, df)
                p_val = 2 * min(chi2.cdf(chi2_stat, df), 1 - chi2.cdf(chi2_stat, df))
                reject = chi2_stat < chi2_crit_left or chi2_stat > chi2_crit_right
                crit_str = f"{chi2_crit_left:.4f}, {chi2_crit_right:.4f}"

            report = f"""
=====================
{test_choice}
=====================
Sample SD = {sd:.4f}
Sample size = {n}
Population SD (null) = {sigma0:.4f}
Chi-squared = {chi2_stat:.4f}
Critical Value(s) = {crit_str}
P-value = {p_val:.4f}
Decision = {'Reject' if reject else 'Fail to reject'}
"""
            st.text(report)

if __name__ == "__main__":
    run_hypothesis_tool()

