# inferences_one_sample_tool.py
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def run():
    st.header("📊 Inferences on One Sample")

    test_options = [
        "Proportion Test (Large Sample)",
        "Proportion Test (Small Sample - Binomial)",
        "t-test for Population Mean (Summary Stats)",
        "t-test for Population Mean (Raw Data)",
        "Chi-Squared Test for Standard Deviation (Summary Stats)",
        "Chi-Squared Test for Standard Deviation (Raw Data)"
    ]
    test_choice = st.selectbox("Choose test type:", test_options)
    decimal = st.number_input("Decimal places for output", min_value=0, max_value=10, value=4)

    # ------------------------
    # 1. Large sample proportion test
    # ------------------------
    if test_choice == test_options[0]:
        p0 = st.number_input("Hypothesized proportion (p0)", value=0.5, min_value=0.0, max_value=1.0, step=0.001, format="%.4f")
        n = st.number_input("Sample size", min_value=1, step=1)
        x = st.number_input("Number of successes", min_value=0, max_value=n, step=1)
        alpha = st.number_input("Significance level", value=0.05)

        if st.button("Run Test"):
            phat = x / n
            se = np.sqrt(p0 * (1 - p0) / n)
            z_stat = (phat - p0) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            st.latex(rf"z = \frac{{\hat{{p}} - p_0}}{{\sqrt{{p_0(1-p_0)/n}}}} = {z_stat:.{decimal}f}")
            st.write(f"Sample Proportion = {phat:.{decimal}f}")
            st.write(f"P-value = {p_value:.{decimal}f}")
            st.write("Reject H₀" if p_value < alpha else "Fail to reject H₀")

    # ------------------------
    # 2. Small sample proportion test (binomial)
    # ------------------------
    elif test_choice == test_options[1]:
        p0 = st.number_input("Hypothesized proportion (p0)", value=0.5, min_value=0.0, max_value=1.0, step=0.001, format="%.4f")
        n = st.number_input("Sample size", min_value=1, step=1)
        x = st.number_input("Number of successes", min_value=0, max_value=n, step=1)
        alpha = st.number_input("Significance level", value=0.05)

        if st.button("Run Test"):
            result = stats.binom_test(x, n, p=p0, alternative='two-sided')
            phat = x / n
            st.write(f"Sample Proportion = {phat:.{decimal}f}")
            st.write(f"P-value = {result:.{decimal}f}")
            st.write("Reject H₀" if result < alpha else "Fail to reject H₀")

    # ------------------------
    # 3. t-test (summary stats)
    # ------------------------
    elif test_choice == test_options[2]:
        mu0 = st.number_input("Hypothesized mean (μ0)")
        mean = st.number_input("Sample mean")
        s = st.number_input("Sample standard deviation", min_value=0.0)
        n = st.number_input("Sample size", min_value=1, step=1)
        alpha = st.number_input("Significance level", value=0.05)

        if st.button("Run Test"):
            se = s / np.sqrt(n)
            t_stat = (mean - mu0) / se
            df = n - 1
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))

            st.latex(rf"t = \frac{{\bar{{x}} - \mu_0}}{{s / \sqrt{{n}}}} = {t_stat:.{decimal}f}")
            st.write(f"df = {df}")
            st.write(f"P-value = {p_value:.{decimal}f}")
            st.write("Reject H₀" if p_value < alpha else "Fail to reject H₀")

    # ------------------------
    # 4. t-test (raw data)
    # ------------------------
    elif test_choice == test_options[3]:
        mu0 = st.number_input("Hypothesized mean (μ0)")
        file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        if file:
            if file.name.endswith(".csv"):
                data = pd.read_csv(file).iloc[:, 0]
            else:
                data = pd.read_excel(file).iloc[:, 0]
        else:
            raw_data = st.text_area("Enter values separated by commas")
            data = pd.Series([float(x) for x in raw_data.split(",") if x.strip() != ""]) if raw_data else None

        alpha = st.number_input("Significance level", value=0.05)

        if st.button("Run Test") and data is not None and len(data) > 0:
            t_stat, p_value = stats.ttest_1samp(data, mu0)
            df = len(data) - 1

            st.latex(rf"t = {t_stat:.{decimal}f},\ df = {df}")
            st.write(f"P-value = {p_value:.{decimal}f}")
            st.write("Reject H₀" if p_value < alpha else "Fail to reject H₀")

            fig, ax = plt.subplots()
            ax.hist(data, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
            ax.axvline(mu0, color="red", linestyle="--", label="Hypothesized Mean")
            ax.legend()
            st.pyplot(fig)

    # ------------------------
    # 5. Chi-squared test for std dev (summary stats)
    # ------------------------
    elif test_choice == test_options[4]:
        sigma0 = st.number_input("Hypothesized standard deviation (σ0)", min_value=0.0)
        s = st.number_input("Sample standard deviation", min_value=0.0)
        n = st.number_input("Sample size", min_value=1, step=1)
        alpha = st.number_input("Significance level", value=0.05)

        if st.button("Run Test"):
            df = n - 1
            chi2_stat = df * (s**2) / (sigma0**2)
            p_lower = stats.chi2.cdf(chi2_stat, df=df)
            p_upper = 1 - p_lower
            p_value = 2 * min(p_lower, p_upper)

            st.latex(rf"\chi^2 = \frac{{(n-1)s^2}}{{\sigma_0^2}} = {chi2_stat:.{decimal}f}")
            st.write(f"P-value = {p_value:.{decimal}f}")
            st.write("Reject H₀" if p_value < alpha else "Fail to reject H₀")

    # ------------------------
    # 6. Chi-squared test for std dev (raw data)
    # ------------------------
    elif test_choice == test_options[5]:
        sigma0 = st.number_input("Hypothesized standard deviation (σ0)", min_value=0.0)
        file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        if file:
            if file.name.endswith(".csv"):
                data = pd.read_csv(file).iloc[:, 0]
            else:
                data = pd.read_excel(file).iloc[:, 0]
        else:
            raw_data = st.text_area("Enter values separated by commas")
            data = pd.Series([float(x) for x in raw_data.split(",") if x.strip() != ""]) if raw_data else None

        alpha = st.number_input("Significance level", value=0.05)

        if st.button("Run Test") and data is not None and len(data) > 0:
            n = len(data)
            s = np.std(data, ddof=1)
            df = n - 1
            chi2_stat = df * (s**2) / (sigma0**2)
            p_lower = stats.chi2.cdf(chi2_stat, df=df)
            p_upper = 1 - p_lower
            p_value = 2 * min(p_lower, p_upper)

            st.latex(rf"\chi^2 = \frac{{(n-1)s^2}}{{\sigma_0^2}} = {chi2_stat:.{decimal}f}")
            st.write(f"P-value = {p_value:.{decimal}f}")
            st.write("Reject H₀" if p_value < alpha else "Fail to reject H₀")

            fig, ax = plt.subplots()
            ax.hist(data, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
            ax.axvline(sigma0, color="red", linestyle="--", label="Hypothesized σ")
            ax.legend()
            st.pyplot(fig)
