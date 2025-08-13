# inference_one_sample_tool.py

import streamlit as st
import numpy as np
import pandas as pd
import math
from scipy.stats import norm, t, chi2, binom
import matplotlib.pyplot as plt


def run():
    st.header("📊 Inferences on One Sample")
    test_type = st.sidebar.selectbox(
        "Select a Test",
        [
            "Proportion Test (Large Sample)",
            "Proportion Test (Small Sample - Binomial)",
            "t-Test for Population Mean (Summary Stats)",
            "t-Test for Population Mean (Raw Data)",
            "Chi-Squared Test for Standard Deviation (Summary Stats)",
            "Chi-Squared Test for Standard Deviation (Raw Data)",
        ],
    )

    alpha = st.sidebar.number_input("Significance Level (α)", 0.001, 0.20, 0.05, step=0.001)
    tails = st.sidebar.selectbox("Tails", ["two", "left", "right"])
    decimal_places = st.sidebar.number_input("Decimal Places", 1, 6, 4)

    st.markdown("---")

    if test_type == "Proportion Test (Large Sample)":
        x = st.number_input("Number of successes", 0)
        n = st.number_input("Sample size", 1)
        p_null = st.number_input("Null proportion (p₀)", 0.0, 1.0, 0.5)
        if st.button("Run Test"):
            proportion_large_test(x, n, p_null, alpha, tails, decimal_places)

    elif test_type == "Proportion Test (Small Sample - Binomial)":
        x = st.number_input("Number of successes", 0)
        n = st.number_input("Sample size", 1)
        p_null = st.number_input("Null proportion (p₀)", 0.0, 1.0, 0.5)
        if st.button("Run Test"):
            proportion_small_test(x, n, p_null, alpha, tails, decimal_places)

    elif test_type == "t-Test for Population Mean (Summary Stats)":
        sample_mean = st.number_input("Sample Mean")
        sample_std = st.number_input("Sample Standard Deviation")
        sample_size = st.number_input("Sample Size", 1)
        population_mean = st.number_input("Null Hypothesis Mean (μ₀)")
        if st.button("Run Test"):
            t_test_summary(sample_mean, sample_std, sample_size, population_mean, alpha, tails, decimal_places)

    elif test_type == "t-Test for Population Mean (Raw Data)":
        data = get_data_upload_or_manual()
        population_mean = st.number_input("Null Hypothesis Mean (μ₀)")
        if st.button("Run Test") and len(data) > 0:
            t_test_with_data(data, population_mean, alpha, tails, decimal_places)

    elif test_type == "Chi-Squared Test for Standard Deviation (Summary Stats)":
        sample_std = st.number_input("Sample Standard Deviation")
        sample_size = st.number_input("Sample Size", 1)
        population_std = st.number_input("Population Standard Deviation (σ₀)")
        if st.button("Run Test"):
            chi_sq_summary(sample_std, sample_size, population_std, alpha, tails, decimal_places)

    elif test_type == "Chi-Squared Test for Standard Deviation (Raw Data)":
        data = get_data_upload_or_manual()
        population_std = st.number_input("Population Standard Deviation (σ₀)")
        if st.button("Run Test") and len(data) > 0:
            chi_sq_with_data(data, population_std, alpha, tails, decimal_places)


# ==== Helper Functions ====

def get_data_upload_or_manual():
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        col = st.selectbox("Select column for data", df.columns)
        return df[col].dropna().values
    else:
        raw = st.text_area("Or paste comma-separated values")
        if raw:
            try:
                return np.array(list(map(float, raw.split(","))))
            except:
                st.error("Invalid number format.")
    return []


# ==== Statistical Tests ====

def proportion_large_test(x, n, p_null, alpha, tails, dp):
    p_hat = x / n
    se = math.sqrt(p_null * (1 - p_null) / n)
    z = (p_hat - p_null) / se

    if tails == "two":
        crit = norm.ppf(1 - alpha / 2)
        p_val = 2 * (1 - norm.cdf(abs(z)))
    elif tails == "left":
        crit = norm.ppf(alpha)
        p_val = norm.cdf(z)
    else:
        crit = norm.ppf(1 - alpha)
        p_val = 1 - norm.cdf(z)

    conclusion = "Reject H₀" if (
        (tails == "two" and abs(z) > abs(crit)) or
        (tails == "left" and z < crit) or
        (tails == "right" and z > crit)
    ) else "Do not reject H₀"

    st.latex(fr"\hat{{p}} = \frac{{x}}{{n}} = \frac{{{x}}}{{{n}}} = {p_hat:.{dp}f}")
    st.latex(fr"SE = \sqrt{{\frac{{p_0(1-p_0)}}{{n}}}} = \sqrt{{\frac{{{p_null}({1-p_null})}}{{{n}}}}} = {se:.{dp}f}")
    st.latex(fr"z = \frac{{\hat{{p}} - p_0}}{{SE}} = \frac{{{p_hat:.{dp}f} - {p_null}}}{{{se:.{dp}f}}} = {z:.{dp}f}")

    st.write(f"**Sample Proportion:** {p_hat:.{dp}f}")
    st.write(f"**Critical Value:** {crit:.{dp}f}")
    st.write(f"**Test Statistic (z):** {z:.{dp}f}")
    st.write(f"**P-value:** {p_val:.{dp}f}")
    st.write(f"**Conclusion:** {conclusion}")


def proportion_small_test(x, n, p_null, alpha, tails, dp):
    if tails == "two":
        p_val = 2 * min(binom.cdf(x, n, p_null), 1 - binom.cdf(x - 1, n, p_null))
    elif tails == "left":
        p_val = binom.cdf(x, n, p_null)
    else:
        p_val = 1 - binom.cdf(x - 1, n, p_null)

    conclusion = "Reject H₀" if p_val < alpha else "Do not reject H₀"
    st.write(f"**P-value:** {p_val:.{dp}f}")
    st.write(f"**Conclusion:** {conclusion}")


def t_test_summary(sample_mean, sample_std, sample_size, pop_mean, alpha, tails, dp):
    df = sample_size - 1
    se = sample_std / math.sqrt(sample_size)
    t_stat = (sample_mean - pop_mean) / se

    if tails == "two":
        crit = t.ppf(1 - alpha / 2, df)
        p_val = 2 * (1 - t.cdf(abs(t_stat), df))
    elif tails == "left":
        crit = t.ppf(alpha, df)
        p_val = t.cdf(t_stat, df)
    else:
        crit = t.ppf(1 - alpha, df)
        p_val = 1 - t.cdf(t_stat, df)

    conclusion = "Reject H₀" if (
        (tails == "two" and abs(t_stat) > abs(crit)) or
        (tails == "left" and t_stat < crit) or
        (tails == "right" and t_stat > crit)
    ) else "Do not reject H₀"

    st.latex(fr"t = \frac{{\bar{{x}} - \mu_0}}{{s/\sqrt{{n}}}} = \frac{{{sample_mean} - {pop_mean}}}{{{sample_std}/\sqrt{{{sample_size}}}}} = {t_stat:.{dp}f}")
    st.write(f"**Critical Value:** {crit:.{dp}f}")
    st.write(f"**Test Statistic (t):** {t_stat:.{dp}f}")
    st.write(f"**P-value:** {p_val:.{dp}f}")
    st.write(f"**Conclusion:** {conclusion}")


def t_test_with_data(data, pop_mean, alpha, tails, dp):
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    sample_size = len(data)
    t_test_summary(sample_mean, sample_std, sample_size, pop_mean, alpha, tails, dp)


def chi_sq_summary(sample_std, sample_size, pop_std, alpha, tails, dp):
    df = sample_size - 1
    chi_stat = (df * sample_std**2) / pop_std**2

    if tails == "two":
        crit_low = chi2.ppf(alpha / 2, df)
        crit_high = chi2.ppf(1 - alpha / 2, df)
        p_val = 2 * min(chi2.cdf(chi_stat, df), 1 - chi2.cdf(chi_stat, df))
    elif tails == "left":
        crit_low = chi2.ppf(alpha, df)
        p_val = chi2.cdf(chi_stat, df)
        crit_high = None
    else:
        crit_high = chi2.ppf(1 - alpha, df)
        p_val = 1 - chi2.cdf(chi_stat, df)
        crit_low = None

    conclusion = "Reject H₀" if (
        (tails == "two" and (chi_stat < crit_low or chi_stat > crit_high)) or
        (tails == "left" and chi_stat < crit_low) or
        (tails == "right" and chi_stat > crit_high)
    ) else "Do not reject H₀"

    st.latex(fr"\chi^2 = \frac{{(n-1)s^2}}{{\sigma_0^2}} = \frac{{({sample_size}-1)({sample_std}^2)}}{{{pop_std}^2}} = {chi_stat:.{dp}f}")
    if tails == "two":
        st.write(f"**Critical Values:** {crit_low:.{dp}f}, {crit_high:.{dp}f}")
    elif crit_low:
        st.write(f"**Critical Value:** {crit_low:.{dp}f}")
    elif crit_high:
        st.write(f"**Critical Value:** {crit_high:.{dp}f}")
    st.write(f"**Chi-squared Statistic:** {chi_stat:.{dp}f}")
    st.write(f"**P-value:** {p_val:.{dp}f}")
    st.write(f"**Conclusion:** {conclusion}")


def chi_sq_with_data(data, pop_std, alpha, tails, dp):
    sample_std = np.std(data, ddof=1)
    sample_size = len(data)
    chi_sq_summary(sample_std, sample_size, pop_std, alpha, tails, dp)

