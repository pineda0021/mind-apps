import streamlit as st
import math
import numpy as np
import pandas as pd
from scipy.stats import norm, t, chi2, binom

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

    alpha = st.number_input("Significance level (e.g., 0.05)", value=0.05, min_value=0.0001, max_value=0.5, step=0.01)
    tails = st.selectbox("Tails", ["two", "left", "right"])
    decimal_places = st.number_input("Decimal places for rounding", min_value=1, max_value=10, value=4, step=1)

    if test_choice == "Proportion test (large sample)":
        x = st.number_input("Number of successes", min_value=0, step=1)
        n = st.number_input("Sample size", min_value=1, step=1)
        p_null = st.number_input("Null proportion (p0)", min_value=0.0, max_value=1.0, format="%.10f")

        if st.button("👨‍💻 Calculate Proportion (Large Sample)"):
            p_hat = x / n
            se = math.sqrt(p_null * (1 - p_null) / n)
            z_stat = (p_hat - p_null) / se

            if tails == "left":
                crit_val = -abs(norm.ppf(alpha))
                p_val = norm.cdf(z_stat)
                reject = z_stat < crit_val
                crit_str = f"{crit_val:.{decimal_places}f}"
            elif tails == "right":
                crit_val = abs(norm.ppf(1 - alpha))
                p_val = 1 - norm.cdf(z_stat)
                reject = z_stat > crit_val
                crit_str = f"{crit_val:.{decimal_places}f}"
            else:
                crit_val_left = -abs(norm.ppf(alpha / 2))
                crit_val_right = abs(norm.ppf(alpha / 2))
                p_val = 2 * (1 - norm.cdf(abs(z_stat)))
                reject = abs(z_stat) > crit_val_right
                crit_str = f"{crit_val_left:.{decimal_places}f}, {crit_val_right:.{decimal_places}f}"

            st.latex(r"z = \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}}")
            st.write(f"Sample Proportion: {p_hat:.{decimal_places}f}")
            st.write(f"Critical Value(s): {crit_str}")
            st.write(f"Test statistic: {z_stat:.{decimal_places}f}")
            st.write(f"P-value: {p_val:.{decimal_places}f}")
            st.write(f"Conclusion: {'Reject' if reject else 'Do not reject'} the null hypothesis")

    elif test_choice == "Proportion test (small sample, binomial)":
        x = st.number_input("Number of successes", min_value=0, step=1)
        n = st.number_input("Sample size", min_value=1, step=1)
        p_null = st.number_input("Null proportion (p0)", min_value=0.0, max_value=1.0, format="%.10f")

        if st.button("👨‍💻 Calculate Proportion (Small Sample)"):
            if tails == "left":
                p_val = binom.cdf(x, n, p_null)
                reject = p_val < alpha
            elif tails == "right":
                p_val = 1 - binom.cdf(x - 1, n, p_null)
                reject = p_val < alpha
            else:
                p_val = 2 * min(binom.cdf(x, n, p_null), 1 - binom.cdf(x - 1, n, p_null))
                reject = p_val < alpha
                
            st.write(f"P-value: {p_val:.{decimal_places}f}")
            st.write(f"Conclusion: {'Reject' if reject else 'Do not reject'} the null hypothesis")

    elif test_choice == "t-test for population mean (summary stats)":
        sample_mean = st.number_input("Sample mean", format="%.10f")
        sample_std = st.number_input("Sample standard deviation", format="%.10f")
        sample_size = st.number_input("Sample size", min_value=2, step=1)
        population_mean = st.number_input("Null hypothesis mean", format="%.10f")

        if st.button("👨‍💻 Calculate t-test (Summary Stats)"):
            df = sample_size - 1
            se = sample_std / math.sqrt(sample_size)
            t_stat = (sample_mean - population_mean) / se

            if tails == "left":
                crit_val = -abs(t.ppf(alpha, df))
                p_val = t.cdf(t_stat, df)
                reject = t_stat < crit_val
                crit_str = f"{crit_val:.{decimal_places}f}"
            elif tails == "right":
                crit_val = abs(t.ppf(1 - alpha, df))
                p_val = 1 - t.cdf(t_stat, df)
                reject = t_stat > crit_val
                crit_str = f"{crit_val:.{decimal_places}f}"
            else:
                crit_val_left = -abs(t.ppf(alpha / 2, df))
                crit_val_right = abs(t.ppf(alpha / 2, df))
                p_val = 2 * (1 - t.cdf(abs(t_stat), df))
                reject = abs(t_stat) > crit_val_right
                crit_str = f"{crit_val_left:.{decimal_places}f}, {crit_val_right:.{decimal_places}f}"

            st.latex(r"t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}")
            st.write(f"Critical Value(s): {crit_str}")
            st.write(f"Test statistic: {t_stat:.{decimal_places}f}")
            st.write(f"P-value: {p_val:.{decimal_places}f}")
            st.write(f"Conclusion: {'Reject' if reject else 'Do not reject'} the null hypothesis")

    elif test_choice == "t-test for population mean (raw data)":
        raw = st.text_area("Enter comma-separated values:")
        population_mean = st.number_input("Null hypothesis mean", format="%.10f")

        if st.button("👨‍💻 Calculate t-test (Raw Data)") and raw:
            data = np.array(list(map(float, raw.split(","))))
            sample_mean = np.mean(data)
            sample_std = np.std(data, ddof=1)
            sample_size = len(data)
            df = sample_size - 1
            se = sample_std / math.sqrt(sample_size)
            t_stat = (sample_mean - population_mean) / se

            if tails == "left":
                crit_val = -abs(t.ppf(alpha, df))
                p_val = t.cdf(t_stat, df)
                reject = t_stat < crit_val
                crit_str = f"{crit_val:.{decimal_places}f}"
            elif tails == "right":
                crit_val = abs(t.ppf(1 - alpha, df))
                p_val = 1 - t.cdf(t_stat, df)
                reject = t_stat > crit_val
                crit_str = f"{crit_val:.{decimal_places}f}"
            else:
                crit_val_left = -abs(t.ppf(alpha / 2, df))
                crit_val_right = abs(t.ppf(alpha / 2, df))
                p_val = 2 * (1 - t.cdf(abs(t_stat), df))
                reject = abs(t_stat) > crit_val_right
                crit_str = f"{crit_val_left:.{decimal_places}f}, {crit_val_right:.{decimal_places}f}"

            st.latex(r"t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}")
            st.write(f"Critical Value(s): {crit_str}")
            st.write(f"Test statistic: {t_stat:.{decimal_places}f}")
            st.write(f"P-value: {p_val:.{decimal_places}f}")
            st.write(f"Conclusion: {'Reject' if reject else 'Do not reject'} the null hypothesis")

    # Chi-squared tests
    elif test_choice in ["Chi-squared test for std dev (summary stats)", "Chi-squared test for std dev (raw data)"]:
        if test_choice == "Chi-squared test for std dev (raw data)":
            raw = st.text_area("Enter comma-separated values:")
            if raw:
                data = np.array(list(map(float, raw.split(","))))
                sample_std = np.std(data, ddof=1)
                sample_size = len(data)
            else:
                sample_std = None
                sample_size = None
        else:
            sample_std = st.number_input("Sample standard deviation", format="%.10f")
            sample_size = st.number_input("Sample size", min_value=2, step=1)

        population_std = st.number_input("Population std dev (null hypothesis)", format="%.10f")

        if st.button("👨‍💻 Calculate Chi-squared") and sample_std is not None:
            df = sample_size - 1
            chi2_stat = (df * sample_std**2) / population_std**2

            if tails == "left":
                crit_val = chi2.ppf(alpha, df)
                p_val = 1 - chi2.cdf(chi2_stat, df)
                reject = chi2_stat < crit_val
                crit_str = f"{crit_val:.{decimal_places}f}"
            elif tails == "right":
                crit_val = chi2.ppf(1 - alpha, df)
                p_val = chi2.cdf(chi2_stat, df)
                reject = chi2_stat > crit_val
                crit_str = f"{crit_val:.{decimal_places}f}"
            else:
                crit_val_left = chi2.ppf(alpha / 2, df)
                crit_val_right = chi2.ppf(1 - alpha / 2, df)
                p_val = 2 * min(chi2.cdf(chi2_stat, df), 1 - chi2.cdf(chi2_stat, df))
                reject = chi2_stat < crit_val_left or chi2_stat > crit_val_right
                crit_str = f"{crit_val_left:.{decimal_places}f}, {crit_val_right:.{decimal_places}f}"

            st.latex(r"\chi^2 = \frac{(n-1)s^2}{\sigma_0^2}")
            st.write(f"Critical Value(s): {crit_str}")
            st.write(f"Chi-squared Statistic: {chi2_stat:.{decimal_places}f}")
            st.write(f"P-value: {p_val:.{decimal_places}f}")
            st.write(f"Conclusion: {'Reject' if reject else 'Do not reject'} the null hypothesis")

if __name__ == "__main__":
    run_hypothesis_tool()
