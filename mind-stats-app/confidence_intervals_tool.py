# confidence_intervals_tool.py
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def round_value(value, decimals=4):
    return round(value, decimals)

def run():
    st.header("📏 Confidence Interval Calculator")

    categories = [
        "Confidence Interval for Proportion",
        "Sample Size for Proportion",
        "Confidence Interval for Mean (Known Standard Deviation)",
        "Confidence Interval for Mean (With Data)",
        "Sample Size for Mean",
        "Confidence Interval for Variance (Without Data)",
        "Confidence Interval for Variance (With Data)",
        "Confidence Interval for Standard Deviation (Without Data)",
        "Confidence Interval for Standard Deviation (With Data)"
    ]

    choice = st.selectbox("Choose a category:", categories)
    decimal = st.number_input("Decimal places for output (except Sample Size)", min_value=0, max_value=10, value=4, step=1)

   # 1. Confidence Interval for Proportion
    if choice == categories[0]:
    n = st.number_input("Sample size", min_value=1, step=1)
    x = st.number_input("Number of successes", min_value=0, max_value=n, step=1)
    confidence_level = st.number_input("Confidence level (e.g., 0.95)", min_value=0.0, max_value=1.0, value=0.95)

    if st.button("Calculate"):
        if n <= 0:
            st.error("Sample size must be greater than 0.")
        else:
            p_hat = x / n
            se = np.sqrt((p_hat * (1 - p_hat)) / n)
            z = stats.norm.ppf((1 + confidence_level) / 2)
            moe = z * se
            lower, upper = p_hat - moe, p_hat + moe

            st.latex(rf"\hat{{p}} = {p_hat:.{decimal}f}")
            st.latex(rf"Critical Value (Z-Score) = {z:.{decimal}f}")
            st.latex(rf"CI_{{{confidence_level*100:.1f}\%}} = \left({lower:.{decimal}f}, {upper:.{decimal}f}\right)")

    # 2. Sample Size for Proportion
    elif choice == categories[1]:
        confidence_level = st.number_input("Confidence level", value=0.95)
        p_est = st.number_input("Estimated proportion", value=0.5)
        moe = st.number_input("Margin of error", value=0.05)

        if st.button("Calculate"):
            z = stats.norm.ppf((1 + confidence_level) / 2)
            n_req = (z**2 * p_est * (1 - p_est)) / (moe**2)
            st.write("Required sample size:", int(np.ceil(n_req)))
            st.write(f"Critical Value (Z-Score) = {round_value(z, decimal)}")

    # 3. CI for Mean (Known SD)
    elif choice == categories[2]:
        mean = st.number_input("Sample mean")
        sd = st.number_input("Population standard deviation", min_value=0.0)
        n = st.number_input("Sample size", min_value=1, step=1)
        confidence_level = st.number_input("Confidence level", value=0.95)

        if st.button("Calculate"):
            z = stats.norm.ppf((1 + confidence_level) / 2)
            se = sd / np.sqrt(n)
            moe = z * se
            lower, upper = mean - moe, mean + moe
            st.latex(rf"Critical Value (Z-Score) = {z:.{decimal}f}")
            st.latex(rf"CI = \left({lower:.{decimal}f}, {upper:.{decimal}f}\right)")

    # 4. CI for Mean (With Data)
    elif choice == categories[3]:
        st.write("Upload CSV/Excel or enter data manually.")
        file = st.file_uploader("Upload file", type=["csv", "xlsx"])
        if file:
            if file.name.endswith(".csv"):
                data = pd.read_csv(file).iloc[:, 0]
            else:
                data = pd.read_excel(file).iloc[:, 0]
        else:
            raw_data = st.text_area("Enter values separated by commas")
            data = pd.Series([float(x) for x in raw_data.split(",") if x.strip() != ""]) if raw_data else None

        confidence_level = st.number_input("Confidence level", value=0.95)

        if st.button("Calculate") and data is not None and len(data) > 0:
            n = len(data)
            mean = np.mean(data)
            sd = np.std(data, ddof=1)
            df = n - 1
            t_crit = stats.t.ppf((1 + confidence_level) / 2, df=df)
            se = sd / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe

            st.latex(rf"\bar{{x}} = {mean:.{decimal}f},\ s = {sd:.{decimal}f}")
            st.latex(rf"Critical Value (t-Score) = {t_crit:.{decimal}f}")
            st.latex(rf"CI_{{{confidence_level*100:.1f}\%}} = \left({lower:.{decimal}f}, {upper:.{decimal}f}\right)")

            # Plot with shaded CI
            fig, ax = plt.subplots()
            ax.hist(data, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
            ax.axvline(lower, color="red", linestyle="--", label="Lower CI")
            ax.axvline(upper, color="green", linestyle="--", label="Upper CI")
            ax.axvspan(lower, upper, color="yellow", alpha=0.3)
            ax.set_title("Histogram with Confidence Interval")
            ax.legend()
            st.pyplot(fig)

    # 5. Sample Size for Mean
    elif choice == categories[4]:
        confidence_level = st.number_input("Confidence level", value=0.95)
        sigma = st.number_input("Population standard deviation", min_value=0.0)
        moe = st.number_input("Margin of error", min_value=0.0)

        if st.button("Calculate"):
            z = stats.norm.ppf((1 + confidence_level) / 2)
            n_req = (z * sigma / moe)**2
            st.write("Required sample size:", int(np.ceil(n_req)))
            st.write(f"Critical Value (Z-Score) = {round_value(z, decimal)}")

    # 6-9. CI for Variance & Std Dev (with data)
    elif choice in [categories[6], categories[8]]:
        st.write("Upload CSV/Excel or enter data manually.")

        file = st.file_uploader("Upload file", type=["csv", "xlsx"])
        if file:
            if file.name.endswith(".csv"):
                data = pd.read_csv(file).iloc[:, 0]
            else:
                data = pd.read_excel(file).iloc[:, 0]
        else:
            raw_data = st.text_area("Enter values separated by commas")
            data = pd.Series([float(x) for x in raw_data.split(",") if x.strip() != ""]) if raw_data else None

        confidence_level = st.number_input("Confidence level", value=0.95)

        if st.button("Calculate") and data is not None and len(data) > 0:
            n = len(data)
            var = np.var(data, ddof=1)
            df = n - 1
            chi2_lower = stats.chi2.ppf((1 - confidence_level) / 2, df=df)
            chi2_upper = stats.chi2.ppf(1 - (1 - confidence_level) / 2, df=df)

            if choice == categories[6]:  # Variance
                lower = (df * var) / chi2_upper
                upper = (df * var) / chi2_lower
                st.latex(rf"s^2 = {var:.{decimal}f}")
                st.latex(rf"Critical Values (Chi-Square): Lower = {chi2_lower:.{decimal}f}, Upper = {chi2_upper:.{decimal}f}")
                st.latex(rf"CI = \left({lower:.{decimal}f}, {upper:.{decimal}f}\right)")
            else:  # Std Dev
                lower = np.sqrt((df * var) / chi2_upper)
                upper = np.sqrt((df * var) / chi2_lower)
                st.latex(rf"s = {np.sqrt(var):.{decimal}f}")
                st.latex(rf"Critical Values (Chi-Square): Lower = {chi2_lower:.{decimal}f}, Upper = {chi2_upper:.{decimal}f}")
                st.latex(rf"CI = \left({lower:.{decimal}f}, {upper:.{decimal}f}\right)")

            # Histogram with shaded CI
            fig, ax = plt.subplots()
            ax.hist(data, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
            ax.axvline(lower, color="red", linestyle="--", label="Lower CI")
            ax.axvline(upper, color="green", linestyle="--", label="Upper CI")
            ax.axvspan(lower, upper, color="yellow", alpha=0.3)
            ax.set_title("Histogram with Confidence Interval")
            ax.legend()
            st.pyplot(fig)
