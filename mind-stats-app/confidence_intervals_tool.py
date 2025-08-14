import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def round_value(value, decimals=4):
    return round(value, decimals)

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

def run():
    st.header("🔮 Confidence Interval Calculator")

    categories = [
        "Confidence Interval for Proportion",
        "Sample Size for Proportion",
        "Confidence Interval for Mean (Known SD)",
        "Confidence Interval for Mean (With Data)",
        "Sample Size for Mean",
        "Confidence Interval for Variance (Without Data)",
        "Confidence Interval for Variance (With Data)",
        "Confidence Interval for Standard Deviation (Without Data)",
        "Confidence Interval for Standard Deviation (With Data)"
    ]

    choice = st.selectbox("Choose a category:", categories)
    decimal = st.number_input("Decimal places for output", min_value=0, max_value=10, value=4)

    # ------------------- WITH DATA / RAW INPUT HELPER -------------------
    def get_data():
        uploaded_data = load_uploaded_data()
        raw_input = st.text_area("Or enter comma-separated values")
        if uploaded_data is not None:
            return uploaded_data
        elif raw_input:
            try:
                return np.array([float(x.strip()) for x in raw_input.split(",") if x.strip() != ""])
            except:
                st.error("Invalid data input")
                return None
        else:
            st.warning("Provide data via file or manual entry")
            return None

    # ------------------- 1. CI for Proportion -------------------
    if choice == categories[0]:
        n = st.number_input("Sample size", min_value=1, step=1)
        x = st.number_input("Number of successes", min_value=0, max_value=n, step=1)
        confidence_level = st.number_input("Confidence level (0-1)", min_value=0.0, max_value=1.0, value=0.95)

        if st.button("👨‍💻 Calculate"):
            p_hat = x / n
            se = np.sqrt(p_hat * (1 - p_hat) / n)
            z = stats.norm.ppf((1 + confidence_level) / 2)
            moe = z * se
            lower, upper = p_hat - moe, p_hat + moe
            st.text(f"""
=====================
Confidence Interval for Proportion
=====================
Sample successes = {x}
Sample size = {n}
Sample proportion = {p_hat:.{decimal}f}
Critical Value (Z) = {z:.{decimal}f}
{confidence_level*100:.1f}% CI = ({lower:.{decimal}f}, {upper:.{decimal}f})
""")

    # ------------------- 2. Sample Size for Proportion -------------------
    elif choice == categories[1]:
        confidence_level = st.number_input("Confidence level", value=0.95)
        p_est = st.number_input("Estimated proportion", value=0.5, min_value=0.0, max_value=1.0, step=0.001)
        moe = st.number_input("Margin of error", value=0.05, min_value=0.0, step=0.001)

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + confidence_level) / 2)
            n_req = (z**2 * p_est * (1 - p_est)) / (moe**2)
            st.text(f"Required sample size: {int(np.ceil(n_req))}")

    # ------------------- 3. CI for Mean (Known SD) -------------------
    elif choice == categories[2]:
        mean = st.number_input("Sample mean")
        sd = st.number_input("Population SD", min_value=0.0)
        n = st.number_input("Sample size", min_value=1, step=1)
        confidence_level = st.number_input("Confidence level", value=0.95)

        if st.button("👨‍💻 Calculate"):
            df = n - 1
            t_crit = stats.t.ppf((1 + confidence_level) / 2, df=df)
            se = sd / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe
            st.text(f"""
=====================
Confidence Interval for Mean (Known SD)
=====================
Sample mean = {mean:.{decimal}f}
Population SD = {sd:.{decimal}f}
Sample size = {n}
Critical Value (t) = {t_crit:.{decimal}f}
{confidence_level*100:.1f}% CI = ({lower:.{decimal}f}, {upper:.{decimal}f})
""")

    # ------------------- 4. CI for Mean (With Data) -------------------
    elif choice == categories[3]:
        data = get_data()
        confidence_level = st.number_input("Confidence level", value=0.95)
        if st.button("👨‍💻 Calculate") and data is not None and len(data) > 0:
            n = len(data)
            mean = np.mean(data)
            sd = np.std(data, ddof=1)
            df = n - 1
            t_crit = stats.t.ppf((1 + confidence_level)/2, df=df)
            se = sd / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe
            st.text(f"""
=====================
Confidence Interval for Mean (With Data)
=====================
Sample mean = {mean:.{decimal}f}
Sample SD = {sd:.{decimal}f}
Sample size = {n}
Critical Value (t) = {t_crit:.{decimal}f}
{confidence_level*100:.1f}% CI = ({lower:.{decimal}f}, {upper:.{decimal}f})
""")
            # Histogram
            fig, ax = plt.subplots()
            ax.hist(data, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
            ax.axvline(lower, color="red", linestyle="--", label="Lower CI")
            ax.axvline(upper, color="green", linestyle="--", label="Upper CI")
            ax.axvspan(lower, upper, color="yellow", alpha=0.3)
            ax.set_title("Histogram with Confidence Interval")
            ax.legend()
            st.pyplot(fig)

    # ------------------- 5. Sample Size for Mean -------------------
    elif choice == categories[4]:
        confidence_level = st.number_input("Confidence level", value=0.95)
        sigma = st.number_input("Population SD", min_value=0.0)
        moe = st.number_input("Margin of error", min_value=0.0)

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + confidence_level)/2)
            n_req = (z * sigma / moe)**2
            st.text(f"Required sample size: {int(np.ceil(n_req))}")

    # ------------------- 6-9. CI for Variance / SD -------------------
    elif choice in categories[5:]:
        data = None
        if "With Data" in choice:
            data = get_data()

        if "Without Data" in choice:
            n = st.number_input("Sample size", min_value=1, step=1)
            if "Variance" in choice:
                var = st.number_input("Sample variance", min_value=0.0)
            else:
                sd = st.number_input("Sample SD", min_value=0.0)
        else:
            if data is not None and len(data) > 0:
                n = len(data)
                var = np.var(data, ddof=1)
                sd = np.sqrt(var)
            else:
                st.warning("Provide data")
                return

        confidence_level = st.number_input("Confidence level", value=0.95)
        df = n - 1
        chi2_lower = stats.chi2.ppf((1 - confidence_level)/2, df=df)
        chi2_upper = stats.chi2.ppf(1 - (1 - confidence_level)/2, df=df)

        if st.button("👨‍💻 Calculate"):
            if "Variance" in choice:
                lower = df * var / chi2_upper
                upper = df * var / chi2_lower
                st.text(f"""
=====================
{choice}
=====================
Sample size = {n}
Sample variance = {var:.{decimal}f}
Critical Values (Chi2): Lower = {chi2_lower:.{decimal}f}, Upper = {chi2_upper:.{decimal}f}
{confidence_level*100:.1f}% CI for Variance = ({lower:.{decimal}f}, {upper:.{decimal}f})
""")
            else:
                lower = np.sqrt(df * sd**2 / chi2_upper)
                upper = np.sqrt(df * sd**2 / chi2_lower)
                st.text(f"""
=====================
{choice}
=====================
Sample size = {n}
Sample SD = {sd:.{decimal}f}
Critical Values (Chi2): Lower = {chi2_lower:.{decimal}f}, Upper = {chi2_upper:.{decimal}f}
{confidence_level*100:.1f}% CI for SD = ({lower:.{decimal}f}, {upper:.{decimal}f})
""")

if __name__ == "__main__":
    run()

