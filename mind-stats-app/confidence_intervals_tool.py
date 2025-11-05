import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

# ---------- Helper Functions ----------
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
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    return df[col].dropna().to_numpy()
            st.error("No numeric column found in file.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    return None

# ---------- Main App ----------
def run():
    st.header("🔮 Confidence Interval Calculator")

    categories = [
        "Confidence Interval for Proportion",
        "Sample Size for Proportion",
        "Confidence Interval for Mean (Known SD)",
        "Confidence Interval for Mean (Given Sample SD)",
        "Confidence Interval for Mean (With Data)",
        "Sample Size for Mean",
        "Confidence Interval for Variance (Without Data)",
        "Confidence Interval for Variance (With Data)",
        "Confidence Interval for Standard Deviation (Without Data)",
        "Confidence Interval for Standard Deviation (With Data)"
    ]

    choice = st.selectbox(
        "Choose a category:",
        categories,
        index=None,
        placeholder="Select a confidence interval type..."
    )

    if not choice:
        st.info("👆 Please select a category to begin.")
        return

    decimal = st.number_input("Decimal places for output", min_value=0, max_value=10, value=4)

    # ---------------- Helper to get data ----------------
    def get_data():
        uploaded_data = load_uploaded_data()
        raw_input = st.text_area("Or enter comma-separated values:")
        if uploaded_data is not None:
            return uploaded_data
        elif raw_input:
            try:
                return np.array([float(x.strip()) for x in raw_input.split(",") if x.strip() != ""])
            except:
                st.error("❌ Invalid data input. Please check your entries.")
                return None
        else:
            st.warning("⚠️ Please provide data via file upload or manual entry.")
            return None

    # =====================================================
    # 1. CI for Proportion
    # =====================================================
    if choice == categories[0]:
        n = st.number_input("Sample size (n)", min_value=1, step=1)
        x = st.number_input("Number of successes (x)", min_value=0, max_value=n, step=1)
        confidence_level = st.number_input("Confidence level (0–1)", min_value=0.0, max_value=1.0, value=0.95, format="%.3f")

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
Point estimator (p̂) = {p_hat:.{decimal}f}
Critical Value (Z) = {z:.{decimal}f}
Standard Error = {se:.{decimal}f}
{confidence_level*100:.1f}% CI = ({lower:.{decimal}f}, {upper:.{decimal}f})
""")

    # =====================================================
    # 2. Sample Size for Proportion
    # =====================================================
    elif choice == categories[1]:
        confidence_level = st.number_input("Confidence level", value=0.95, format="%.3f")
        p_est = st.number_input("Estimated proportion (p̂)", value=0.5, min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
        moe = st.number_input("Margin of error (E)", value=0.05, min_value=0.000001, step=0.001, format="%.6f")

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + confidence_level) / 2)
            n_req = (z**2 * p_est * (1 - p_est)) / (moe**2)
            st.text(f"""
=====================
Sample Size for Proportion
=====================
Confidence Level = {confidence_level*100:.1f}%
Estimated p̂ = {p_est:.{decimal}f}
Critical Value (Z) = {z:.{decimal}f}
Margin of Error (E) = {moe}
Required Sample Size (n) = {np.ceil(n_req):.0f}
""")

    # =====================================================
    # 3. CI for Mean (Known SD)
    # =====================================================
    elif choice == categories[2]:
        mean = st.number_input("Sample mean")
        sd = st.number_input("Population standard deviation (σ)", min_value=0.0, format="%.4f")
        n = st.number_input("Sample size (n)", min_value=1, step=1)
        confidence_level = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + confidence_level) / 2)
            se = sd / np.sqrt(n)
            moe = z * se
            lower, upper = mean - moe, mean + moe
            st.text(f"""
=====================
Confidence Interval for Mean (Known SD)
=====================
Sample mean = {mean:.{decimal}f}
Population SD (σ) = {sd:.{decimal}f}
Sample size = {n}
Critical Value (Z) = {z:.{decimal}f}
Standard Error = {se:.{decimal}f}
{confidence_level*100:.1f}% CI = ({lower:.{decimal}f}, {upper:.{decimal}f})
""")

    # =====================================================
    # 4. CI for Mean (Given Sample SD)
    # =====================================================
    elif choice == categories[3]:
        mean = st.number_input("Sample mean")
        sd = st.number_input("Sample standard deviation (s)", min_value=0.0, format="%.4f")
        n = st.number_input("Sample size (n)", min_value=2, step=1)
        confidence_level = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            df = n - 1
            t_crit = stats.t.ppf((1 + confidence_level)/2, df=df)
            se = sd / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe
            st.text(f"""
=====================
Confidence Interval for Mean (Given Sample SD)
=====================
Sample mean = {mean:.{decimal}f}
Sample SD (s) = {sd:.{decimal}f}
Sample size = {n}
Critical Value (t) = {t_crit:.{decimal}f}
Standard Error = {se:.{decimal}f}
{confidence_level*100:.1f}% CI = ({lower:.{decimal}f}, {upper:.{decimal}f})
""")

    # =====================================================
    # 5. Confidence Interval for Mean (With Raw Data)
    # =====================================================
    elif choice == categories[4]:
        st.subheader("📊 Confidence Interval for Mean (Using Sample Data)")

        data = load_uploaded_data()
        raw_input = st.text_area("Or enter comma-separated values:")

        if data is None and raw_input:
            try:
                data = np.array([float(x.strip()) for x in raw_input.split(",") if x.strip() != ""])
            except Exception:
                st.error("❌ Invalid input detected. Please enter only numeric values separated by commas.")
                data = None

        confidence_level = st.number_input(
            "Confidence level (e.g., 0.95)",
            value=0.95, min_value=0.0, max_value=1.0, step=0.001, format="%.3f"
        )

        if st.button("👨‍💻 Calculate"):
            if data is None or len(data) == 0:
                st.warning("⚠️ Please provide sample data before calculating.")
                st.stop()
            if len(data) < 2:
                st.error("❌ You need at least two observations to compute a t-interval (sample SD).")
                st.stop()

            n = len(data)
            mean = float(np.mean(data))
            sd = float(np.std(data, ddof=1))
            df = n - 1
            t_crit = float(stats.t.ppf((1 + confidence_level) / 2, df))
            se = sd / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe

            st.text(f"""
=====================
Confidence Interval for Mean (With Data)
=====================
Sample size (n)     = {n}
Degrees of freedom  = {df}
Sample mean         = {mean:.{decimal}f}
Sample SD (s)       = {sd:.{decimal}f}
Standard Error (SE) = {se:.{decimal}f}
Critical Value (t)  = {t_crit:.{decimal}f}
Margin of Error (E) = {moe:.{decimal}f}

{confidence_level*100:.1f}% Confidence Interval:
({lower:.{decimal}f}, {upper:.{decimal}f})
=====================
""")

            # Descriptive Summary Table
            st.write("**Descriptive Summary:**")
            summary = pd.DataFrame({
                "Statistic": [
                    "Count (n)", "Mean", "Standard Deviation", "Standard Error",
                    "t Critical", "Margin of Error", "CI Lower", "CI Upper"
                ],
                "Value": [
                    n, round(mean, decimal), round(sd, decimal), round(se, decimal),
                    round(t_crit, decimal), round(moe, decimal),
                    round(lower, decimal), round(upper, decimal)
                ]
            })
            st.dataframe(summary, use_container_width=True)

    # =====================================================
    # 6. Sample Size for Mean
    # =====================================================
    elif choice == categories[5]:
        confidence_level = st.number_input("Confidence level", value=0.95, format="%.3f")
        sigma = st.number_input("Population SD (σ)", min_value=0.0, format="%.4f")
        moe = st.number_input("Margin of error (E)", min_value=0.000001, value=0.05, step=0.001, format="%.6f")

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + confidence_level)/2)
            n_req = (z * sigma / moe)**2
            st.text(f"""
=====================
Sample Size for Mean
=====================
Confidence Level = {confidence_level*100:.1f}%
Critical Value (Z) = {z:.{decimal}f}
Population SD (σ) = {sigma}
Margin of Error (E) = {moe}
Required Sample Size (n) = {np.ceil(n_req):.0f}
""")

    # =====================================================
    # 7–10. CI for Variance / SD
    # =====================================================
    elif choice in categories[6:]:
        data = None
        if "With Data" in choice:
            data = get_data()

        if "Without Data" in choice:
            n = st.number_input("Sample size (n)", min_value=2, step=1)
            if "Variance" in choice:
                var = st.number_input("Sample variance (s²)", min_value=0.0, format="%.6f")
            else:
                sd = st.number_input("Sample SD (s)", min_value=0.0, format="%.6f")
        else:
            if data is not None and len(data) > 0:
                n = len(data)
                var = np.var(data, ddof=1)
                sd = np.sqrt(var)
            else:
                st.warning("⚠️ Please provide sample data.")
                return

        confidence_level = st.number_input("Confidence level", value=0.95, format="%.3f")
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
Critical Values (χ²): Lower = {chi2_lower:.{decimal}f}, Upper = {chi2_upper:.{decimal}f}
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
Critical Values (χ²): Lower = {chi2_lower:.{decimal}f}, Upper = {chi2_upper:.{decimal}f}
{confidence_level*100:.1f}% CI for SD = ({lower:.{decimal}f}, {upper:.{decimal}f})
""")

# ---------- Run ----------
if __name__ == "__main__":
    run()
