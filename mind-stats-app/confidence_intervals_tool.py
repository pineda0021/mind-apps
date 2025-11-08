# ==========================================================
# confidence_intervals_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

# ==========================================================
# Helper Functions
# ==========================================================
def round_value(value, decimals=4):
    """Round numeric values consistently"""
    return round(value, decimals)


def load_uploaded_data():
    """Load numeric data from CSV or Excel file"""
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
            st.error("❌ No numeric column found in uploaded file.")
        except Exception as e:
            st.error(f"⚠️ Error reading file: {e}")
    return None


# ==========================================================
# Main App
# ==========================================================
def run():
    st.header("🔮 MIND: Confidence Interval Calculator")
    st.markdown("---")

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

    # ==========================================================
    # 1. Confidence Interval for Proportion
    # ==========================================================
    if choice == categories[0]:
        n = st.number_input("Sample size (n)", min_value=1, step=1)
        x = st.number_input("Number of successes (x)", min_value=0, max_value=n, step=1)
        conf = st.number_input("Confidence level (0–1)", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            p_hat = x / n
            z = stats.norm.ppf((1 + conf) / 2)
            se = np.sqrt(p_hat * (1 - p_hat) / n)
            moe = z * se
            lower, upper = p_hat - moe, p_hat + moe

            st.latex(r"\hat{p} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}")
            st.text(f"""
=====================
Confidence Interval for Proportion
=====================
Sample successes = {x}
Sample size       = {n}
p̂ (Sample Proportion) = {p_hat:.{decimal}f}
Z Critical Value        = {z:.{decimal}f}
Standard Error          = {se:.{decimal}f}
Margin of Error (E)     = {moe:.{decimal}f}
{conf*100:.1f}% CI = ({lower:.{decimal}f}, {upper:.{decimal}f})
""")

    # ==========================================================
    # 2. Sample Size for Proportion
    # ==========================================================
    elif choice == categories[1]:
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        p_est = st.number_input("Estimated proportion (p̂)", value=0.5, min_value=0.0, max_value=1.0, step=0.001)
        E = st.number_input("Margin of error (E)", value=0.05, min_value=0.000001, step=0.001)

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + conf) / 2)
            n_req = (z**2 * p_est * (1 - p_est)) / (E**2)
            st.text(f"""
=====================
Sample Size for Proportion
=====================
Confidence Level = {conf*100:.1f}%
Estimated p̂      = {p_est:.{decimal}f}
Z Critical Value  = {z:.{decimal}f}
Margin of Error   = {E}
Required Sample Size (n) = {np.ceil(n_req):.0f}
""")

    # ==========================================================
    # 3. Confidence Interval for Mean (Known SD)
    # ==========================================================
    elif choice == categories[2]:
        mean = st.number_input("Sample mean (x̄)")
        sigma = st.number_input("Population SD (σ)", min_value=0.0, format="%.4f")
        n = st.number_input("Sample size (n)", min_value=1, step=1)
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + conf) / 2)
            se = sigma / np.sqrt(n)
            moe = z * se
            lower, upper = mean - moe, mean + moe

            st.latex(r"\bar{X} \pm z_{\alpha/2}\left(\frac{\sigma}{\sqrt{n}}\right)")
            st.text(f"""
=====================
Confidence Interval for Mean (Known SD)
=====================
Sample mean = {mean:.{decimal}f}
Population SD (σ) = {sigma:.{decimal}f}
Sample size = {n}
Z Critical Value = {z:.{decimal}f}
Standard Error = {se:.{decimal}f}
Margin of Error = {moe:.{decimal}f}
{conf*100:.1f}% CI = ({lower:.{decimal}f}, {upper:.{decimal}f})
""")

    # ==========================================================
    # 4. Confidence Interval for Mean (Given Sample SD)
    # ==========================================================
    elif choice == categories[3]:
        mean = st.number_input("Sample mean (x̄)")
        s = st.number_input("Sample SD (s)", min_value=0.0, format="%.4f")
        n = st.number_input("Sample size (n)", min_value=2, step=1)
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            df = n - 1
            t_crit = stats.t.ppf((1 + conf) / 2, df)
            se = s / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe

            st.latex(r"\bar{X} \pm t_{\alpha/2,\,n-1}\left(\frac{s}{\sqrt{n}}\right)")
            st.text(f"""
=====================
Confidence Interval for Mean (Given Sample SD)
=====================
Sample mean = {mean:.{decimal}f}
Sample SD (s) = {s:.{decimal}f}
Sample size (n) = {n}
Degrees of freedom = {df}
t Critical Value = {t_crit:.{decimal}f}
Standard Error = {se:.{decimal}f}
Margin of Error = {moe:.{decimal}f}
{conf*100:.1f}% CI = ({lower:.{decimal}f}, {upper:.{decimal}f})
""")

    # ==========================================================
    # 5. Confidence Interval for Mean (With Raw Data)
    # ==========================================================
    elif choice == categories[4]:
        st.subheader("📊 Confidence Interval for Mean (Using Raw Data)")
        data = load_uploaded_data()
        raw_input = st.text_area("Or enter comma-separated values:")

        if data is None and raw_input:
            try:
                data = np.array([float(x.strip()) for x in raw_input.split(",") if x.strip() != ""])
            except:
                st.error("❌ Invalid input. Please use only numeric comma-separated values.")
                data = None

        conf = st.number_input("Confidence level (e.g., 0.95)", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            if data is None or len(data) < 2:
                st.warning("⚠️ Please provide at least two data points.")
                st.stop()

            n = len(data)
            mean = np.mean(data)
            s = np.std(data, ddof=1)
            df = n - 1
            t_crit = stats.t.ppf((1 + conf) / 2, df)
            se = s / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe

            st.text(f"""
=====================
Confidence Interval for Mean (With Data)
=====================
Sample size (n) = {n}
Degrees of freedom = {df}
Sample mean = {mean:.{decimal}f}
Sample SD (s) = {s:.{decimal}f}
Standard Error = {se:.{decimal}f}
t Critical Value = {t_crit:.{decimal}f}
Margin of Error = {moe:.{decimal}f}
{conf*100:.1f}% CI = ({lower:.{decimal}f}, {upper:.{decimal}f})
=====================
""")

            # Descriptive Table
            summary = pd.DataFrame({
                "Statistic": [
                    "Count (n)", "Mean", "Standard Deviation", "Standard Error",
                    "t Critical", "Margin of Error", "CI Lower", "CI Upper"
                ],
                "Value": [
                    n, round_value(mean, decimal), round_value(s, decimal),
                    round_value(se, decimal), round_value(t_crit, decimal),
                    round_value(moe, decimal), round_value(lower, decimal), round_value(upper, decimal)
                ]
            })
            st.dataframe(summary, use_container_width=True)

    # ==========================================================
    # 6–10. Variance and SD Confidence Intervals
    # ==========================================================
    else:
        st.subheader(f"📈 {choice}")
        data = None
        if "With Data" in choice:
            data = load_uploaded_data()
            if data is not None:
                n = len(data)
                s2 = np.var(data, ddof=1)
                s = np.sqrt(s2)
            else:
                st.warning("⚠️ Please upload data to proceed.")
                return
        else:
            n = st.number_input("Sample size (n)", min_value=2, step=1)
            if "Variance" in choice:
                s2 = st.number_input("Sample variance (s²)", min_value=0.0, format="%.6f")
                s = np.sqrt(s2)
            else:
                s = st.number_input("Sample SD (s)", min_value=0.0, format="%.6f")
                s2 = s**2

        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        df = n - 1
        chi2_lower = stats.chi2.ppf((1 - conf)/2, df)
        chi2_upper = stats.chi2.ppf(1 - (1 - conf)/2, df)

        if st.button("👨‍💻 Calculate"):
            var_lower = df * s2 / chi2_upper
            var_upper = df * s2 / chi2_lower
            sd_lower, sd_upper = np.sqrt(var_lower), np.sqrt(var_upper)

            st.text(f"""
=====================
{choice}
=====================
Sample size = {n}
Degrees of freedom = {df}
Sample variance = {s2:.{decimal}f}
Sample SD = {s:.{decimal}f}
χ² Critical Values: Lower = {chi2_lower:.{decimal}f}, Upper = {chi2_upper:.{decimal}f}
{conf*100:.1f}% CI for Variance = ({var_lower:.{decimal}f}, {var_upper:.{decimal}f})
{conf*100:.1f}% CI for SD = ({sd_lower:.{decimal}f}, {sd_upper:.{decimal}f})
=====================
""")

            st.dataframe(pd.DataFrame({
                "Statistic": [
                    "Degrees of Freedom", "Sample Variance", "Sample SD",
                    "χ² Lower", "χ² Upper", "Variance CI (Lower)", "Variance CI (Upper)",
                    "SD CI (Lower)", "SD CI (Upper)"
                ],
                "Value": [
                    df, round_value(s2, decimal), round_value(s, decimal),
                    round_value(chi2_lower, decimal), round_value(chi2_upper, decimal),
                    round_value(var_lower, decimal), round_value(var_upper, decimal),
                    round_value(sd_lower, decimal), round_value(sd_upper, decimal)
                ]
            }), use_container_width=True)


# ==========================================================
# Run app directly
# ==========================================================
if __name__ == "__main__":
    run()
