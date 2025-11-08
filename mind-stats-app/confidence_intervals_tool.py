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
    return round(float(value), decimals)

def load_uploaded_data():
    uploaded_file = st.file_uploader(
        "📂 Upload CSV or Excel file with a single column of numeric data",
        type=["csv", "xlsx"]
    )
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
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
    st.header("🔮 MIND: Confidence Interval Calculator (Text-Only Edition)")
    st.markdown("---")

    categories = [
        "Confidence Interval for Proportion (p, z)",
        "Sample Size for Proportion (p, z, E)",
        "Confidence Interval for Mean (σ known, z)",
        "Confidence Interval for Mean (s given, t)",
        "Confidence Interval for Mean (with data, t)",
        "Sample Size for Mean (σ known, z, E)",
        "Confidence Interval for Variance & SD (χ²)",
        "Confidence Interval for Variance & SD (with data, χ²)"
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
    # 1) Confidence Interval for Proportion (p, z)
    # ==========================================================
    if choice == categories[0]:
        x = st.number_input("Number of successes (x)", min_value=0, step=1)
        n = st.number_input("Sample size (n)", min_value=max(1, int(x)), step=1)
        conf = st.number_input("Confidence level (0–1)", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            p_hat = x / n
            z = stats.norm.ppf((1 + conf) / 2)
            se = np.sqrt(p_hat * (1 - p_hat) / n)
            moe = z * se
            lower, upper = p_hat - moe, p_hat + moe

            st.latex(r"\hat{p} \pm z_{\alpha/2}\sqrt{\dfrac{\hat{p}(1-\hat{p})}{n}}")
            st.text(f"""
1) Inputs: x={int(x)}, n={int(n)}, p̂={p_hat:.{decimal}f}, confidence={conf:.3f}
2) z_(α/2)={z:.{decimal}f}
3) SE=√[p̂(1−p̂)/n]={se:.{decimal}f}
4) E=z·SE={moe:.{decimal}f}
5) CI=({lower:.{decimal}f}, {upper:.{decimal}f})
""")

            st.success(f"✅ {conf*100:.1f}% CI for p: ({lower:.{decimal}f}, {upper:.{decimal}f})")

    # ==========================================================
    # 2) Sample Size for Proportion
    # ==========================================================
    elif choice == categories[1]:
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        p_est = st.number_input("Estimated proportion (p̂)", value=0.5, min_value=0.0, max_value=1.0, step=0.001)
        E = st.number_input("Margin of error (E)", value=0.05, min_value=0.000001, step=0.001)

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + conf) / 2)
            n_req = p_est * (1 - p_est) * (z / E) ** 2
            st.latex(r"n = \hat{p}(1-\hat{p})\!\left(\dfrac{Z_{\alpha/2}}{E}\right)^{2}")
            st.success(f"✅ Required sample size: n = {np.ceil(n_req):.0f}")

    # ==========================================================
    # 3) Mean CI (σ known, z)
    # ==========================================================
    elif choice == categories[2]:
        mean = st.number_input("Sample mean (x̄)")
        sigma = st.number_input("Population SD (σ)", min_value=0.0)
        n = st.number_input("Sample size (n)", min_value=1, step=1)
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + conf) / 2)
            se = sigma / np.sqrt(n)
            moe = z * se
            lower, upper = mean - moe, mean + moe

            st.latex(r"\bar{X} \pm z_{\alpha/2}\!\left(\dfrac{\sigma}{\sqrt{n}}\right)")
            st.success(f"✅ {conf*100:.1f}% CI for μ: ({lower:.{decimal}f}, {upper:.{decimal}f})")

    # ==========================================================
    # 4) Mean CI (s given, t)
    # ==========================================================
    elif choice == categories[3]:
        mean = st.number_input("Sample mean (x̄)")
        s = st.number_input("Sample SD (s)", min_value=0.0)
        n = st.number_input("Sample size (n)", min_value=2, step=1)
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            df = int(n - 1)
            t_crit = stats.t.ppf((1 + conf) / 2, df)
            se = s / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe

            st.latex(r"\bar{X} \pm t_{\alpha/2,\,n-1}\!\left(\dfrac{s}{\sqrt{n}}\right)")
            st.success(f"✅ {conf*100:.1f}% CI for μ: ({lower:.{decimal}f}, {upper:.{decimal}f})")

    # ==========================================================
    # 5) Mean CI (with data, t)
    # ==========================================================
    elif choice == categories[4]:
        st.subheader("📊 Confidence Interval for Mean (with data, t)")
        data = load_uploaded_data()
        raw_input = st.text_area("Or enter comma-separated values:")
        if data is None and raw_input:
            try:
                data = np.array([float(x.strip()) for x in raw_input.split(",") if x.strip()])
            except:
                st.error("❌ Invalid input. Use numeric comma-separated values only.")
                return

        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            if data is None or len(data) < 2:
                st.warning("⚠️ Provide at least two data points.")
                return

            n, mean, s = len(data), np.mean(data), np.std(data, ddof=1)
            df = n - 1
            t_crit = stats.t.ppf((1 + conf) / 2, df)
            se = s / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe

            st.latex(r"\bar{X} \pm t_{\alpha/2,\,n-1}\!\left(\dfrac{s}{\sqrt{n}}\right)")
            st.success(f"✅ {conf*100:.1f}% CI for μ: ({lower:.{decimal}f}, {upper:.{decimal}f})")

            summary = pd.DataFrame({
                "Statistic": ["n", "Mean (x̄)", "SD (s)", "SE", "t critical", "MOE", "Lower", "Upper"],
                "Value": [n, round_value(mean, decimal), round_value(s, decimal),
                          round_value(se, decimal), round_value(t_crit, decimal),
                          round_value(moe, decimal), round_value(lower, decimal),
                          round_value(upper, decimal)]
            })
            st.dataframe(summary, use_container_width=True)

    # ==========================================================
    # 6) Sample Size for Mean (σ known, z, E)
    # ==========================================================
    elif choice == categories[5]:
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        sigma = st.number_input("Population SD (σ)", min_value=0.0)
        E = st.number_input("Margin of error (E)", value=0.05, min_value=0.000001)

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + conf) / 2)
            n_req = (z * sigma / E) ** 2
            st.latex(r"n = \left(\dfrac{z_{\alpha/2}\sigma}{E}\right)^2")
            st.success(f"✅ Required sample size: n = {np.ceil(n_req):.0f}")

    # ==========================================================
    # 7) Variance & SD CI (χ²)
    # ==========================================================
    elif choice == categories[6]:
        n = st.number_input("Sample size (n)", min_value=2, step=1)
        s = st.number_input("Sample SD (s)", min_value=0.0)
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            df = int(n - 1)
            chi2_lower = stats.chi2.ppf((1 - conf) / 2, df)
            chi2_upper = stats.chi2.ppf(1 - (1 - conf) / 2, df)
            s2 = s ** 2
            var_lower = df * s2 / chi2_upper
            var_upper = df * s2 / chi2_lower
            sd_lower, sd_upper = np.sqrt(var_lower), np.sqrt(var_upper)

            st.latex(r"\text{Var CI: } \left(\dfrac{(n-1)s^2}{\chi^2_{(1-\alpha/2)}}, \dfrac{(n-1)s^2}{\chi^2_{(\alpha/2)}}\right)")
            st.success(f"✅ Variance CI: ({var_lower:.{decimal}f}, {var_upper:.{decimal}f}) | SD CI: ({sd_lower:.{decimal}f}, {sd_upper:.{decimal}f})")

    # ==========================================================
    # 8) Variance & SD CI (χ²) with Data
    # ==========================================================
    else:
        st.subheader("📊 Confidence Interval for Variance & SD (with data, χ²)")
        data = load_uploaded_data()
        raw_input = st.text_area("Or enter comma-separated values:")
        if data is None and raw_input:
            try:
                data = np.array([float(x.strip()) for x in raw_input.split(",") if x.strip()])
            except:
                st.error("❌ Invalid input.")
                return

        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            if data is None or len(data) < 2:
                st.warning("⚠️ Provide at least two data points.")
                return

            n = len(data)
            df = n - 1
            s2 = np.var(data, ddof=1)
            chi2_lower = stats.chi2.ppf((1 - conf)/2, df)
            chi2_upper = stats.chi2.ppf(1 - (1 - conf)/2, df)
            var_lower = df * s2 / chi2_upper
            var_upper = df * s2 / chi2_lower
            sd_lower, sd_upper = np.sqrt(var_lower), np.sqrt(var_upper)

            st.latex(r"\text{CI for Variance and SD using χ² distribution}")
            st.success(f"✅ Variance CI: ({var_lower:.{decimal}f}, {var_upper:.{decimal}f}) | SD CI: ({sd_lower:.{decimal}f}, {sd_upper:.{decimal}f})")

            summary = pd.DataFrame({
                "Statistic": ["n", "Variance (s²)", "df", "χ² lower", "χ² upper", "Var Lower", "Var Upper", "SD Lower", "SD Upper"],
                "Value": [n, round_value(s2, decimal), df,
                          round_value(chi2_lower, decimal), round_value(chi2_upper, decimal),
                          round_value(var_lower, decimal), round_value(var_upper, decimal),
                          round_value(sd_lower, decimal), round_value(sd_upper, decimal)]
            })
            st.dataframe(summary, use_container_width=True)

# ==========================================================
# Run app directly
# ==========================================================
if __name__ == "__main__":
    run()
