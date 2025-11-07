# ==========================================================
# confidence_intervals_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats


# ---------- Helper Functions ----------
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
       st.markdown("""
    ---
    ### 🧭 **Quick Reference**
    - \( \\bar{X} \): sample mean  \( s \): sample SD  \( \\sigma \): population SD  
    - \( \\hat{p} \): sample proportion  \( E \): margin of error  \( n \): sample size  
    - \( \\chi^2 \): chi-square critical values for variance/SD intervals  
    ---
    """)

    categories = [
        "Confidence Interval for Proportion",
        "Sample Size for Proportion",
        "Confidence Interval for Mean (Known SD)",
        "Confidence Interval for Mean (Given Sample SD)",
        "Sample Size for Mean",
        "Confidence Interval for Variance / SD"
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

    decimal = st.number_input("Decimal places for output", 0, 10, 4)

    # =====================================================
    # 1. CI for Proportion
    # =====================================================
    if choice == categories[0]:
        st.latex(r"\text{CI for } p:\quad \hat p \pm Z_{\alpha/2}\sqrt{\frac{\hat p(1-\hat p)}{n}}")

        n = st.number_input("Sample size (n)", 1, step=1)
        x = st.number_input("Number of successes (x)", 0, n, step=1)
        confidence_level = st.number_input("Confidence level", 0.0, 1.0, 0.95)

        if st.button("👨‍💻 Calculate"):
            p_hat = x / n
            z = stats.norm.ppf((1 + confidence_level) / 2)
            se = np.sqrt(p_hat * (1 - p_hat) / n)
            moe = z * se
            lower, upper = p_hat - moe, p_hat + moe

            st.text(f"p̂ = {p_hat:.{decimal}f},  SE = {se:.{decimal}f},  Z = {z:.{decimal}f}")
            st.latex(r"E = Z_{\alpha/2}\sqrt{\frac{\hat p(1-\hat p)}{n}}")
            st.latex(rf"E = ({z:.4f})\sqrt{{\frac{{({p_hat:.4f})(1-{p_hat:.4f})}}{{{n}}}}} = {moe:.4f}")
            st.latex(rf"\boxed{{CI = ({lower:.4f},\; {upper:.4f})}}")


    # =====================================================
    # 2. Sample Size for Proportion
    # =====================================================
    elif choice == categories[1]:
        st.latex(r"n = \frac{Z_{\alpha/2}^2\,\hat p(1-\hat p)}{E^2}")

        confidence_level = st.number_input("Confidence level", 0.0, 1.0, 0.95)
        p_est = st.number_input("Estimated p̂", 0.0, 1.0, 0.5)
        moe = st.number_input("Margin of error (E)", 0.000001, 1.0, 0.05)

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + confidence_level) / 2)
            n_req = (z**2 * p_est * (1 - p_est)) / (moe**2)
            st.latex(rf"Z={z:.4f},\; \hat p={p_est:.4f},\; E={moe:.4f}")
            st.latex(rf"n^*=\frac{{({z:.4f})^2({p_est:.4f})(1-{p_est:.4f})}}{{({moe:.4f})^2}}={n_req:.4f}")
            st.latex(rf"\boxed{{n=\lceil n^* \rceil={np.ceil(n_req):.0f}}}")


    # =====================================================
    # 3. CI for Mean (Known SD)
    # =====================================================
    elif choice == categories[2]:
        st.latex(r"\text{CI for } \mu:\quad \bar X \pm Z_{\alpha/2}\frac{\sigma}{\sqrt n}")

        mean = st.number_input("Sample mean (x̄)")
        sd = st.number_input("Population SD (σ)", 0.0)
        n = st.number_input("Sample size (n)", 1, step=1)
        confidence_level = st.number_input("Confidence level", 0.0, 1.0, 0.95)

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + confidence_level)/2)
            se = sd / np.sqrt(n)
            moe = z * se
            lower, upper = mean - moe, mean + moe

            st.latex(rf"E=({z:.4f})\frac{{{sd:.4f}}}{{\sqrt{{{n}}}}}={moe:.4f}")
            st.latex(rf"\boxed{{CI=({lower:.4f},\;{upper:.4f})}}")


    # =====================================================
    # 4. CI for Mean (Given Sample SD)
    # =====================================================
    elif choice == categories[3]:
        st.latex(r"\text{CI for } \mu:\quad \bar X \pm t_{\alpha/2,\,n-1}\frac{s}{\sqrt n}")

        mean = st.number_input("Sample mean (x̄)")
        sd = st.number_input("Sample SD (s)", 0.0)
        n = st.number_input("Sample size (n)", 2, step=1)
        confidence_level = st.number_input("Confidence level", 0.0, 1.0, 0.95)

        if st.button("👨‍💻 Calculate"):
            df = n - 1
            t_crit = stats.t.ppf((1 + confidence_level)/2, df)
            se = sd / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe

            st.latex(rf"t={t_crit:.4f},\; s={sd:.4f},\; n={n}")
            st.latex(rf"E=({t_crit:.4f})\frac{{{sd:.4f}}}{{\sqrt{{{n}}}}}={moe:.4f}")
            st.latex(rf"\boxed{{CI=({lower:.4f},\;{upper:.4f})}}")


    # =====================================================
    # 5. Sample Size for Mean
    # =====================================================
    elif choice == categories[4]:
        st.latex(r"n = \left(\frac{Z_{\alpha/2}\sigma}{E}\right)^2")

        confidence_level = st.number_input("Confidence level", 0.0, 1.0, 0.95)
        sigma = st.number_input("Population SD (σ)", 0.0)
        moe = st.number_input("Margin of error (E)", 0.000001, 1.0, 0.05)

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + confidence_level)/2)
            n_req = (z * sigma / moe)**2
            st.latex(rf"Z={z:.4f},\; \sigma={sigma:.4f},\; E={moe:.6f}")
            st.latex(rf"n^*=\left(\frac{{({z:.4f})({sigma:.4f})}}{{{moe:.6f}}}\right)^2={n_req:.4f}")
            st.latex(rf"\boxed{{n=\lceil n^* \rceil={np.ceil(n_req):.0f}}}")


    # =====================================================
    # 6. Confidence Interval for Variance / SD
    # =====================================================
    elif choice == categories[5]:
        st.latex(r"\text{CI for } \sigma^2:\quad \left(\frac{(n-1)s^2}{\chi^2_{upper}},\; \frac{(n-1)s^2}{\chi^2_{lower}}\right)")
        st.latex(r"\text{CI for } \sigma:\quad \left(\sqrt{\frac{(n-1)s^2}{\chi^2_{upper}}},\; \sqrt{\frac{(n-1)s^2}{\chi^2_{lower}}}\right)")

        n = st.number_input("Sample size (n)", 2, step=1)
        sd = st.number_input("Sample SD (s)", 0.0)
        confidence_level = st.number_input("Confidence level", 0.0, 1.0, 0.95)

        if st.button("👨‍💻 Calculate"):
            df = n - 1
            chi2_lower = stats.chi2.ppf((1 - confidence_level)/2, df)
            chi2_upper = stats.chi2.ppf(1 - (1 - confidence_level)/2, df)
            var = sd**2
            var_lower = df * var / chi2_upper
            var_upper = df * var / chi2_lower
            sd_lower, sd_upper = np.sqrt(var_lower), np.sqrt(var_upper)

            st.markdown("### 🧮 Step-by-Step Solution")
            st.latex(rf"df={df},\; s^2={var:.4f}")
            st.latex(rf"\chi^2_{{lower}}={chi2_lower:.4f},\; \chi^2_{{upper}}={chi2_upper:.4f}")
            st.latex(r"\text{Variance Interval: } \left(\frac{(n-1)s^2}{\chi^2_{upper}},\; \frac{(n-1)s^2}{\chi^2_{lower}}\right)")
            st.latex(rf"({var_lower:.4f},\; {var_upper:.4f})")
            st.latex(r"\text{SD Interval: } \left(\sqrt{\frac{(n-1)s^2}{\chi^2_{upper}}},\; \sqrt{\frac{(n-1)s^2}{\chi^2_{lower}}}\right)")
            st.latex(rf"\boxed{{({sd_lower:.4f},\; {sd_upper:.4f})}}")

# ---------- Run ----------
if __name__ == "__main__":
    run()
