# ==========================================================
# inferences_one_sample_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import math
import numpy as np
import pandas as pd
from scipy.stats import norm, t, chi2, binom

# ==========================================================
# Auto Light/Dark Mode Box
# ==========================================================
def themed_box(text):
    st.markdown(f"""
        <style>
            .themed-box {{
                padding: 12px;
                border-radius: 10px;
                margin-bottom: 10px;
                border-left: 5px solid #007acc;
            }}
            @media (prefers-color-scheme: light) {{
                .themed-box {{
                    background-color: #e6f3ff;
                    color: black;
                }}
            }}
            @media (prefers-color-scheme: dark) {{
                .themed-box {{
                    background-color: #2b2b2b;
                    color: white;
                }}
            }}
        </style>
        <div class="themed-box">{text}</div>
    """, unsafe_allow_html=True)


# ==========================================================
# Helper: Upload Numeric Data
# ==========================================================
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


# ==========================================================
# Main App
# ==========================================================
def run_hypothesis_tool():
    st.header("🔎 Inferences on One Sample")

    # Decimal places for ALL output
    decimals = st.number_input(
        "Decimal places for output:",
        min_value=0,
        max_value=10,
        value=4,
        step=1
    )

    fmt = f"{{:.{decimals}f}}"

    test_options = [
        "Proportion test (large sample)",
        "Proportion test (small sample, binomial)",
        "t-test for population mean (summary stats)",
        "t-test for population mean (raw data)",
        "Chi-squared test for std dev (summary stats)",
        "Chi-squared test for std dev (raw data)"
    ]

    test_choice = st.selectbox(
        "Choose a hypothesis test:",
        test_options,
        index=None,
        placeholder="Select a test to begin..."
    )

    if not test_choice:
        st.info("👆 Please select a hypothesis test to begin.")
        return

    alpha = st.number_input(
        "Significance level (α)",
        value=0.05,
        min_value=0.001,
        max_value=0.5,
        step=0.01
    )

    tails = st.selectbox("Tail type:", ["two", "left", "right"])

    # ==========================================================
    # PROPORTION TESTS
    # ==========================================================
    if test_choice in ["Proportion test (large sample)", "Proportion test (small sample, binomial)"]:
        x = st.number_input("Number of successes (x)", min_value=0, step=1)
        n = st.number_input("Sample size (n)", min_value=1, step=1)
        p0 = st.number_input("Null proportion (p₀)", min_value=0.0, max_value=1.0, format="%.6f")

        if st.button("👨‍💻 Calculate"):
            p_hat = x / n

            st.markdown("### 📘 Step-by-Step Solution")
            themed_box("**Step 1:** Compute the sample proportion")
            st.latex(r"\hat{p} = \frac{x}{n}")
            st.latex(fr"\hat{{p}} = \frac{{{x}}}{{{n}}} = {fmt.format(p_hat)}")

            # ------------------------------------------------------
            # LARGE SAMPLE: Z TEST
            # ------------------------------------------------------
            if test_choice == "Proportion test (large sample)":
                st.markdown("### 🧮 Large Sample Z-Test")
                st.latex(r"z = \frac{\hat{p} - p_0}{\sqrt{p_0(1 - p_0)/n}}")

                themed_box("**Step 2:** Compute the Standard Error and z statistic.")
                se = math.sqrt(p0 * (1 - p0) / n)
                z_stat = (p_hat - p0) / se
                st.latex(fr"\text{{SE}} = \sqrt{{p_0(1-p_0)/n}} = {fmt.format(se)}")
                st.latex(fr"z = \frac{{\hat p - p_0}}{{SE}} = {fmt.format(z_stat)}")

                themed_box("**Step 3:** Compute p-value and compare with α")

                if tails == "left":
                    z_crit = -abs(norm.ppf(alpha))
                    p_val = norm.cdf(z_stat)
                    reject = z_stat < z_crit
                    crit_str = fmt.format(z_crit)

                elif tails == "right":
                    z_crit = abs(norm.ppf(1 - alpha))
                    p_val = 1 - norm.cdf(z_stat)
                    reject = z_stat > z_crit
                    crit_str = fmt.format(z_crit)

                else:
                    z_left = -abs(norm.ppf(alpha / 2))
                    z_right = abs(norm.ppf(1 - alpha / 2))
                    p_val = 2 * (1 - norm.cdf(abs(z_stat)))
                    reject = abs(z_stat) > z_right
                    crit_str = f"{fmt.format(z_left)}, {fmt.format(z_right)}"

                themed_box(f"**Step 4:** Compare p-value = {fmt.format(p_val)} with α = {alpha}")

                decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"

                st.markdown(f"""
### **Result Summary**
- Test Statistic (z): {fmt.format(z_stat)}
- Critical Value(s): {crit_str}
- P-value: {fmt.format(p_val)}
- Decision: **{decision}**
""")

            # ------------------------------------------------------
            # SMALL SAMPLE: BINOMIAL EXACT TEST
            # ------------------------------------------------------
            else:
                st.markdown("### 🎯 Exact Binomial Test")

                themed_box("**Step 1:** Compute exact p-value using Binomial CDF.")

                if tails == "left":
                    p_val = binom.cdf(x - 1, n, p0)
                    st.latex(fr"P(X < x) = {fmt.format(p_val)}")
                    reject = p_val < alpha

                elif tails == "right":
                    p_val = 1 - binom.cdf(x - 1, n, p0)
                    st.latex(fr"P(X > x) = {fmt.format(p_val)}")
                    reject = p_val < alpha

                else:
                    left = binom.cdf(x - 1, n, p0)
                    right = 1 - binom.cdf(x - 1, n, p0)
                    p_val = float(min(1, 2 * min(left, right)))
                    st.latex(fr"""
                    \begin{{aligned}}
                    P_\text{{left}} &= {fmt.format(left)} \\
                    P_\text{{right}} &= {fmt.format(right)} \\
                    p\text{{-value}} &= {fmt.format(p_val)}
                    \end{{aligned}}
                    """)
                    reject = p_val < alpha

                themed_box(f"**Step 2:** Compare p-value = {fmt.format(p_val)} with α = {alpha}")

                decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"

                st.markdown(f"""
### **Result Summary**
- P-value: {fmt.format(p_val)}
- Decision: **{decision}**
""")

    # ==========================================================
    # T-TESTS
    # ==========================================================
    elif test_choice in [
        "t-test for population mean (summary stats)",
        "t-test for population mean (raw data)"
    ]:

        if test_choice == "t-test for population mean (summary stats)":
            mean = st.number_input("Sample mean (x̄)", format="%.6f")
            sd = st.number_input("Sample standard deviation (s)", format="%.6f")
            n = st.number_input("Sample size (n)", min_value=2, step=1)

        else:
            st.markdown("### 📊 Provide Sample Data")
            uploaded_data = load_uploaded_data()
            raw_input = st.text_area("Or enter comma-separated values:")

        mu0 = st.number_input("Null hypothesis mean (μ₀)", format="%.6f")

        if st.button("👨‍💻 Calculate"):
            # Load raw data if needed
            if test_choice == "t-test for population mean (raw data)":
                if uploaded_data is not None:
                    data = uploaded_data
                elif raw_input:
                    data = np.array([float(i.strip()) for i in raw_input.split(",")])
                else:
                    st.warning("⚠ Please provide sample data.")
                    return
                mean = np.mean(data)
                sd = np.std(data, ddof=1)
                n = len(data)

            df = n - 1
            se = sd / math.sqrt(n)
            t_stat = (mean - mu0) / se

            st.markdown("### 📘 Step-by-Step Solution")
            themed_box("**Step 1:** Compute the t-statistic.")
            st.latex(r"t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}")
            st.latex(fr"t = {fmt.format(t_stat)}")

            themed_box("**Step 2:** Compute p-value and compare with α")

            if tails == "left":
                t_crit = t.ppf(alpha, df)
                p_val = t.cdf(t_stat, df)
                reject = t_stat < t_crit
                crit_str = fmt.format(t_crit)

            elif tails == "right":
                t_crit = t.ppf(1 - alpha, df)
                p_val = 1 - t.cdf(t_stat, df)
                reject = t_stat > t_crit
                crit_str = fmt.format(t_crit)

            else:
                t_left = t.ppf(alpha / 2, df)
                t_right = t.ppf(1 - alpha / 2, df)
                p_val = 2 * (1 - t.cdf(abs(t_stat), df))
                reject = abs(t_stat) > abs(t_right)
                crit_str = f"{fmt.format(t_left)}, {fmt.format(t_right)}"

            themed_box(f"**Step 3:** Compare p-value = {fmt.format(p_val)} with α = {alpha}")

            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"

            st.markdown(f"""
### **Result Summary**
- df = {df}
- t-statistic = {fmt.format(t_stat)}
- Critical Value(s): {crit_str}
- P-value = {fmt.format(p_val)}
- Decision: **{decision}**
""")

    # ==========================================================
    # CHI-SQUARED TESTS
    # ==========================================================
    else:

        if test_choice == "Chi-squared test for std dev (summary stats)":
            sd = st.number_input("Sample standard deviation (s)", format="%.6f")
            n = st.number_input("Sample size (n)", min_value=2, step=1)

        else:
            st.markdown("### 📊 Provide Sample Data")
            uploaded_data = load_uploaded_data()
            raw_input = st.text_area("Or enter comma-separated values:")

        sigma0 = st.number_input("Population standard deviation (σ₀)", format="%.6f")

        if st.button("👨‍💻 Calculate"):
            # Load raw data if needed
            if test_choice == "Chi-squared test for std dev (raw data)":
                if uploaded_data is not None:
                    data = uploaded_data
                elif raw_input:
                    data = np.array([float(i.strip()) for i in raw_input.split(",")])
                else:
                    st.warning("⚠ Please provide sample data.")
                    return
                sd = np.std(data, ddof=1)
                n = len(data)

            df = n - 1
            chi2_stat = (df * sd**2) / sigma0**2

            st.markdown("### 📘 Step-by-Step Solution")
            themed_box("**Step 1:** Compute χ² statistic.")
            st.latex(r"\chi^2 = \frac{(n-1)s^2}{\sigma_0^2}")
            st.latex(fr"\chi^2 = {fmt.format(chi2_stat)}")

            themed_box("**Step 2:** Compute p-value and compare with α")

            if tails == "left":
                chi_crit = chi2.ppf(alpha, df)
                p_val = chi2.cdf(chi2_stat, df)
                reject = chi2_stat < chi_crit
                crit_str = fmt.format(chi_crit)

            elif tails == "right":
                chi_crit = chi2.ppf(1 - alpha, df)
                p_val = 1 - chi2.cdf(chi2_stat, df)
                reject = chi2_stat > chi_crit
                crit_str = fmt.format(chi_crit)

            else:
                left = chi2.ppf(alpha / 2, df)
                right = chi2.ppf(1 - alpha / 2, df)
                p_val = 2 * min(chi2.cdf(chi2_stat, df), 1 - chi2.cdf(chi2_stat, df))
                reject = chi2_stat < left or chi2_stat > right
                crit_str = f"{fmt.format(left)}, {fmt.format(right)}"

            themed_box(f"**Step 3:** p-value = {fmt.format(p_val)} vs α = {alpha}")

            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"

            st.markdown(f"""
### **Result Summary**
- df = {df}
- χ² statistic = {fmt.format(chi2_stat)}
- Critical Value(s): {crit_str}
- P-value = {fmt.format(p_val)}
- Decision: **{decision}**
""")


# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":
    run_hypothesis_tool()
