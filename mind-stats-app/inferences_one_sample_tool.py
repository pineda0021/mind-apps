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
# Helper Functions
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


def step_box(text):
    st.markdown(
        f"""
        <div style="background-color:#f0f6ff;padding:10px;border-radius:10px;
        border-left:5px solid #007acc;margin-bottom:10px;">
        <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# Main App
# ==========================================================
def run_hypothesis_tool():
    st.header("🔎 Inferences on One Sample")

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

    alpha = st.number_input("Significance level (α)", value=0.05, min_value=0.001, max_value=0.5, step=0.01)
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
            step_box("**Step 1:** Compute the sample proportion")
            st.latex(r"\hat{p} = \frac{x}{n}")
            st.latex(fr"\hat{{p}} = \frac{{{x}}}{{{n}}} = {p_hat:.4f}")

            # Large Sample (Z-test)
            if test_choice == "Proportion test (large sample)":
                st.markdown("### 🧮 Large Sample Z-Test")
                st.latex(r"z = \frac{\hat{p} - p_0}{\sqrt{p_0(1 - p_0)/n}}")

                step_box("**Step 2:** Compute the Standard Error and test statistic.")
                se = math.sqrt(p0 * (1 - p0) / n)
                z_stat = (p_hat - p0) / se
                st.latex(fr"\text{{SE}} = \sqrt{{{p0:.4f}(1-{p0:.4f})/{n}}} = {se:.6f}")
                st.latex(fr"z = \frac{{{p_hat:.4f}-{p0:.4f}}}{{{se:.6f}}} = {z_stat:.4f}")

                step_box("**Step 3:** Determine critical values and compute p-value.")
                if tails == "left":
                    z_crit = -abs(norm.ppf(alpha))
                    p_val = norm.cdf(z_stat)
                    reject = z_stat < z_crit
                    crit_str = f"{z_crit:.4f}"
                elif tails == "right":
                    z_crit = abs(norm.ppf(1 - alpha))
                    p_val = 1 - norm.cdf(z_stat)
                    reject = z_stat > z_crit
                    crit_str = f"{z_crit:.4f}"
                else:
                    z_crit_left = -abs(norm.ppf(alpha / 2))
                    z_crit_right = abs(norm.ppf(1 - alpha / 2))
                    p_val = 2 * (1 - norm.cdf(abs(z_stat)))
                    reject = abs(z_stat) > z_crit_right
                    crit_str = f"{z_crit_left:.4f}, {z_crit_right:.4f}"

                step_box(f"**Step 4:** Compare p-value = {p_val:.4f} with α = {alpha:.2f}")

                decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"
                st.markdown(f"""
**Result Summary**

- Test Statistic (z): {z_stat:.4f}  
- Critical Value(s): {crit_str}  
- P-value: {p_val:.4f}  
- Decision: **{decision}**

**Interpretation:**  
If p-value < α → Reject H₀; otherwise, fail to reject H₀.
""")

            # Small Sample (Binomial)
            else:
                st.markdown("### 🎯 Small Sample Binomial Test (Exact)")
                st.latex(r"""
                \textbf{Left-tail: }\; P(X < x) = \mathrm{BinomCDF}(x - 1, n, p_0)
                """)
                st.latex(r"""
                \textbf{Right-tail: }\; P(X > x) = 1 - \mathrm{BinomCDF}(x - 1, n, p_0)
                """)
                st.latex(r"""
                \textbf{Two-tail: }\; p\text{-value} = 2 \times \min\!\Big(\mathrm{BinomCDF}(x - 1, n, p_0),\; 1 - \mathrm{BinomCDF}(x - 1, n, p_0)\Big)
                """)

                step_box("**Step 1:** Identify the tail based on H₁.")
                if tails == "left":
                    p_val = binom.cdf(x - 1, n, p0)
                    st.latex(fr"P(X < x) = \mathrm{{BinomCDF}}({x-1}, {n}, {p0:.4f}) = {p_val:.4f}")
                elif tails == "right":
                    p_val = 1 - binom.cdf(x - 1, n, p0)
                    st.latex(fr"P(X > x) = 1 - \mathrm{{BinomCDF}}({x-1}, {n}, {p0:.4f}) = {p_val:.4f}")
                else:
                    left = binom.cdf(x - 1, n, p0)
                    right = 1 - binom.cdf(x - 1, n, p0)
                    p_val = 2 * min(left, right)
                    p_val = float(min(1.0, p_val))
                    st.latex(fr"""
                    \begin{{aligned}}
                    P_\text{{left}} &= \mathrm{{BinomCDF}}({x-1}, {n}, {p0:.4f}) = {left:.4f} \\
                    P_\text{{right}} &= 1 - \mathrm{{BinomCDF}}({x-1}, {n}, {p0:.4f}) = {right:.4f} \\
                    p\text{{-value}} &= 2 \times \min(P_\text{{left}}, P_\text{{right}}) = {p_val:.4f}
                    \end{{aligned}}
                    """)

                step_box(f"**Step 2:** Compare p-value = {p_val:.4f} with α = {alpha:.2f}")

                reject = p_val < alpha
                decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"

                st.markdown(f"""
**Result Summary**

- P-value: {p_val:.4f}  
- α = {alpha:.2f}  
- Decision: **{decision}**

**Interpretation:**  
If p-value < α → Reject H₀; otherwise, fail to reject H₀.
""")

    # ==========================================================
    # T-TESTS
    # ==========================================================
    elif test_choice in ["t-test for population mean (summary stats)", "t-test for population mean (raw data)"]:
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
            if test_choice == "t-test for population mean (raw data)":
                if uploaded_data is not None:
                    data = uploaded_data
                elif raw_input:
                    data = np.array([float(i.strip()) for i in raw_input.split(",")])
                else:
                    st.warning("⚠️ Please upload or enter your sample data.")
                    return
                mean = np.mean(data)
                sd = np.std(data, ddof=1)
                n = len(data)

            df = n - 1
            se = sd / math.sqrt(n)
            t_stat = (mean - mu0) / se

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Compute the test statistic.")
            st.latex(r"t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}")
            st.latex(fr"t = \frac{{{mean:.4f} - {mu0:.4f}}}{{{sd:.4f}/\sqrt{{{n}}}}} = {t_stat:.4f}")

            step_box("**Step 2:** Determine critical values and compute p-value.")
            if tails == "left":
                t_crit = -abs(t.ppf(alpha, df))
                p_val = t.cdf(t_stat, df)
                reject = t_stat < t_crit
                crit_str = f"{t_crit:.4f}"
            elif tails == "right":
                t_crit = abs(t.ppf(1 - alpha, df))
                p_val = 1 - t.cdf(t_stat, df)
                reject = t_stat > t_crit
                crit_str = f"{t_crit:.4f}"
            else:
                t_crit_left = -abs(t.ppf(alpha / 2, df))
                t_crit_right = abs(t.ppf(1 - alpha / 2, df))
                p_val = 2 * (1 - t.cdf(abs(t_stat), df))
                reject = abs(t_stat) > t_crit_right
                crit_str = f"{t_crit_left:.4f}, {t_crit_right:.4f}"

            step_box(f"**Step 3:** Compare p-value = {p_val:.4f} with α = {alpha:.2f}")

            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"
            st.markdown(f"""
**Result Summary**

- Degrees of Freedom: {df}  
- Test Statistic (t): {t_stat:.4f}  
- Critical Value(s): {crit_str}  
- P-value: {p_val:.4f}  
- Decision: **{decision}**

**Interpretation:**  
If p-value < α → Reject H₀; otherwise, fail to reject H₀.
""")

    # ==========================================================
    # CHI-SQUARED TESTS
    # ==========================================================
    elif test_choice in ["Chi-squared test for std dev (summary stats)", "Chi-squared test for std dev (raw data)"]:
        if test_choice == "Chi-squared test for std dev (summary stats)":
            sd = st.number_input("Sample standard deviation (s)", format="%.6f")
            n = st.number_input("Sample size (n)", min_value=2, step=1)
        else:
            st.markdown("### 📊 Provide Sample Data")
            uploaded_data = load_uploaded_data()
            raw_input = st.text_area("Or enter comma-separated values:")

        sigma0 = st.number_input("Population standard deviation (σ₀)", format="%.6f")

        if st.button("👨‍💻 Calculate"):
            if test_choice == "Chi-squared test for std dev (raw data)":
                if uploaded_data is not None:
                    data = uploaded_data
                elif raw_input:
                    data = np.array([float(i.strip()) for i in raw_input.split(",")])
                else:
                    st.warning("⚠️ Please upload or enter your sample data.")
                    return
                sd = np.std(data, ddof=1)
                n = len(data)

            df = n - 1
            chi2_stat = (df * sd**2) / sigma0**2

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Compute the test statistic.")
            st.latex(r"\chi^2 = \frac{(n - 1)s^2}{\sigma_0^2}")
            st.latex(fr"\chi^2 = \frac{{({df})({sd:.4f})^2}}{{({sigma0:.4f})^2}} = {chi2_stat:.4f}")

            step_box("**Step 2:** Determine critical values and compute p-value.")
            if tails == "left":
                chi2_crit = chi2.ppf(alpha, df)
                p_val = chi2.cdf(chi2_stat, df)
                reject = chi2_stat < chi2_crit
                crit_str = f"{chi2_crit:.4f}"
            elif tails == "right":
                chi2_crit = chi2.ppf(1 - alpha, df)
                p_val = 1 - chi2.cdf(chi2_stat, df)
                reject = chi2_stat > chi2_crit
                crit_str = f"{chi2_crit:.4f}"
            else:
                chi2_crit_left = chi2.ppf(alpha / 2, df)
                chi2_crit_right = chi2.ppf(1 - alpha / 2, df)
                p_val = 2 * min(chi2.cdf(chi2_stat, df), 1 - chi2.cdf(chi2_stat, df))
                reject = chi2_stat < chi2_crit_left or chi2_stat > chi2_crit_right
                crit_str = f"{chi2_crit_left:.4f}, {chi2_crit_right:.4f}"

            step_box(f"**Step 3:** Compare p-value = {p_val:.4f} with α = {alpha:.2f}")

            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"
            st.markdown(f"""
**Result Summary**

- Degrees of Freedom: {df}  
- Test Statistic (χ²): {chi2_stat:.4f}  
- Critical Value(s): {crit_str}  
- P-value: {p_val:.4f}  
- Decision: **{decision}**

**Interpretation:**  
If p-value < α → Reject H₀; otherwise, fail to reject H₀.
""")

# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":
    run_hypothesis_tool()

