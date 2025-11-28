# ==========================================================
# two_sample_tool.py 
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

# ==========================================================
# Helper: Step Box (simple, theme-safe)
# ==========================================================
def step_box(text):
    st.markdown(f"**{text}**")

# ==========================================================
# Tail Metric Utilities
# ==========================================================
def z_tail_metrics(z, alpha, tail):
    if tail == "left":
        crit = stats.norm.ppf(alpha)
        p = stats.norm.cdf(z)
        reject = z < crit
        crit_str = f"{crit:.4f}"
    elif tail == "right":
        crit = stats.norm.ppf(1 - alpha)
        p = 1 - stats.norm.cdf(z)
        reject = z > crit
        crit_str = f"{crit:.4f}"
    else:
        crit = stats.norm.ppf(1 - alpha/2)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        reject = abs(z) > crit
        crit_str = f"±{crit:.4f}"
    return p, reject, crit_str

def t_tail_metrics(tval, df, alpha, tail):
    if tail == "left":
        crit = stats.t.ppf(alpha, df)
        p = stats.t.cdf(tval, df)
        reject = tval < crit
        crit_str = f"{crit:.4f}"
    elif tail == "right":
        crit = stats.t.ppf(1 - alpha, df)
        p = 1 - stats.t.cdf(tval, df)
        reject = tval > crit
        crit_str = f"{crit:.4f}"
    else:
        crit = stats.t.ppf(1 - alpha/2, df)
        p = 2 * (1 - stats.t.cdf(abs(tval), df))
        reject = abs(tval) > crit
        crit_str = f"±{crit:.4f}"
    return p, reject, crit_str

def f_tail_metrics(F, df1, df2, alpha, tail):
    if tail == "left":
        crit = stats.f.ppf(alpha, df1, df2)
        p = stats.f.cdf(F, df1, df2)
        reject = F < crit
        crit_str = f"{crit:.4f}"
    elif tail == "right":
        crit = stats.f.ppf(1 - alpha, df1, df2)
        p = 1 - stats.f.cdf(F, df1, df2)
        reject = F > crit
        crit_str = f"{crit:.4f}"
    else:
        crit_low = stats.f.ppf(alpha/2, df1, df2)
        crit_high = stats.f.ppf(1 - alpha/2, df1, df2)
        p = 2 * min(stats.f.cdf(F, df1, df2), 1 - stats.f.cdf(F, df1, df2))
        p = float(min(1.0, p))
        reject = (F < crit_low) or (F > crit_high)
        crit_str = f"({crit_low:.4f}, {crit_high:.4f})"
    return p, reject, crit_str

# ==========================================================
# Main App
# ==========================================================
def run_two_sample_tool():
    st.header("👨🏻‍🔬 Two-Sample Inference Tool (MIND)")

    # Decimal places everywhere
    dec = st.number_input("Decimal places for output", min_value=0, max_value=10, value=4)

    test_choice = st.selectbox(
        "Choose a Two-Sample Test:",
        [
            "Two-Proportion Z-Test",
            "Paired t-Test using Data",
            "Paired t-Test using Summary Statistics",
            "Independent t-Test using Data (Welch)",
            "Independent t-Test using Summary Statistics (Welch)",
            "F-Test for Standard Deviations using Data",
            "F-Test for Standard Deviations using Summary Statistics"
        ],
        index=None,
        placeholder="Select a test to begin..."
    )

    if not test_choice:
        st.info("👆 Please select a two-sample test to begin.")
        return

    alpha = st.number_input("Significance level (α)", value=0.05)
    tails = st.selectbox("Tail type:", ["two", "left", "right"])
    st.markdown("---")

    # ==========================================================
    # TWO-PROPORTION Z-TEST
    # ==========================================================
    if test_choice == "Two-Proportion Z-Test":
        x1 = st.number_input("Successes in Sample 1 (x₁)", min_value=0, step=1)
        n1 = st.number_input("Sample size 1 (n₁)", min_value=1, step=1)
        x2 = st.number_input("Successes in Sample 2 (x₂)", min_value=0, step=1)
        n2 = st.number_input("Sample size 2 (n₂)", min_value=1, step=1)

        if st.button("Calculate"):
            p1 = x1 / n1
            p2 = x2 / n2
            p_pool = (x1 + x2) / (n1 + n2)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            z = (p1 - p2) / se

            step_box("Step 1: Compute sample and pooled proportions.")
            step_box("Step 2: Compute standard error and z-statistic.")
            step_box("Step 3: Compute tail-specific p-value.")

            p_val, reject, crit = z_tail_metrics(z, alpha, tails)

            decision_symbol = "✔" if reject else "✖"
            decision_text = "Reject H₀" if reject else "Fail to reject H₀"

            st.markdown(f"""
### 📊 Result Summary

- Test Statistic (z): {z:.{dec}f}  
- Critical Value(s): {crit}  
- P-value: {p_val:.{dec}f}  
- Decision: **{decision_symbol} {decision_text}**

### 📘 Interpretation
If p-value < α → Reject H₀; otherwise → Fail to reject H₀.
""")

    # ==========================================================
    # PAIRED t-TEST USING DATA
    # ==========================================================
    elif test_choice == "Paired t-Test using Data":
        up = st.file_uploader("Upload CSV with columns Sample1 and Sample2", type="csv")
        s1 = st.text_area("Sample 1 (comma-separated)")
        s2 = st.text_area("Sample 2 (comma-separated)")

        if st.button("Calculate"):
            if up:
                df = pd.read_csv(up)
                if {"Sample1", "Sample2"} - set(df.columns):
                    st.error("CSV must include Sample1 and Sample2.")
                    return
                x1 = df["Sample1"].to_numpy(float)
                x2 = df["Sample2"].to_numpy(float)
            else:
                try:
                    x1 = np.array([float(i) for i in s1.split(",") if i.strip()])
                    x2 = np.array([float(i) for i in s2.split(",") if i.strip()])
                except:
                    st.error("Invalid data.")
                    return

            if len(x1) != len(x2):
                st.error("Samples must have same length.")
                return

            d = x1 - x2
            n = len(d)
            mean_d = np.mean(d)
            sd_d = np.std(d, ddof=1)
            se = sd_d / np.sqrt(n)
            tval = mean_d / se
            dfree = n - 1

            # Show difference table
            st.markdown("### Differences Table")
            diff_df = pd.DataFrame({"Sample1": x1, "Sample2": x2, "d = x1 - x2": d})
            st.dataframe(diff_df)

            step_box("Step 1: Compute differences and summary statistics.")
            step_box("Step 2: Compute test statistic.")
            step_box("Step 3: Compute tail-specific p-value.")

            p_val, reject, crit = t_tail_metrics(tval, dfree, alpha, tails)

            decision_symbol = "✔" if reject else "✖"
            decision_text = "Reject H₀" if reject else "Fail to reject H₀"

            st.markdown(f"""
### 📊 Result Summary

- Mean Difference (d̄): {mean_d:.{dec}f}  
- SD of Differences (s_d): {sd_d:.{dec}f}  
- Test Statistic (t): {tval:.{dec}f}  
- Degrees of Freedom: {dfree}  
- Critical Value(s): {crit}  
- P-value: {p_val:.{dec}f}  
- Decision: **{decision_symbol} {decision_text}**

### 📘 Interpretation
If p-value < α → Reject H₀; otherwise → Fail to reject H₀.
""")

    # ==========================================================
    # PAIRED SUMMARY
    # ==========================================================
    elif test_choice == "Paired t-Test using Summary Statistics":
        mean_d = st.number_input("Mean of differences (d̄)", value=0.0)
        sd_d = st.number_input("Std Dev of differences (s_d)", value=1.0)
        n = st.number_input("Sample size (n)", min_value=2, step=1)

        if st.button("Calculate"):
            dfree = n - 1
            se = sd_d / np.sqrt(n)
            tval = mean_d / se

            p_val, reject, crit = t_tail_metrics(tval, dfree, alpha, tails)
            decision_symbol = "✔" if reject else "✖"
            decision_text = "Reject H₀" if reject else "Fail to reject H₀"

            st.markdown(f"""
### 📊 Result Summary

- Mean Difference (d̄): {mean_d:.{dec}f}  
- SD of Differences (s_d): {sd_d:.{dec}f}  
- Test Statistic (t): {tval:.{dec}f}  
- Degrees of Freedom: {dfree}  
- Critical Value(s): {crit}  
- P-value: {p_val:.{dec}f}  
- Decision: **{decision_symbol} {decision_text}**

### 📘 Interpretation
If p-value < α → Reject H₀; otherwise → Fail to reject H₀.
""")

    # ==========================================================
    # INDEPENDENT t-TEST USING DATA (Welch)
    # ==========================================================
    elif test_choice == "Independent t-Test using Data (Welch)":
        up = st.file_uploader("Upload CSV with Sample1, Sample2", type="csv")
        s1 = st.text_area("Sample 1 (comma-separated)")
        s2 = st.text_area("Sample 2 (comma-separated)")

        if st.button("Calculate"):
            if up:
                df = pd.read_csv(up)
                if {"Sample1", "Sample2"} - set(df.columns):
                    st.error("CSV must include Sample1 and Sample2.")
                    return
                x1 = df["Sample1"].to_numpy(float)
                x2 = df["Sample2"].to_numpy(float)
            else:
                x1 = np.array([float(i) for i in s1.split(",") if i.strip()])
                x2 = np.array([float(i) for i in s2.split(",") if i.strip()])

            n1, n2 = len(x1), len(x2)
            mean1, mean2 = np.mean(x1), np.mean(x2)
            s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)

            se = np.sqrt(s1**2/n1 + s2**2/n2)
            tval = (mean1 - mean2) / se

            dfree = (se**4) / ((s1**4)/(n1**2*(n1-1)) + (s2**4)/(n2**2*(n2-1)))

            p_val, reject, crit = t_tail_metrics(tval, dfree, alpha, tails)
            decision_symbol = "✔" if reject else "✖"
            decision_text = "Reject H₀" if reject else "Fail to reject H₀"

            st.markdown(f"""
### 📊 Result Summary

- Mean₁ = {mean1:.{dec}f}, Mean₂ = {mean2:.{dec}f}  
- SD₁ = {s1:.{dec}f}, SD₂ = {s2:.{dec}f}  
- Test Statistic (t): {tval:.{dec}f}  
- Degrees of Freedom (Welch): {dfree:.2f}  
- Critical Value(s): {crit}  
- P-value: {p_val:.{dec}f}  
- Decision: **{decision_symbol} {decision_text}**

### 📘 Interpretation
If p-value < α → Reject H₀; otherwise → Fail to reject H₀.
""")

    # ==========================================================
    # INDEPENDENT SUMMARY (Welch)
    # ==========================================================
    elif test_choice == "Independent t-Test using Summary Statistics (Welch)":
        mean1 = st.number_input("Mean₁", value=0.0)
        s1 = st.number_input("SD₁", value=1.0)
        n1 = st.number_input("n₁", min_value=2)
        mean2 = st.number_input("Mean₂", value=0.0)
        s2 = st.number_input("SD₂", value=1.0)
        n2 = st.number_input("n₂", min_value=2)

        if st.button("Calculate"):
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            tval = (mean1 - mean2) / se

            dfree = (se**4) / ((s1**4)/(n1**2*(n1-1)) + (s2**4)/(n2**2*(n2-1)))

            p_val, reject, crit = t_tail_metrics(tval, dfree, alpha, tails)

            decision_symbol = "✔" if reject else "✖"
            decision_text = "Reject H₀" if reject else "Fail to reject H₀"

            st.markdown(f"""
### 📊 Result Summary

- Mean₁ = {mean1:.{dec}f}, Mean₂ = {mean2:.{dec}f}  
- SD₁ = {s1:.{dec}f}, SD₂ = {s2:.{dec}f}  
- Test Statistic (t): {tval:.{dec}f}  
- Degrees of Freedom (Welch): {dfree:.2f}  
- Critical Value(s): {crit}  
- P-value: {p_val:.{dec}f}  
- Decision: **{decision_symbol} {decision_text}**

### 📘 Interpretation
If p-value < α → Reject H₀; otherwise → Fail to reject H₀.
""")

    # ==========================================================
    # F-TEST USING DATA
    # ==========================================================
    elif test_choice == "F-Test for Standard Deviations using Data":
        up = st.file_uploader("Upload CSV with Sample1, Sample2", type="csv")
        s1 = st.text_area("Sample 1 (comma-separated)")
        s2 = st.text_area("Sample 2 (comma-separated)")

        if st.button("Calculate"):
            if up:
                df = pd.read_csv(up)
                x1 = df["Sample1"].to_numpy(float)
                x2 = df["Sample2"].to_numpy(float)
            else:
                x1 = np.array([float(i) for i in s1.split(",") if i.strip()])
                x2 = np.array([float(i) for i in s2.split(",") if i.strip()])

            n1, n2 = len(x1), len(x2)
            s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)

            F = (s1**2) / (s2**2)
            df1, df2 = n1 - 1, n2 - 1

            p_val, reject, crit_str = f_tail_metrics(F, df1, df2, alpha, tails)

            decision_symbol = "✔" if reject else "✖"
            decision_text = "Reject H₀" if reject else "Fail to reject H₀"

            st.markdown(f"""
### 📊 Result Summary

- F Statistic = {F:.{dec}f}  
- Degrees of Freedom: df₁={df1}, df₂={df2}  
- Critical Value(s): {crit_str}  
- P-value: {p_val:.{dec}f}  
- Decision: **{decision_symbol} {decision_text}**

### 📘 Interpretation
If p-value < α → Reject H₀; otherwise → Fail to reject H₀.
""")

    # ==========================================================
    # F-TEST SUMMARY
    # ==========================================================
    elif test_choice == "F-Test for Standard Deviations using Summary Statistics":
        n1 = st.number_input("n₁", min_value=2)
        s1 = st.number_input("SD₁", value=1.0)
        n2 = st.number_input("n₂", min_value=2)
        s2 = st.number_input("SD₂", value=1.0)

        if st.button("Calculate"):
            F = (s1**2) / (s2**2)
            df1, df2 = n1 - 1, n2 - 1

            p_val, reject, crit_str = f_tail_metrics(F, df1, df2, alpha, tails)

            decision_symbol = "✔" if reject else "✖"
            decision_text = "Reject H₀" if reject else "Fail to reject H₀"

            st.markdown(f"""
### 📊 Result Summary

- F Statistic = {F:.{dec}f}  
- Degrees of Freedom: df₁={df1}, df₂={df2}  
- Critical Value(s): {crit_str}  
- P-value: {p_val:.{dec}f}  
- Decision: **{decision_symbol} {decision_text}**

### 📘 Interpretation
If p-value < α → Reject H₀; otherwise → Fail to reject H₀.
""")

# ==========================================================
# Run app
# ==========================================================
if __name__ == "__main__":
    run_two_sample_tool()


