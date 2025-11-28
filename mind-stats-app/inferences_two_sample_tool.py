# ==========================================================
# two_sample_inference_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import math
from scipy.stats import t, norm

# ==========================================================
# Universal Dark/Light Mode Safe Step Box
# ==========================================================
def step_box(text):
    st.markdown(
        f"""
        <div style="
            padding:12px;
            border-radius:10px;
            border-left:5px solid #4da3ff;
            background-color:rgba(0,0,0,0);
            margin-top:10px;
            margin-bottom:10px;">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================================
# Load CSV or Excel
# ==========================================================
def load_uploaded_data():
    uploaded = st.file_uploader("📂 Upload CSV or Excel file", type=["csv","xlsx"])
    if not uploaded:
        return None

    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        # Return numeric columns only
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric_cols) == 0:
            st.error("No numeric columns found.")
            return None
        return df[numeric_cols]
    except Exception as e:
        st.error(f"File read error: {e}")
        return None

# ==========================================================
# Main App
# ==========================================================
def run_two_sample_tool():
    st.header("📊 Two-Sample Inference Tools")

    decimals = st.selectbox(
        "Decimal places for outputs:",
        [2, 3, 4, 5, 6],
        index=2
    )

    test_type = st.selectbox(
        "Select Test Type:",
        [
            "Two-sample t-test (independent, equal variances)",
            "Two-sample t-test (independent, unequal variances — Welch)",
            "Paired t-test",
            "Two-sample proportion test"
        ]
    )

    alpha = st.number_input("Significance Level (α)", value=0.05, min_value=0.0001, max_value=0.5)

    tails = st.selectbox("Tail type (H₁):", ["two", "left", "right"])

    st.markdown("---")

    # ==========================================================
    # INDEPENDENT TWO-SAMPLE MEANS — EQUAL VARIANCES
    # ==========================================================
    if test_type == "Two-sample t-test (independent, equal variances)":
        st.subheader("🎯 Two-Sample t-Test (Equal Variances)")

        xbar1 = st.number_input("Sample mean x̄₁")
        s1 = st.number_input("Sample SD s₁")
        n1 = st.number_input("Sample size n₁", min_value=2, step=1)

        xbar2 = st.number_input("Sample mean x̄₂")
        s2 = st.number_input("Sample SD s₂")
        n2 = st.number_input("Sample size n₂", min_value=2, step=1)

        mu0 = st.number_input("Null difference (μ₁ − μ₂)", value=0.0)

        if st.button("Calculate"):
            df = n1 + n2 - 2
            sp2 = (((n1 - 1) * s1**2) + ((n2 - 1) * s2**2)) / df
            sp = math.sqrt(sp2)

            se = sp * math.sqrt((1/n1) + (1/n2))

            t_stat = (xbar1 - xbar2 - mu0) / se

            # Critical values & p-values
            if tails == "left":
                crit = t.ppf(alpha, df)
                p_val = t.cdf(t_stat, df)
                reject = t_stat < crit
            elif tails == "right":
                crit = t.ppf(1 - alpha, df)
                p_val = 1 - t.cdf(t_stat, df)
                reject = t_stat > crit
            else:
                crit_left = t.ppf(alpha/2, df)
                crit_right = t.ppf(1 - alpha/2, df)
                p_val = 2 * (1 - t.cdf(abs(t_stat), df))
                reject = (t_stat < crit_left) or (t_stat > crit_right)

            st.markdown("## 📘 Step-by-Step")

            step_box("### Step 1 — Pooled Standard Deviation")
            st.latex(r"""
            s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}
            """)
            st.write(f"**sp = {sp:.{decimals}f}**")

            step_box("### Step 2 — Standard Error")
            st.latex(r"SE = s_p\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}")
            st.write(f"**SE = {se:.{decimals}f}**")

            step_box("### Step 3 — Test Statistic")
            st.latex(r"t = \frac{(x̄_1 - x̄_2) - \mu_0}{SE}")
            st.write(f"**t = {t_stat:.{decimals}f}**")

            step_box("### Step 4 — Decision")
            if tails == "two":
                st.write(f"Critical t-values: {crit_left:.{decimals}f}, {crit_right:.{decimals}f}")
            else:
                st.write(f"Critical value: {crit:.{decimals}f}")

            st.write(f"p-value = **{p_val:.{decimals}f}**")

            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"
            st.subheader(decision)

    # ==========================================================
    # WELCH T-TEST (UNEQUAL VARIANCES)
    # ==========================================================
    if test_type == "Two-sample t-test (independent, unequal variances — Welch)":
        st.subheader("🎯 Welch's t-Test (Unequal Variances)")

        xbar1 = st.number_input("Sample mean x̄₁")
        s1 = st.number_input("Sample SD s₁")
        n1 = st.number_input("Sample size n₁", min_value=2, step=1)

        xbar2 = st.number_input("Sample mean x̄₂")
        s2 = st.number_input("Sample SD s₂")
        n2 = st.number_input("Sample size n₂", min_value=2, step=1)

        mu0 = st.number_input("Null difference (μ₁ − μ₂)", value=0.0)

        if st.button("Calculate"):
            se = math.sqrt((s1**2/n1) + (s2**2/n2))
            t_stat = (xbar1 - xbar2 - mu0) / se

            # Welch df
            df = ( (s1**2/n1 + s2**2/n2)**2 ) / (
                ( (s1**2/n1)**2 / (n1 - 1) ) +
                ( (s2**2/n2)**2 / (n2 - 1) )
            )

            # Criticals & P-values
            if tails == "left":
                crit = t.ppf(alpha, df)
                p_val = t.cdf(t_stat, df)
                reject = t_stat < crit
            elif tails == "right":
                crit = t.ppf(1 - alpha, df)
                p_val = 1 - t.cdf(t_stat, df)
                reject = t_stat > crit
            else:
                crit_left = t.ppf(alpha/2, df)
                crit_right = t.ppf(1 - alpha/2, df)
                p_val = 2*(1 - t.cdf(abs(t_stat), df))
                reject = (t_stat < crit_left or t_stat > crit_right)

            st.markdown("## 📘 Step-by-Step")

            step_box("### Step 1 — Standard Error")
            st.write(f"SE = **{se:.{decimals}f}**")

            step_box("### Step 2 — Test Statistic")
            st.write(f"t = **{t_stat:.{decimals}f}**")

            step_box("### Step 3 — Welch Degrees of Freedom")
            st.write(f"df = **{df:.{decimals}f}**")

            step_box("### Step 4 — Decision")
            st.write(f"p-value = **{p_val:.{decimals}f}**")
            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"
            st.subheader(decision)

    # ==========================================================
    # PAIRED T-TEST
    # ==========================================================
    if test_type == "Paired t-test":
        st.subheader("🔗 Paired t-Test")

        st.write("Enter paired data manually (comma-separated) OR upload a file with two numeric columns.")

        uploaded = load_uploaded_data()

        col1_text = st.text_input("Before values (comma-separated)")
        col2_text = st.text_input("After values (comma-separated)")

        if uploaded is not None:
            cols = list(uploaded.columns)
            col1 = st.selectbox("Column for Before", cols)
            col2 = st.selectbox("Column for After", cols)
            before = uploaded[col1].to_numpy()
            after = uploaded[col2].to_numpy()
        else:
            before = np.array([float(x.strip()) for x in col1_text.split(",")]) if col1_text else None
            after = np.array([float(x.strip()) for x in col2_text.split(",")]) if col2_text else None

        if st.button("Calculate"):
            if before is None or after is None:
                st.error("Please enter or upload BOTH sets of data.")
                return

            if len(before) != len(after):
                st.error("Before and After must have equal length.")
                return

            d = before - after
            dbar = np.mean(d)
            sd = np.std(d, ddof=1)
            n = len(d)
            se = sd / math.sqrt(n)
            t_stat = dbar / se
            df = n - 1

            # Tail logic
            if tails == "left":
                crit = t.ppf(alpha, df)
                p_val = t.cdf(t_stat, df)
                reject = t_stat < crit
            elif tails == "right":
                crit = t.ppf(1 - alpha, df)
                p_val = 1 - t.cdf(t_stat, df)
                reject = t_stat > crit
            else:
                crit_left = t.ppf(alpha/2, df)
                crit_right = t.ppf(1 - alpha/2, df)
                p_val = 2*(1 - t.cdf(abs(t_stat), df))
                reject = (t_stat < crit_left or t_stat > crit_right)

            st.markdown("## 📘 Step-by-Step")

            step_box("### Step 1 — Compute Differences")
            st.write("d = Before − After")
            st.write(f"Mean difference d̄ = **{dbar:.{decimals}f}**")

            step_box("### Step 2 — Test Statistic")
            st.write(f"t = **{t_stat:.{decimals}f}**")

            step_box("### Step 3 — Decision")
            st.write(f"p-value = **{p_val:.{decimals}f}**")
            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"
            st.subheader(decision)

    # ==========================================================
    # TWO-SAMPLE PROPORTION TEST
    # ==========================================================
    if test_type == "Two-sample proportion test":
        st.subheader("📊 Two-Sample Proportion Test")

        x1 = st.number_input("Successes x₁", min_value=0, step=1)
        n1 = st.number_input("Sample size n₁", min_value=1, step=1)
        x2 = st.number_input("Successes x₂", min_value=0, step=1)
        n2 = st.number_input("Sample size n₂", min_value=1, step=1)

        if st.button("Calculate"):
            p1 = x1/n1
            p2 = x2/n2

            p_pool = (x1 + x2) / (n1 + n2)
            se = math.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
            z = (p1 - p2) / se

            if tails == "left":
                crit = norm.ppf(alpha)
                p_val = norm.cdf(z)
                reject = z < crit
            elif tails == "right":
                crit = norm.ppf(1 - alpha)
                p_val = 1 - norm.cdf(z)
                reject = z > crit
            else:
                crit_left = norm.ppf(alpha/2)
                crit_right = norm.ppf(1 - alpha/2)
                p_val = 2*(1 - norm.cdf(abs(z)))
                reject = (z < crit_left or z > crit_right)

            st.markdown("## 📘 Step-by-Step")

            step_box("### Step 1 — Sample Proportions")
            st.write(f"p̂₁ = {p1:.{decimals}f},  p̂₂ = {p2:.{decimals}f}")

            step_box("### Step 2 — Pooled Proportion")
            st.write(f"p̂_pool = {p_pool:.{decimals}f}")

            step_box("### Step 3 — Test Statistic")
            st.write(f"z = **{z:.{decimals}f}**")

            step_box("### Step 4 — Decision")
            st.write(f"p-value = **{p_val:.{decimals}f}**")
            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"
            st.subheader(decision)


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    run_two_sample_tool()
