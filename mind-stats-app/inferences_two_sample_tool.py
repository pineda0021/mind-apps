# ==========================================================
# two_sample_tool.py
# Created by Professor Edward Pineda-Castro
# MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

# ==========================================================
# UI Helper (Step Box)
# ==========================================================
def step_box(text: str):
    st.markdown(
        f"""
        <div style="background-color:#f0f6ff;padding:10px;border-radius:10px;
        border-left:5px solid #007acc;margin-bottom:12px;">
        <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================
# Decision Format (Green ✔️ / Red ✖️)
# ==========================================================
def decision_block(reject: bool):
    return (
        "<span style='color:green; font-weight:bold;'>✔️ Reject H₀</span>"
        if reject
        else "<span style='color:red; font-weight:bold;'>✖️ Do not reject H₀</span>"
    )

# ==========================================================
# Tail metric helpers
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
        reject = (F < crit_low) or (F > crit_high)
        crit_str = f"({crit_low:.4f}, {crit_high:.4f})"
    return p, reject, crit_str

# ==========================================================
# MAIN APP
# ==========================================================
def run_two_sample_tool():
    st.header("👨🏻‍🔬 Two-Sample Inference (Step-by-Step)")
    st.markdown("---")

    # Decimal output
    decimals = st.number_input(
        "Decimal places for output:",
        min_value=0,
        max_value=10,
        value=4,
        step=1
    )

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

    alpha = st.number_input("Significance level (α)", value=0.05, min_value=0.001, max_value=0.5)
    tails = st.selectbox("Tail type:", ["two", "left", "right"])
    st.markdown("---")

    # ==========================================================
    # TWO-PROPORTION Z TEST
    # ==========================================================
    if test_choice == "Two-Proportion Z-Test":
        st.subheader("📊 Enter Sample Counts")
        x1 = st.number_input("Successes in Sample 1 (x₁)", min_value=0)
        n1 = st.number_input("Sample size 1 (n₁)", min_value=1)
        x2 = st.number_input("Successes in Sample 2 (x₂)", min_value=0)
        n2 = st.number_input("Sample size 2 (n₂)", min_value=1)

        if st.button("👨‍💻 Calculate"):
            p1 = x1 / n1
            p2 = x2 / n2
            p_pool = (x1 + x2) / (n1 + n2)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            z_stat = (p1 - p2) / se

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Compute sample proportions and pooled proportion.")
            st.latex(
                r"\hat{p}_1=\frac{x_1}{n_1},\quad \hat{p}_2=\frac{x_2}{n_2},\quad \hat{p}=\frac{x_1+x_2}{n_1+n_2}"
            )
            st.write(f"p̂₁ = {p1:.{decimals}f}, p̂₂ = {p2:.{decimals}f}, pooled p̂ = {p_pool:.{decimals}f}")

            step_box("**Step 2:** Compute standard error and test statistic.")
            st.latex(
                r"z=\frac{\hat{p}_1-\hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(1/n_1+1/n_2)}}"
            )
            st.write(f"SE = {se:.{decimals}f},   z = {z_stat:.{decimals}f}")

            step_box("**Step 3:** Tail-specific p-value and critical value(s).")
            p_val, reject, crit = z_tail_metrics(z_stat, alpha, tails)

            # -------------------------
            # Minimal Result Summary
            # -------------------------
            st.markdown("### **Result Summary**")
            st.write(f"Test Statistic (z): {z_stat:.{decimals}f}")
            st.write(f"Critical Value(s): {crit}")
            st.write(f"P-value: {p_val:.{decimals}f}")
            st.markdown(decision_block(reject), unsafe_allow_html=True)

    # ==========================================================
    # PAIRED t-TEST USING DATA
    # ==========================================================
    elif test_choice == "Paired t-Test using Data":
        st.subheader("📊 Enter Paired Data")
        up = st.file_uploader("Upload CSV with columns Sample1, Sample2", type="csv")
        s1 = st.text_area("Sample 1", "10, 12, 9, 11")
        s2 = st.text_area("Sample 2", "9, 11, 8, 10")

        if st.button("👨‍💻 Calculate"):
            if up:
                df = pd.read_csv(up)
                x1 = df["Sample1"].astype(float).to_numpy()
                x2 = df["Sample2"].astype(float).to_numpy()
            else:
                x1 = np.array([float(i) for i in s1.split(",")])
                x2 = np.array([float(i) for i in s2.split(",")])

            d = x1 - x2
            n = len(d)
            mean_d = np.mean(d)
            sd_d = np.std(d, ddof=1)
            se = sd_d / np.sqrt(n)
            t_stat = mean_d / se
            dfree = n - 1

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Compute differences and summary statistics.")
            st.latex(r"d_i = x_{1i} - x_{2i}")

            # Difference table
            diff_df = pd.DataFrame({"Index": np.arange(1, n+1), "dᵢ": d})
            st.table(diff_df)

            step_box("**Step 2:** Compute test statistic.")
            st.latex(r"t=\frac{\bar d}{s_d/\sqrt{n}}")
            st.write(f"d̄ = {mean_d:.{decimals}f},  s_d = {sd_d:.{decimals}f},  t = {t_stat:.{decimals}f}")

            step_box("**Step 3:** Compute p-value and critical value(s).")
            p_val, reject, crit = t_tail_metrics(t_stat, dfree, alpha, tails)

            st.markdown("### **Result Summary**")
            st.write(f"Test Statistic (t): {t_stat:.{decimals}f} (df = {dfree})")
            st.write(f"Critical Value(s): {crit}")
            st.write(f"P-value: {p_val:.{decimals}f}")
            st.markdown(decision_block(reject), unsafe_allow_html=True)

    # ==========================================================
    # PAIRED t-TEST (SUMMARY STATS + Difference Table Illustration)
    # ==========================================================
    elif test_choice == "Paired t-Test using Summary Statistics":
        st.subheader("📋 Summary Stats for Differences")
        mean_d = st.number_input("Mean of differences (d̄)", value=0.0, format="%.6f")
        sd_d = st.number_input("Std Dev of differences (s_d)", value=1.0, format="%.6f")
        n = st.number_input("Sample size (n)", min_value=2)

        if st.button("👨‍💻 Calculate"):
            dfree = n - 1
            se = sd_d / np.sqrt(n)
            t_stat = mean_d / se

            st.markdown("### 📘 Step-by-Step Solution")

            step_box("**Step 1:** Difference table illustration (summary stats only).")
            diff_table = pd.DataFrame({
                "Index": np.arange(1, n+1),
                "dᵢ": ["(summary only)" for _ in range(n)]
            })
            st.table(diff_table)

            step_box("**Step 2:** Compute test statistic.")
            st.latex(r"t=\frac{\bar d}{s_d/\sqrt{n}}")
            st.write(f"d̄ = {mean_d:.{decimals}f},  s_d = {sd_d:.{decimals}f},  t = {t_stat:.{decimals}f}")

            step_box("**Step 3:** Compute p-value and critical value(s).")
            p_val, reject, crit = t_tail_metrics(t_stat, dfree, alpha, tails)

            st.markdown("### **Result Summary**")
            st.write(f"Test Statistic (t): {t_stat:.{decimals}f} (df = {dfree})")
            st.write(f"Critical Value(s): {crit}")
            st.write(f"P-value: {p_val:.{decimals}f}")
            st.markdown(decision_block(reject), unsafe_allow_html=True)

    # ==========================================================
    # INDEPENDENT t-TEST USING DATA (WELCH)
    # ==========================================================
    elif test_choice == "Independent t-Test using Data (Welch)":
        st.subheader("📊 Enter Independent Sample Data")
        up = st.file_uploader("Upload CSV with Sample1 and Sample2", type="csv")
        s1 = st.text_area("Sample 1", "10, 12, 14, 13")
        s2 = st.text_area("Sample 2", "9, 11, 10, 12")

        if st.button("👨‍💻 Calculate"):
            if up:
                df = pd.read_csv(up)
                x1 = df["Sample1"].astype(float).to_numpy()
                x2 = df["Sample2"].astype(float).to_numpy()
            else:
                x1 = np.array([float(i) for i in s1.split(",")])
                x2 = np.array([float(i) for i in s2.split(",")])

            n1, n2 = len(x1), len(x2)
            m1, m2 = np.mean(x1), np.mean(x2)
            s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            t_stat = (m1 - m2) / se

            df_welch = ((s1**2/n1 + s2**2/n2)**2) / (
                (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
            )

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Compute means, SDs, and standard error.")
            st.write(f"x̄₁ = {m1:.{decimals}f},  x̄₂ = {m2:.{decimals}f}")
            st.write(f"s₁ = {s1:.{decimals}f},  s₂ = {s2:.{decimals}f}")
            st.write(f"SE = {se:.{decimals}f}")

            step_box("**Step 2:** Compute test statistic and df (Welch).")
            st.latex(r"t=\frac{\bar{x}_1-\bar{x}_2}{\text{SE}}")
            st.write(f"t = {t_stat:.{decimals}f},  df ≈ {df_welch:.{decimals}f}")

            step_box("**Step 3:** Compute p-value and critical value(s).")
            p_val, reject, crit = t_tail_metrics(t_stat, df_welch, alpha, tails)

            st.markdown("### **Result Summary**")
            st.write(f"Test Statistic (t): {t_stat:.{decimals}f} (df ≈ {df_welch:.2f})")
            st.write(f"Critical Value(s): {crit}")
            st.write(f"P-value: {p_val:.{decimals}f}")
            st.markdown(decision_block(reject), unsafe_allow_html=True)

    # ==========================================================
    # INDEPENDENT t-TEST SUMMARY STATS (WELCH)
    # ==========================================================
    elif test_choice == "Independent t-Test using Summary Statistics (Welch)":
        st.subheader("📋 Enter Summary Statistics")
        m1 = st.number_input("x̄₁", value=0.0)
        s1 = st.number_input("s₁", value=1.0)
        n1 = st.number_input("n₁", value=2)
        m2 = st.number_input("x̄₂", value=0.0)
        s2 = st.number_input("s₂", value=1.0)
        n2 = st.number_input("n₂", value=2)

        if st.button("👨‍💻 Calculate"):
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            df_welch = ((s1**2/n1 + s2**2/n2)**2) / (
                (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
            )
            diff = m1 - m2
            t_stat = diff / se

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Compute SE and test statistic.")
            st.latex(r"t=\frac{\bar{x}_1-\bar{x}_2}{\text{SE}}")
            st.write(f"SE = {se:.{decimals}f},  t = {t_stat:.{decimals}f}")

            step_box("**Step 2:** Compute p-value and critical value(s).")
            p_val, reject, crit = t_tail_metrics(t_stat, df_welch, alpha, tails)

            st.markdown("### **Result Summary**")
            st.write(f"Test Statistic (t): {t_stat:.{decimals}f} (df ≈ {df_welch:.2f})")
            st.write(f"Critical Value(s): {crit}")
            st.write(f"P-value: {p_val:.{decimals}f}")
            st.markdown(decision_block(reject), unsafe_allow_html=True)

    # ==========================================================
    # F-TEST USING DATA
    # ==========================================================
    elif test_choice == "F-Test for Standard Deviations using Data":
        st.subheader("📊 Enter Sample Data")
        up = st.file_uploader("Upload CSV with Sample1, Sample2", type="csv")
        s1 = st.text_area("Sample 1", "10,12,11,13")
        s2 = st.text_area("Sample 2", "9,11,10,8")

        if st.button("👨‍💻 Calculate"):
            if up:
                df_csv = pd.read_csv(up)
                x1 = df_csv["Sample1"].astype(float).to_numpy()
                x2 = df_csv["Sample2"].astype(float).to_numpy()
            else:
                x1 = np.array([float(i) for i in s1.split(",")])
                x2 = np.array([float(i) for i in s2.split(",")])

            s1_val = np.std(x1, ddof=1)
            s2_val = np.std(x2, ddof=1)
            F = (s1_val**2) / (s2_val**2)

            df1, df2 = len(x1) - 1, len(x2) - 1

            step_box("**Step 1:** Compute F statistic and df.")
            st.latex(r"F=\frac{s_1^2}{s_2^2}")

            step_box("**Step 2:** Compute p-value and critical value(s).")
            p_val, reject, crit = f_tail_metrics(F, df1, df2, alpha, tails)

            st.markdown("### **Result Summary**")
            st.write(f"F statistic: {F:.{decimals}f} (df₁={df1}, df₂={df2})")
            st.write(f"Critical Value(s): {crit}")
            st.write(f"P-value: {p_val:.{decimals}f}")
            st.markdown(decision_block(reject), unsafe_allow_html=True)

    # ==========================================================
    # F-TEST SUMMARY STATS
    # ==========================================================
    else:
        st.subheader("📋 Enter Summary Statistics")
        n1 = st.number_input("Sample size 1 (n₁)", min_value=2)
        s1_val = st.number_input("Std Dev 1 (s₁)", value=1.0)
        n2 = st.number_input("Sample size 2 (n₂)", min_value=2)
        s2_val = st.number_input("Std Dev 2 (s₂)", value=1.0)

        if st.button("👨‍💻 Calculate"):
            F = (s1_val**2) / (s2_val**2)
            df1, df2 = n1 - 1, n2 - 1

            step_box("**Step 1:** Compute F statistic and df.")
            st.latex(r"F=\frac{s_1^2}{s_2^2}")

            step_box("**Step 2:** Compute p-value and critical value(s).")
            p_val, reject, crit = f_tail_metrics(F, df1, df2, alpha, tails)

            st.markdown("### **Result Summary**")
            st.write(f"F statistic: {F:.{decimals}f} (df₁={df1}, df₂={df2})")
            st.write(f"Critical Value(s): {crit}")
            st.write(f"P-value: {p_val:.{decimals}f}")
            st.markdown(decision_block(reject), unsafe_allow_html=True)

# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":
    run_two_sample_tool()
