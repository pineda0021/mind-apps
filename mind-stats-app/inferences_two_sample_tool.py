# ==========================================================
# two_sample_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# MIND: Statistics Visualizer Suite
# ==========================================================
import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

# ---------- UI Helper ----------
def step_box(text: str):
    st.markdown(
        f"""
        <div style="background-color:#eef6ff;padding:10px;border-radius:10px;
        border-left:5px solid #007acc;margin-bottom:10px;">
        <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Tail utilities ----------
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
        p = 2 * min(stats.f.cdf(F, df1, df2),
                    1 - stats.f.cdf(F, df1, df2))
        reject = (F < crit_low) or (F > crit_high)
        crit_str = f"{crit_low:.4f}, {crit_high:.4f}"
    return p, reject, crit_str

# ==========================================================
# MAIN TOOL
# ==========================================================
def run_two_sample_tool():
    st.header("🧪 Two-Sample Hypothesis Tests (Step-by-Step)")

    test_choice = st.selectbox(
        "Choose a Two-Sample Test:",
        [
            "Two-Proportion Z-Test",
            "Paired t-Test (Data)",
            "Paired t-Test (Summary)",
            "Independent t-Test (Data, Welch)",
            "Independent t-Test (Summary, Welch)",
            "F-Test (Data)",
            "F-Test (Summary)"
        ],
        index=None,
        placeholder="Select a test...",
        key="test_choice"
    )

    if not test_choice:
        st.info("👆 Select a test to begin.")
        return

    dec = st.number_input("Decimal places:", 0, 10, 4, key="dec")
    alpha = st.number_input("Significance level (α):", 0.001, 0.5, 0.05, step=0.01, key="alpha")
    tails = st.selectbox("Tail type:", ["two", "left", "right"], key="tails")
    show_ci = st.checkbox("Show Confidence Interval (two-sided only)", key="show_ci")

    # ==========================================================
    # PAIRED t-TEST (SUMMARY)
    # ==========================================================
    if test_choice == "Paired t-Test (Summary)":
        mean_d = st.number_input("Mean difference (d̄):", 0.0, key="paired_sum_mean")
        sd_d = st.number_input("SD of differences (s_d):", 1.0, key="paired_sum_sd")
        n = st.number_input("Sample size n:", 2, step=1, key="paired_sum_n")

        if st.button("Calculate", key="paired_sum_calc"):
            df = n - 1
            se = sd_d / np.sqrt(n)
            tstat = mean_d / se

            st.markdown("### 🧮 Step-by-Step")
            step_box("**Step 1: Test statistic**")
            st.latex(fr"t = {tstat:.{dec}f}")

            p_val, reject, crit_str = t_tail_metrics(tstat, df, alpha, tails)

            st.markdown("### 📝 Result Summary")
            st.caption("🧪 Hypothesis-test critical values depend on the alternative hypothesis.")
            st.markdown(f"""
• Test Statistic (t): {tstat:.{dec}f}  
• Critical Value (Hypothesis Test): {crit_str}  
• P-value: {p_val:.{dec}f}  
• Decision: {"Reject H₀" if reject else "Do not reject H₀"}
""")

            if show_ci:
                st.caption("💡 Confidence intervals are always two-sided.")
                tcrit = stats.t.ppf(1 - alpha/2, df)
                ci_low = mean_d - tcrit * se
                ci_high = mean_d + tcrit * se
                st.markdown(f"CI: ({ci_low:.{dec}f}, {ci_high:.{dec}f})")

    # ==========================================================
    # INDEPENDENT t-TEST (SUMMARY, WELCH)
    # ==========================================================
    elif test_choice == "Independent t-Test (Summary, Welch)":
        m1 = st.number_input("Mean 1:", 0.0, key="welch_sum_m1")
        s1 = st.number_input("SD 1:", 1.0, key="welch_sum_s1")
        n1 = st.number_input("n₁:", 2, step=1, key="welch_sum_n1")
        m2 = st.number_input("Mean 2:", 0.0, key="welch_sum_m2")
        s2 = st.number_input("SD 2:", 1.0, key="welch_sum_s2")
        n2 = st.number_input("n₂:", 2, step=1, key="welch_sum_n2")

        if st.button("Calculate", key="welch_sum_calc"):
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            diff = m1 - m2
            tstat = diff / se

            df = (se**4) / (
                ((s1**2/n1)**2)/(n1-1) +
                ((s2**2/n2)**2)/(n2-1)
            )

            step_box("**Step 1: Test statistic**")
            st.latex(fr"t = {tstat:.{dec}f}, \; df \approx {df:.2f}")

            p_val, reject, crit_str = t_tail_metrics(tstat, df, alpha, tails)

            st.markdown("### 📝 Result Summary")
            st.caption("🧪 Hypothesis-test critical values depend on the alternative hypothesis.")
            st.markdown(f"""
• Test Statistic (t): {tstat:.{dec}f}  
• Critical Value (Hypothesis Test): {crit_str}  
• P-value: {p_val:.{dec}f}  
• Decision: {"Reject H₀" if reject else "Do not reject H₀"}
""")

            if show_ci:
                st.caption("💡 Confidence intervals are always two-sided.")
                tcrit = stats.t.ppf(1 - alpha/2, df)
                ci_low = diff - tcrit * se
                ci_high = diff + tcrit * se
                st.markdown(f"CI: ({ci_low:.{dec}f}, {ci_high:.{dec}f})")

    # ==========================================================
    # F-TEST (SUMMARY)
    # ==========================================================
    elif test_choice == "F-Test (Summary)":
        n1 = st.number_input("n₁:", 2, step=1, key="f_sum_n1")
        s1 = st.number_input("s₁:", 1.0, key="f_sum_s1")
        n2 = st.number_input("n₂:", 2, step=1, key="f_sum_n2")
        s2 = st.number_input("s₂:", 1.0, key="f_sum_s2")

        if st.button("Calculate", key="f_sum_calc"):
            F = (s1**2) / (s2**2)
            df1, df2 = n1-1, n2-1

            step_box("**Step 1: Compute F**")
            st.latex(fr"F = {F:.{dec}f}")

            p_val, reject, crit_str = f_tail_metrics(F, df1, df2, alpha, tails)

            st.markdown("### 📝 Result Summary")
            st.caption("🧪 Hypothesis-test critical values depend on the alternative hypothesis.")
            st.markdown(f"""
• Test Statistic (F): {F:.{dec}f}  
• Critical Value (Hypothesis Test): {crit_str}  
• P-value: {p_val:.{dec}f}  
• Decision: {"Reject H₀" if reject else "Do not reject H₀"}
""")

# ---------- RUN ----------
if __name__ == "__main__":
    run_two_sample_tool()

     

