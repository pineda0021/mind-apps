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

# ---------- Tail utilities (math unchanged) ----------
def z_tail_metrics(z, alpha, tail):
    if tail == "left":
        crit = stats.norm.ppf(alpha)
        p = stats.norm.cdf(z)
        reject = z < crit
    elif tail == "right":
        crit = stats.norm.ppf(1 - alpha)
        p = 1 - stats.norm.cdf(z)
        reject = z > crit
    else:
        crit = stats.norm.ppf(1 - alpha/2)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        reject = abs(z) > crit
    return p, reject, crit

def t_tail_metrics(tval, df, alpha, tail):
    if tail == "left":
        crit = stats.t.ppf(alpha, df)
        p = stats.t.cdf(tval, df)
        reject = tval < crit
    elif tail == "right":
        crit = stats.t.ppf(1 - alpha, df)
        p = 1 - stats.t.cdf(tval, df)
        reject = tval > crit
    else:
        crit = stats.t.ppf(1 - alpha/2, df)
        p = 2 * (1 - stats.t.cdf(abs(tval), df))
        reject = abs(tval) > crit
    return p, reject, crit

def f_tail_metrics(F, df1, df2, alpha, tail):
    if tail == "left":
        crit = stats.f.ppf(alpha, df1, df2)
        p = stats.f.cdf(F, df1, df2)
        reject = F < crit
        return p, reject, crit
    elif tail == "right":
        crit = stats.f.ppf(1 - alpha, df1, df2)
        p = 1 - stats.f.cdf(F, df1, df2)
        reject = F > crit
        return p, reject, crit
    else:
        crit_low = stats.f.ppf(alpha/2, df1, df2)
        crit_high = stats.f.ppf(1 - alpha/2, df1, df2)
        p = 2 * min(stats.f.cdf(F, df1, df2),
                    1 - stats.f.cdf(F, df1, df2))
        reject = (F < crit_low) or (F > crit_high)
        return p, reject, (crit_low, crit_high)

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
        placeholder="Select a test..."
    )

    if not test_choice:
        st.info("👆 Select a test to begin.")
        return

    dec = st.number_input("Decimal places for output:", 0, 10, 4)
    alpha = st.number_input("Significance level (α):", 0.001, 0.5, 0.05, step=0.01)
    tails = st.selectbox("Tail type:", ["two", "left", "right"])
    show_ci = st.checkbox("Show Confidence Interval (two-sided only)")

    # ==========================================================
    # TWO-PROPORTION Z-TEST
    # ==========================================================
    if test_choice == "Two-Proportion Z-Test":
        x1 = st.number_input("Successes x₁:", 0, step=1)
        n1 = st.number_input("Sample size n₁:", 1, step=1)
        x2 = st.number_input("Successes x₂:", 0, step=1)
        n2 = st.number_input("Sample size n₂:", 1, step=1)

        if st.button("Calculate"):
            p1, p2 = x1/n1, x2/n2
            p_pool = (x1 + x2)/(n1 + n2)
            se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
            z = (p1 - p2)/se

            p_val, reject, crit = z_tail_metrics(z, alpha, tails)

            st.markdown("### 📝 Result Summary")

            if tails == "left":
                st.markdown(f"• Critical Value (Hypothesis Test): −{abs(crit):.{dec}f}  (Reject if z < −{abs(crit):.{dec}f})")
            elif tails == "right":
                st.markdown(f"• Critical Value (Hypothesis Test): {abs(crit):.{dec}f}  (Reject if z > {abs(crit):.{dec}f})")
            else:
                st.markdown(f"• Critical Value (Hypothesis Test): ±{abs(crit):.{dec}f}")

            st.markdown(f"• Test Statistic (z): {z:.{dec}f}")
            st.markdown(f"• P-value: {p_val:.{dec}f}")

            if show_ci:
                zcrit = stats.norm.ppf(1 - alpha/2)
                diff = p1 - p2
                ci_low = diff - zcrit*se
                ci_high = diff + zcrit*se
                st.markdown(f"• Confidence Interval ({100*(1-alpha):.0f}%): ({ci_low:.{dec}f}, {ci_high:.{dec}f})")

            st.markdown(f"• Decision: {'✅ Reject H₀' if reject else '❌ Do not reject H₀'}")

    # ==========================================================
    # PAIRED t-TEST (DATA)
    # ==========================================================
    elif test_choice == "Paired t-Test (Data)":
        s1 = st.text_area("Sample 1:", "1,2,3")
        s2 = st.text_area("Sample 2:", "1,2,3")

        if st.button("Calculate"):
            x1 = np.array([float(i) for i in s1.split(",")])
            x2 = np.array([float(i) for i in s2.split(",")])
            d = x1 - x2

            mean_d = np.mean(d)
            sd_d = np.std(d, ddof=1)
            se = sd_d/np.sqrt(len(d))
            tstat = mean_d/se
            df = len(d)-1

            p_val, reject, crit = t_tail_metrics(tstat, df, alpha, tails)

            st.markdown("### 📝 Result Summary")

            if tails == "left":
                st.markdown(f"• Critical Value (Hypothesis Test): −{abs(crit):.{dec}f}  (Reject if t < −{abs(crit):.{dec}f})")
            elif tails == "right":
                st.markdown(f"• Critical Value (Hypothesis Test): {abs(crit):.{dec}f}  (Reject if t > {abs(crit):.{dec}f})")
            else:
                st.markdown(f"• Critical Value (Hypothesis Test): ±{abs(crit):.{dec}f}")

            st.markdown(f"• Test Statistic (t): {tstat:.{dec}f}")
            st.markdown(f"• P-value: {p_val:.{dec}f}")

            if show_ci:
                tcrit = stats.t.ppf(1 - alpha/2, df)
                ci_low = mean_d - tcrit*se
                ci_high = mean_d + tcrit*se
                st.markdown(f"• Confidence Interval ({100*(1-alpha):.0f}%): ({ci_low:.{dec}f}, {ci_high:.{dec}f})")

            st.markdown(f"• Decision: {'✅ Reject H₀' if reject else '❌ Do not reject H₀'}")

    # ==========================================================
    # PAIRED t-TEST (SUMMARY)
    # ==========================================================
    elif test_choice == "Paired t-Test (Summary)":
        mean_d = st.number_input("Mean difference (d̄):", 0.0)
        sd_d = st.number_input("SD of differences:", 1.0)
        n = st.number_input("Sample size n:", 2, step=1)

        if st.button("Calculate"):
            se = sd_d/np.sqrt(n)
            tstat = mean_d/se
            df = n-1

            p_val, reject, crit = t_tail_metrics(tstat, df, alpha, tails)

            st.markdown("### 📝 Result Summary")

            if tails == "left":
                st.markdown(f"• Critical Value (Hypothesis Test): −{abs(crit):.{dec}f}")
            elif tails == "right":
                st.markdown(f"• Critical Value (Hypothesis Test): {abs(crit):.{dec}f}")
            else:
                st.markdown(f"• Critical Value (Hypothesis Test): ±{abs(crit):.{dec}f}")

            st.markdown(f"• Test Statistic (t): {tstat:.{dec}f}")
            st.markdown(f"• P-value: {p_val:.{dec}f}")
            st.markdown(f"• Decision: {'✅ Reject H₀' if reject else '❌ Do not reject H₀'}")

    # ==========================================================
    # INDEPENDENT t-TEST (DATA, WELCH)
    # ==========================================================
    elif test_choice == "Independent t-Test (Data, Welch)":
        a = st.text_area("Sample 1:", "1,2,3")
        b = st.text_area("Sample 2:", "4,5,6")

        if st.button("Calculate"):
            x1 = np.array([float(i) for i in a.split(",")])
            x2 = np.array([float(i) for i in b.split(",")])

            m1, m2 = np.mean(x1), np.mean(x2)
            s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
            n1, n2 = len(x1), len(x2)

            se = np.sqrt(s1**2/n1 + s2**2/n2)
            tstat = (m1 - m2)/se
            df = (se**4)/(((s1**2/n1)**2)/(n1-1) + ((s2**2/n2)**2)/(n2-1))

            p_val, reject, crit = t_tail_metrics(tstat, df, alpha, tails)

            st.markdown("### 📝 Result Summary")

            if tails == "left":
                st.markdown(f"• Critical Value (Hypothesis Test): −{abs(crit):.{dec}f}")
            elif tails == "right":
                st.markdown(f"• Critical Value (Hypothesis Test): {abs(crit):.{dec}f}")
            else:
                st.markdown(f"• Critical Value (Hypothesis Test): ±{abs(crit):.{dec}f}")

            st.markdown(f"• Test Statistic (t): {tstat:.{dec}f}")
            st.markdown(f"• P-value: {p_val:.{dec}f}")
            st.markdown(f"• Decision: {'✅ Reject H₀' if reject else '❌ Do not reject H₀'}")

    # ==========================================================
    # INDEPENDENT t-TEST (SUMMARY, WELCH)
    # ==========================================================
    elif test_choice == "Independent t-Test (Summary, Welch)":
        m1 = st.number_input("Mean 1:", 0.0)
        s1 = st.number_input("SD 1:", 1.0)
        n1 = st.number_input("n₁:", 2, step=1)
        m2 = st.number_input("Mean 2:", 0.0)
        s2 = st.number_input("SD 2:", 1.0)
        n2 = st.number_input("n₂:", 2, step=1)

        if st.button("Calculate"):
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            tstat = (m1 - m2)/se
            df = (se**4)/(((s1**2/n1)**2)/(n1-1) + ((s2**2/n2)**2)/(n2-1))

            p_val, reject, crit = t_tail_metrics(tstat, df, alpha, tails)

            st.markdown("### 📝 Result Summary")

            if tails == "left":
                st.markdown(f"• Critical Value (Hypothesis Test): −{abs(crit):.{dec}f}")
            elif tails == "right":
                st.markdown(f"• Critical Value (Hypothesis Test): {abs(crit):.{dec}f}")
            else:
                st.markdown(f"• Critical Value (Hypothesis Test): ±{abs(crit):.{dec}f}")

            st.markdown(f"• Test Statistic (t): {tstat:.{dec}f}")
            st.markdown(f"• P-value: {p_val:.{dec}f}")
            st.markdown(f"• Decision: {'✅ Reject H₀' if reject else '❌ Do not reject H₀'}")

    # ==========================================================
    # F-TEST (DATA)
    # ==========================================================
    elif test_choice == "F-Test (Data)":
        a = st.text_area("Sample 1:", "1,2,3")
        b = st.text_area("Sample 2:", "4,5,6")

        if st.button("Calculate"):
            x1 = np.array([float(i) for i in a.split(",")])
            x2 = np.array([float(i) for i in b.split(",")])
            s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
            F = (s1**2)/(s2**2)
            df1, df2 = len(x1)-1, len(x2)-1

            p_val, reject, crit = f_tail_metrics(F, df1, df2, alpha, tails)

            st.markdown("### 📝 Result Summary")

            if tails == "two":
                st.markdown(f"• Critical Value (Hypothesis Test): ({crit[0]:.{dec}f}, {crit[1]:.{dec}f})")
            else:
                st.markdown(f"• Critical Value (Hypothesis Test): {crit:.{dec}f}")

            st.markdown(f"• Test Statistic (F): {F:.{dec}f}")
            st.markdown(f"• P-value: {p_val:.{dec}f}")
            st.markdown(f"• Decision: {'✅ Reject H₀' if reject else '❌ Do not reject H₀'}")

    # ==========================================================
    # F-TEST (SUMMARY)
    # ==========================================================
    elif test_choice == "F-Test (Summary)":
        n1 = st.number_input("n₁:", 2, step=1)
        s1 = st.number_input("s₁:", 1.0)
        n2 = st.number_input("n₂:", 2, step=1)
        s2 = st.number_input("s₂:", 1.0)

        if st.button("Calculate"):
            F = (s1**2)/(s2**2)
            df1, df2 = n1-1, n2-1

            p_val, reject, crit = f_tail_metrics(F, df1, df2, alpha, tails)

            st.markdown("### 📝 Result Summary")

            if tails == "two":
                st.markdown(f"• Critical Value (Hypothesis Test): ({crit[0]:.{dec}f}, {crit[1]:.{dec}f})")
            else:
                st.markdown(f"• Critical Value (Hypothesis Test): {crit:.{dec}f}")

            st.markdown(f"• Test Statistic (F): {F:.{dec}f}")
            st.markdown(f"• P-value: {p_val:.{dec}f}")
            st.markdown(f"• Decision: {'✅ Reject H₀' if reject else '❌ Do not reject H₀'}")

# ---------- RUN ----------
if __name__ == "__main__":
    run_two_sample_tool()

            
