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

    dec = st.number_input("Decimal places for output:", 0, 10, 4, key="dec")
    alpha = st.number_input("Significance level (α):", 0.001, 0.5, 0.05, step=0.01, key="alpha")
    tails = st.selectbox("Tail type:", ["two", "left", "right"], key="tails")
    show_ci = st.checkbox("Show Confidence Interval (two-sided only)", key="show_ci")

    # ==========================================================
    # TWO-PROPORTION Z-TEST
    # ==========================================================
    if test_choice == "Two-Proportion Z-Test":
        x1 = st.number_input("Successes x₁:", 0, step=1, key="z_x1")
        n1 = st.number_input("Sample size n₁:", 1, step=1, key="z_n1")
        x2 = st.number_input("Successes x₂:", 0, step=1, key="z_x2")
        n2 = st.number_input("Sample size n₂:", 1, step=1, key="z_n2")

        if st.button("Calculate", key="z_calc"):
            p1, p2 = x1/n1, x2/n2
            p_pool = (x1 + x2)/(n1 + n2)
            se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
            z = (p1 - p2)/se

            p_val, reject, crit_str = z_tail_metrics(z, alpha, tails)

            st.markdown("### 📝 Result Summary")
            st.caption("🧪 Hypothesis-test critical values depend on the alternative hypothesis.")
            st.markdown(f"""
• z = {z:.{dec}f}  
• Critical Value (Hypothesis Test): {crit_str}  
• P-value: {p_val:.{dec}f}  
• Decision: {"Reject H₀" if reject else "Do not reject H₀"}
""")

            if show_ci:
                st.caption("💡 Confidence intervals are always two-sided.")
                zcrit = stats.norm.ppf(1 - alpha/2)
                diff = p1 - p2
                ci_low = diff - zcrit*se
                ci_high = diff + zcrit*se
                st.markdown(f"CI: ({ci_low:.{dec}f}, {ci_high:.{dec}f})")

    # ==========================================================
    # PAIRED t-TEST (DATA)
    # ==========================================================
    elif test_choice == "Paired t-Test (Data)":
        s1 = st.text_area("Sample 1:", "1,2,3,4", key="paired_s1")
        s2 = st.text_area("Sample 2:", "1,2,3,4", key="paired_s2")

        if st.button("Calculate", key="paired_calc"):
            x1 = np.array([float(i) for i in s1.split(",")])
            x2 = np.array([float(i) for i in s2.split(",")])
            d = x1 - x2

            st.dataframe(pd.DataFrame({"x₁": x1, "x₂": x2, "dᵢ": d}))

            mean_d = np.mean(d)
            sd_d = np.std(d, ddof=1)
            se = sd_d/np.sqrt(len(d))
            tstat = mean_d/se
            df = len(d)-1

            p_val, reject, crit_str = t_tail_metrics(tstat, df, alpha, tails)

            st.markdown("### 📝 Result Summary")
            st.caption("🧪 Hypothesis-test critical values depend on the alternative hypothesis.")
            st.markdown(f"""
• t = {tstat:.{dec}f}  
• Critical Value (Hypothesis Test): {crit_str}  
• P-value: {p_val:.{dec}f}  
• Decision: {"Reject H₀" if reject else "Do not reject H₀"}
""")

    # (Other tests follow same key pattern — omitted here only for brevity)

# ---------- RUN ----------
if __name__ == "__main__":
    run_two_sample_tool()
