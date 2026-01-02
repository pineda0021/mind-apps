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
        crit = stats.norm.ppf(1 - alpha / 2)
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
        crit = stats.t.ppf(1 - alpha / 2, df)
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
        crit_low = stats.f.ppf(alpha / 2, df1, df2)
        crit_high = stats.f.ppf(1 - alpha / 2, df1, df2)
        p = 2 * min(stats.f.cdf(F, df1, df2), 1 - stats.f.cdf(F, df1, df2))
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
            "F-Test (Summary)",
        ],
        index=None,
        placeholder="Select a test...",
    )

    if not test_choice:
        st.info("👆 Select a test to begin.")
        return

    dec = st.number_input("Decimal places for output:", 0, 10, 4)
    alpha = st.number_input("Significance level (α):", 0.001, 0.5, 0.05, step=0.01)
    tails = st.selectbox("Tail type:", ["two", "left", "right"])
    show_ci = st.checkbox("Show Confidence Interval")

    st.caption(
        "ℹ️ Critical values depend on the tail type. "
        "Confidence intervals **always use a two-tailed critical value** "
        "and are only valid for **two-sided tests**."
    )

    # ==========================================================
    # F-TEST (DATA)
    # ==========================================================
    if test_choice == "F-Test (Data)":
        st.subheader("Independent Samples Data")
        a = st.text_area("Sample 1:", "1,2,3,4")
        b = st.text_area("Sample 2:", "1,2,3,4")

        if st.button("Calculate"):
            x1 = np.array([float(i) for i in a.split(",")])
            x2 = np.array([float(i) for i in b.split(",")])

            s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
            n1, n2 = len(x1), len(x2)
            F = (s1**2) / (s2**2)
            df1, df2 = n1 - 1, n2 - 1

            st.markdown("### 📘 Step-by-Step")
            step_box("**Step 1: F statistic**")
            st.latex(fr"F = {F:.{dec}f}")

            p_val, reject, crit_str = f_tail_metrics(F, df1, df2, alpha, tails)

            st.markdown("### 📝 Result Summary")
            st.markdown(
                f"""
• **Test Statistic (F):** {F:.{dec}f}  
• **Critical Value(s):** {crit_str}  
• **P-value:** {p_val:.{dec}f}  
• **Decision:** {"✅ Reject H₀" if reject else "❌ Do not reject H₀"}
"""
            )


# ---------- RUN ----------
if __name__ == "__main__":
    run_two_sample_tool()
