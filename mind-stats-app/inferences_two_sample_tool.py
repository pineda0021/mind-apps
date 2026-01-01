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

# ==========================================================
# MAIN TOOL
# ==========================================================
def run_two_sample_tool():
    st.header("🧪 Two-Sample Hypothesis Tests (Step-by-Step)")

    test_choice = st.selectbox(
        "Choose a Two-Sample Test:",
        [
            "Independent t-Test (Data, Welch)",
            "Independent t-Test (Summary, Welch)"
        ],
        index=None,
        placeholder="Select a test..."
    )

    if not test_choice:
        return

    dec = st.number_input("Decimal places:", 0, 10, 4)
    alpha = st.number_input("Significance level (α):", 0.001, 0.5, 0.05)
    tails = st.selectbox("Tail type:", ["two", "left", "right"])
    show_ci = st.checkbox("Show Confidence Interval (two-sided only)")

    # ==========================================================
    # WELCH t-TEST (DATA)
    # ==========================================================
    if test_choice == "Independent t-Test (Data, Welch)":
        a = st.text_area("Sample 1:", "1,2,3,4")
        b = st.text_area("Sample 2:", "1,2,3,4")

        if st.button("Calculate"):
            x1 = np.array([float(i) for i in a.split(",")])
            x2 = np.array([float(i) for i in b.split(",")])

            n1, n2 = len(x1), len(x2)
            m1, m2 = np.mean(x1), np.mean(x2)
            s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)

            se = np.sqrt(s1**2/n1 + s2**2/n2)
            tstat = (m1 - m2)/se

            df = (se**4) / (
                ((s1**2/n1)**2)/(n1-1) +
                ((s2**2/n2)**2)/(n2-1)
            )

            df_crit = np.floor(df)

            step_box("**Step 1: Test statistic and degrees of freedom**")
            st.latex(fr"t={tstat:.{dec}f}")
            st.latex(fr"df_{{Welch}}\approx {df:.2f},\quad df_{{crit}}={int(df_crit)}")

            p_val, reject, crit_str = t_tail_metrics(tstat, df_crit, alpha, tails)

            st.markdown("### 📝 Result Summary")
            st.markdown(f"""
• Test Statistic (t): {tstat:.{dec}f}  
• Critical Value(s): {crit_str}  
• P-value: {p_val:.{dec}f}  
""")

            if show_ci:
                tcrit = stats.t.ppf(1 - alpha/2, df)
                diff = m1 - m2
                ci_low = diff - tcrit*se
                ci_high = diff + tcrit*se
                st.markdown(
                    f"• Confidence Interval ({100*(1-alpha):.0f}%): "
                    f"({ci_low:.{dec}f}, {ci_high:.{dec}f})"
                )

            st.markdown("• Decision: " + ("✅ Reject H₀" if reject else "❌ Do not reject H₀"))

    # ==========================================================
    # WELCH t-TEST (SUMMARY)
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
            diff = m1 - m2
            tstat = diff/se

            df = (se**4) / (
                ((s1**2/n1)**2)/(n1-1) +
                ((s2**2/n2)**2)/(n2-1)
            )

            df_crit = np.floor(df)

            step_box("**Step 1: Test statistic and degrees of freedom**")
            st.latex(fr"t={tstat:.{dec}f}")
            st.latex(fr"df_{{Welch}}\approx {df:.2f},\quad df_{{crit}}={int(df_crit)}")

            p_val, reject, crit_str = t_tail_metrics(tstat, df_crit, alpha, tails)

            st.markdown("### 📝 Result Summary")
            st.markdown(f"""
• Test Statistic (t): {tstat:.{dec}f}  
• Critical Value(s): {crit_str}  
• P-value: {p_val:.{dec}f}  
""")

            if show_ci:
                tcrit = stats.t.ppf(1 - alpha/2, df)
                ci_low = diff - tcrit*se
                ci_high = diff + tcrit*se
                st.markdown(
                    f"• Confidence Interval ({100*(1-alpha):.0f}%): "
                    f"({ci_low:.{dec}f}, {ci_high:.{dec}f})"
                )

            st.markdown("• Decision: " + ("✅ Reject H₀" if reject else "❌ Do not reject H₀"))

# ---------- RUN ----------
if __name__ == "__main__":
    run_two_sample_tool()
