# ==========================================================
# two_sample_tool_correct.py
# Clean, correct hypothesis testing & confidence intervals
# ==========================================================
import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------
# UI Helper
# ----------------------------------------------------------
def step_box(text):
    st.markdown(
        f"""
        <div style="background-color:#eef6ff;padding:10px;border-radius:10px;
        border-left:5px solid #007acc;margin-bottom:10px;">
        <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------------------------------------
# Generic utilities
# ----------------------------------------------------------
def t_critical(df, alpha, tail):
    if tail == "left":
        return stats.t.ppf(alpha, df)
    elif tail == "right":
        return stats.t.ppf(1 - alpha, df)
    else:
        return stats.t.ppf(1 - alpha/2, df)

def t_p_value(t, df, tail):
    if tail == "left":
        return stats.t.cdf(t, df)
    elif tail == "right":
        return 1 - stats.t.cdf(t, df)
    else:
        return 2 * (1 - stats.t.cdf(abs(t), df))

def z_critical(alpha, tail):
    if tail == "left":
        return stats.norm.ppf(alpha)
    elif tail == "right":
        return stats.norm.ppf(1 - alpha)
    else:
        return stats.norm.ppf(1 - alpha/2)

def z_p_value(z, tail):
    if tail == "left":
        return stats.norm.cdf(z)
    elif tail == "right":
        return 1 - stats.norm.cdf(z)
    else:
        return 2 * (1 - stats.norm.cdf(abs(z)))

# ----------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------
def run_two_sample_tool():
    st.header("🧪 Two-Sample Hypothesis Tests (Step-by-Step)")

    test_choice = st.selectbox(
        "Choose a test:",
        [
            "Two-Proportion Z-Test",
            "Paired t-Test (Data)",
            "Paired t-Test (Summary)",
            "Independent t-Test (Data, Welch)",
            "Independent t-Test (Summary, Welch)",
            "F-Test (Data)",
            "F-Test (Summary)"
        ],
        index=None
        placeholder="Select a test..."
    )

    if not test_choice:
    st.info("👆 Please select a hypothesis test to begin.")
    return

    dec = st.number_input("Decimal places:", 0, 10, 4)
    alpha = st.number_input("Significance level α:", 0.001, 0.5, 0.05, step=0.01)
    tail = st.selectbox("Tail type:", ["two", "left", "right"])
    show_ci = st.checkbox("Show confidence interval (two-tailed only)")

    # ======================================================
    # TWO-PROPORTION Z-TEST
    # ======================================================
    if test_choice == "Two-Proportion Z-Test":
        x1 = st.number_input("Successes x₁:", 0)
        n1 = st.number_input("Sample size n₁:", 1)
        x2 = st.number_input("Successes x₂:", 0)
        n2 = st.number_input("Sample size n₂:", 1)

        if st.button("Calculate"):
            p1, p2 = x1/n1, x2/n2
            p_pool = (x1 + x2)/(n1 + n2)
            se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
            z = (p1 - p2)/se

            crit = z_critical(alpha, tail)
            p_val = z_p_value(z, tail)
            reject = abs(z) > abs(crit) if tail=="two" else (
                z < crit if tail=="left" else z > crit
            )

            st.markdown(f"""
**z:** {z:.{dec}f}  
**Critical value(s):** {"±" if tail=="two" else ""}{abs(crit):.{dec}f}  
**P-value:** {p_val:.{dec}f}  
**Decision:** {"Reject H₀" if reject else "Do not reject H₀"}
""")

    # ======================================================
    # PAIRED t-TEST (DATA)
    # ======================================================
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
            t = mean_d/se
            df = len(d) - 1

            crit = t_critical(df, alpha, tail)
            p_val = t_p_value(t, df, tail)
            reject = abs(t) > abs(crit) if tail=="two" else (
                t < crit if tail=="left" else t > crit
            )

            st.markdown(f"""
**t:** {t:.{dec}f}  
**df:** {df}  
**Critical value(s):** {"±" if tail=="two" else ""}{abs(crit):.{dec}f}  
**P-value:** {p_val:.{dec}f}  
**Decision:** {"Reject H₀" if reject else "Do not reject H₀"}
""")

            if show_ci and tail=="two":
                t_ci = stats.t.ppf(1-alpha/2, df)
                ci = (mean_d - t_ci*se, mean_d + t_ci*se)
                st.markdown(f"**{100*(1-alpha):.0f}% CI:** ({ci[0]:.{dec}f}, {ci[1]:.{dec}f})")

    # ======================================================
    # INDEPENDENT t-TEST (DATA, WELCH)
    # ======================================================
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
            t = (m1 - m2)/se

            df = (s1**2/n1 + s2**2/n2)**2 / (
                (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
            )

            crit = t_critical(df, alpha, tail)
            p_val = t_p_value(t, df, tail)
            reject = abs(t) > abs(crit) if tail=="two" else (
                t < crit if tail=="left" else t > crit
            )

            st.markdown(f"""
**t:** {t:.{dec}f}  
**df (Welch):** {df:.2f}  
**Critical value(s):** {"±" if tail=="two" else ""}{abs(crit):.{dec}f}  
**P-value:** {p_val:.{dec}f}  
**Decision:** {"Reject H₀" if reject else "Do not reject H₀"}
""")

            if show_ci and tail=="two":
                t_ci = stats.t.ppf(1-alpha/2, df)
                diff = m1 - m2
                ci = (diff - t_ci*se, diff + t_ci*se)
                st.markdown(f"**{100*(1-alpha):.0f}% CI:** ({ci[0]:.{dec}f}, {ci[1]:.{dec}f})")

    # ======================================================
    # F-TESTS (DATA & SUMMARY)
    # ======================================================
    elif test_choice == "F-Test (Data)":
        a = st.text_area("Sample 1:", "1,2,3")
        b = st.text_area("Sample 2:", "4,5,6")

        if st.button("Calculate"):
            x1 = np.array([float(i) for i in a.split(",")])
            x2 = np.array([float(i) for i in b.split(",")])

            F = np.var(x1, ddof=1)/np.var(x2, ddof=1)
            df1, df2 = len(x1)-1, len(x2)-1

            crit_low = stats.f.ppf(alpha/2, df1, df2)
            crit_high = stats.f.ppf(1-alpha/2, df1, df2)
            p_val = 2 * min(stats.f.cdf(F, df1, df2), 1-stats.f.cdf(F, df1, df2))
            reject = F < crit_low or F > crit_high

            st.markdown(f"""
**F:** {F:.{dec}f}  
**Critical values:** {crit_low:.{dec}f}, {crit_high:.{dec}f}  
**P-value:** {p_val:.{dec}f}  
**Decision:** {"Reject H₀" if reject else "Do not reject H₀"}
""")

# ----------------------------------------------------------
# RUN
# ----------------------------------------------------------
if __name__ == "__main__":
    run_two_sample_tool()

  
