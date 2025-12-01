# ==========================================================
# two_sample_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# MIND: Statistics Visualizer Suite
# Updated with Dark/Light Mode Safe Interpretation Boxes
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
            "F-Test (Summary)"
        ],
        index=None,
        placeholder="Select a test..."
    )

    if not test_choice:
        st.info("👆 Select a test to begin.")
        return

    # Decimal places
    dec = st.number_input("Decimal places for output:", 0, 10, 4)

    alpha = st.number_input("Significance level (α):", 0.001, 0.5, 0.05, step=0.01)
    tails = st.selectbox("Tail type:", ["two", "left", "right"])

    # Show CI only for Z and t tests
    show_ci = st.checkbox("Show Confidence Interval (two-sided only)")

    # ==========================================================
    # TWO-PROPORTION Z-TEST
    # ==========================================================
    if test_choice == "Two-Proportion Z-Test":
        st.subheader("Counts Input")
        x1 = st.number_input("Successes x₁:", 0, step=1)
        n1 = st.number_input("Sample size n₁:", 1, step=1)
        x2 = st.number_input("Successes x₂:", 0, step=1)
        n2 = st.number_input("Sample size n₂:", 1, step=1)

        if st.button("Calculate"):
            p1, p2 = x1/n1, x2/n2
            p_pool = (x1 + x2) / (n1 + n2)
            se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
            z = (p1 - p2)/se

            st.markdown("### 📘 Step-by-Step")
            step_box("**Step 1: Compute sample proportions**")
            st.latex(fr"\hat p_1={p1:.{dec}f},\; \hat p_2={p2:.{dec}f},\; \hat p={p_pool:.{dec}f}")

            step_box("**Step 2: Test statistic**")
            st.latex(r"z=\frac{\hat p_1-\hat p_2}{\sqrt{\hat p(1-\hat p)(1/n_1+1/n_2)}}")
            st.latex(fr"z={z:.{dec}f}")

            step_box("**Step 3: Tail-specific p-value**")
            p_val, reject, crit_str = z_tail_metrics(z, alpha, tails)

            st.markdown("### 📝 Result Summary")
            st.markdown(f"""
• Test Statistic (z): {z:.{dec}f}  
• Critical Value(s): {crit_str}  
• P-value: {p_val:.{dec}f}  
""")

            if show_ci:
                zcrit = stats.norm.ppf(1 - alpha/2)
                se_u = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
                diff = p1 - p2
                ci_low = diff - zcrit * se_u
                ci_high = diff + zcrit * se_u
                st.markdown(f"• Confidence Interval ({100*(1-alpha):.0f}%): ({ci_low:.{dec}f}, {ci_high:.{dec}f})")

            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"
            st.markdown(f"• Decision: {decision}")

    # ==========================================================
    # PAIRED t-TEST (DATA) — UPDATED WITH DATAFRAME PREVIEW
    # ==========================================================
    elif test_choice == "Paired t-Test (Data)":
        st.subheader("Enter Paired Samples")
        s1 = st.text_area("Sample 1:", "1,2,3,4")
        s2 = st.text_area("Sample 2:", "1,2,3,4")

        if st.button("Calculate"):
            x1 = np.array([float(i) for i in s1.split(",")])
            x2 = np.array([float(i) for i in s2.split(",")])

            # Compute differences
            d = x1 - x2
            n = len(d)
            mean_d = np.mean(d)
            sd_d = np.std(d, ddof=1)
            se = sd_d/np.sqrt(n)
            tstat = mean_d/se
            df = n - 1

            # --------------------------
            # DataFrame preview
            # --------------------------
            df_preview = pd.DataFrame({
                "x₁": x1,
                "x₂": x2,
                "dᵢ = x₁ − x₂": d
            })

            st.markdown("### 🔍 Data Preview (x₁, x₂, dᵢ)")
            st.dataframe(df_preview.style.format("{:.4f}"))

            # --------------------------
            # Step-by-step
            # --------------------------
            st.markdown("### 📘 Step-by-Step")
            step_box("**Step 1: Differences**")
            st.latex(r"d_i=x_{1i}-x_{2i}")
            st.markdown(f"**Differences (dᵢ):** {np.round(d, 4).tolist()}")
            st.latex(fr"\bar d={mean_d:.{dec}f},\; s_d={sd_d:.{dec}f},\; n={n}")

            step_box("**Step 2: Test statistic**")
            st.latex(r"t=\frac{\bar d}{s_d/\sqrt{n}}")
            st.latex(fr"t={tstat:.{dec}f}")

            p_val, reject, crit_str = t_tail_metrics(tstat, df, alpha, tails)

            st.markdown("### 📝 Result Summary")
            st.markdown(f"""
• Test Statistic (t): {tstat:.{dec}f}  
• Critical Value(s): {crit_str}  
• P-value: {p_val:.{dec}f}  
""")

            if show_ci:
                tcrit = stats.t.ppf(1 - alpha/2, df)
                ci_low = mean_d - tcrit*se
                ci_high = mean_d + tcrit*se
                st.markdown(
                    f"• Confidence Interval ({100*(1-alpha):.0f}%): "
                    f"({ci_low:.{dec}f}, {ci_high:.{dec}f})"
                )

            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"
            st.markdown(f"• Decision: {decision}")

    # ==========================================================
    # PAIRED SUMMARY
    # ==========================================================
    elif test_choice == "Paired t-Test (Summary)":
        st.subheader("Summary of Differences")
        mean_d = st.number_input("Mean difference (d̄):", value=0.0)
        sd_d = st.number_input("SD of differences (s_d):", value=1.0)
        n = st.number_input("Sample size n:", min_value=2, step=1)

        if st.button("Calculate"):
            df = n - 1
            se = sd_d/np.sqrt(n)
            tstat = mean_d/se

            st.markdown("### 📘 Step-by-Step")
            step_box("**Step 1: Test statistic**")
            st.latex(fr"t={tstat:.{dec}f}")

            p_val, reject, crit_str = t_tail_metrics(tstat, df, alpha, tails)

            st.markdown("### 📝 Result Summary")
            st.markdown(f"""
• Test Statistic (t): {tstat:.{dec}f}  
• Critical Value(s): {crit_str}  
• P-value: {p_val:.{dec}f}  
""")

            if show_ci:
                tcrit = stats.t.ppf(1 - alpha/2, df)
                ci_low = mean_d - tcrit*se
                ci_high = mean_d + tcrit*se
                st.markdown(f"• Confidence Interval ({100*(1-alpha):.0f}%): ({ci_low:.{dec}f}, {ci_high:.{dec}f})")

            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"
            st.markdown(f"• Decision: {decision}")

    # ==========================================================
    # WELCH t-TEST (DATA)
    # ==========================================================
    elif test_choice == "Independent t-Test (Data, Welch)":
        st.subheader("Independent Samples Data")
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
            df = (se**4)/(((s1**2/n1)**2)/(n1-1) + ((s2**2/n2)**2)/(n2-1))

            st.markdown("### 📘 Step-by-Step")
            step_box("**Step 1: Test statistic**")
            st.latex(fr"t={tstat:.{dec}f},\; df\approx{df:.2f}")

            p_val, reject, crit_str = t_tail_metrics(tstat, df, alpha, tails)

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
                st.markdown(f"• Confidence Interval ({100*(1-alpha):.0f}%): ({ci_low:.{dec}f}, {ci_high:.{dec}f})")

            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"
            st.markdown(f"• Decision: {decision}")

    # ==========================================================
    # SUMMARY WELCH
    # ==========================================================
    elif test_choice == "Independent t-Test (Summary, Welch)":
        st.subheader("Summary Statistics Input")
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
            df = (se**4)/(((s1**2/n1)**2)/(n1-1) + ((s2**2/n2)**2)/(n2-1))

            st.markdown("### 📘 Step-by-Step")
            step_box("**Step 1: Test statistic**")
            st.latex(fr"t={tstat:.{dec}f},\; df\approx{df:.2f}")

            p_val, reject, crit_str = t_tail_metrics(tstat, df, alpha, tails)

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
                st.markdown(f"• Confidence Interval ({100*(1-alpha):.0f}%): ({ci_low:.{dec}f}, {ci_high:.{dec}f})")

            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"
            st.markdown(f"• Decision: {decision}")

    # ==========================================================
    # F-TESTS
    # ==========================================================
    elif test_choice == "F-Test (Data)":
        st.subheader("Independent Samples Data")
        a = st.text_area("Sample 1:", "1,2,3,4")
        b = st.text_area("Sample 2:", "1,2,3,4")

        if st.button("Calculate"):
            x1 = np.array([float(i) for i in a.split(",")])
            x2 = np.array([float(i) for i in b.split(",")])
            s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
            n1, n2 = len(x1), len(x2)
            F = (s1**2)/(s2**2)
            df1, df2 = n1-1, n2-1

            st.markdown("### 📘 Step-by-Step")
            step_box("**Step 1: F statistic**")
            st.latex(fr"F={F:.{dec}f}")

            p_val, reject, crit_str = f_tail_metrics(F, df1, df2, alpha, tails)

            st.markdown("### 📝 Result Summary")
            st.markdown(f"""
• Test Statistic (F): {F:.{dec}f}  
• Critical Value(s): {crit_str}  
• P-value: {p_val:.{dec}f}  
• Decision: {"✅ Reject H₀" if reject else "❌ Do not reject H₀"}
""")

    elif test_choice == "F-Test (Summary)":
        st.subheader("Summary Statistics Input")
        n1 = st.number_input("n₁:", 2, step=1)
        s1 = st.number_input("s₁:", 1.0)
        n2 = st.number_input("n₂:", 2, step=1)
        s2 = st.number_input("s₂:", 1.0)

        if st.button("Calculate"):
            F = (s1**2)/(s2**2)
            df1, df2 = n1-1, n2-1

            st.markdown("### 📘 Step-by-Step")
            step_box("**Step 1: Compute F**")
            st.latex(fr"F={F:.{dec}f}")

            p_val, reject, crit_str = f_tail_metrics(F, df1, df2, alpha, tails)

            st.markdown("### 📝 Result Summary")
            st.markdown(f"""
• Test Statistic (F): {F:.{dec}f}  
• Critical Value(s): {crit_str}  
• P-value: {p_val:.{dec}f}  
• Decision: {"✅ Reject H₀" if reject else "❌ Do not reject H₀"}
""")

# ---------- RUN ----------
if __name__ == "__main__":
    run_two_sample_tool()
