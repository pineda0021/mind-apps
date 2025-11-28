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
# UI Helper – STEP BOX (Screenshot Style A)
# ==========================================================
def step_box(text: str):
    st.markdown(
        f"""
        <div style="
            background-color:#e6f3ff;
            padding:12px;
            border-radius:10px;
            border-left:6px solid #007acc;
            margin-top:12px;
            margin-bottom:12px;
            ">
            <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================
# Utility Functions
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
# MAIN APP
# ==========================================================
def run_two_sample_tool():

    st.header("👨🏻‍🔬 Two-Sample Inference Suite")
    st.caption("Created by Professor Edward Pineda-Castro — MIND Statistics Visualizer Suite")

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
    decimals = st.number_input("Decimal places for output:", 0, 10, 4)
    show_ci = st.checkbox("Show Confidence Interval (two-sided)")

    # ==========================================================
    # 1) TWO-PROPORTION Z-TEST
    # ==========================================================
    if test_choice == "Two-Proportion Z-Test":
        st.subheader("📊 Two-Proportion Z-Test")

        x1 = st.number_input("Successes in Sample 1 (x₁)", min_value=0, step=1)
        n1 = st.number_input("Sample size 1 (n₁)", min_value=1, step=1)
        x2 = st.number_input("Successes in Sample 2 (x₂)", min_value=0, step=1)
        n2 = st.number_input("Sample size 2 (n₂)", min_value=1, step=1)

        if st.button("Calculate"):
            p1 = x1 / n1
            p2 = x2 / n2
            pooled = (x1 + x2) / (n1 + n2)
            se = np.sqrt(pooled * (1 - pooled) * (1/n1 + 1/n2))
            z = (p1 - p2) / se

            st.markdown("### 📘 Step-by-Step")
            step_box("**Step 1:** Compute sample proportions and the pooled estimate.")
            st.latex(r"\hat p_1=\frac{x_1}{n_1},\; \hat p_2=\frac{x_2}{n_2}")
            st.latex(fr"\hat p_1={p1:.{decimals}f},\; \hat p_2={p2:.{decimals}f}")
            st.latex(fr"\hat p={{x_1+x_2}\over{n_1+n_2}}={pooled:.{decimals}f}")

            step_box("**Step 2:** Standard error and test statistic.")
            st.latex(r"z=\frac{\hat p_1-\hat p_2}{\sqrt{\hat p(1-\hat p)(1/n_1+1/n_2)}}")
            st.latex(fr"\text{{SE}}={se:.{decimals}f},\; z={z:.{decimals}f}")

            step_box("**Step 3:** Compute p-value and critical value(s).")
            p_val, reject, crit_str = z_tail_metrics(z, alpha, tails)

            step_box("**Step 4:** Make a decision.")
            decision = "✔ Reject H₀" if reject else "✖ Fail to reject H₀"

            st.markdown(f"""
### **Result Summary**
- Test Statistic (z): {z:.{decimals}f}  
- Critical Value(s): {crit_str}  
- P-value: {p_val:.{decimals}f}  
- **Decision:** {decision}
""")

            if show_ci:
                zcrit = stats.norm.ppf(1 - alpha/2)
                se_unpooled = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
                diff = p1 - p2
                L = diff - zcrit * se_unpooled
                U = diff + zcrit * se_unpooled

                st.markdown("### Confidence Interval")
                st.latex(r"(\hat p_1-\hat p_2)\pm z_{\alpha/2}\sqrt{\frac{\hat p_1(1-\hat p_1)}{n_1}+\frac{\hat p_2(1-\hat p_2)}{n_2}}")
                st.markdown(f"**CI ({100*(1-alpha):.0f}%):** ({L:.{decimals}f}, {U:.{decimals}f})")

    # ==========================================================
    # 2) PAIRED t-TEST USING DATA
    # ==========================================================
    if test_choice == "Paired t-Test using Data":
        st.subheader("📊 Paired t-Test using Raw Data")

        st.markdown("Upload CSV with columns **Sample1**, **Sample2** or enter manually.")
        up = st.file_uploader("Upload CSV", type="csv")
        s1 = st.text_area("Sample 1 (comma-separated)")
        s2 = st.text_area("Sample 2 (comma-separated)")

        if st.button("Calculate"):
            if up:
                df = pd.read_csv(up)
                if not {"Sample1","Sample2"}.issubset(df.columns):
                    st.error("CSV must contain Sample1 and Sample2.")
                    return
                a = df["Sample1"].to_numpy(dtype=float)
                b = df["Sample2"].to_numpy(dtype=float)
            else:
                a = np.array([float(i) for i in s1.split(",") if i.strip()!=""])
                b = np.array([float(i) for i in s2.split(",") if i.strip()!=""])

            if len(a) != len(b):
                st.error("Samples must have same length.")
                return

            d = a - b
            n = len(d)
            mean_d = np.mean(d)
            sd_d = np.std(d, ddof=1)
            se = sd_d / np.sqrt(n)
            tstat = mean_d / se
            dfree = n - 1

            st.markdown("### 📘 Step-by-Step")
            step_box("**Step 1:** Compute differences and summary statistics.")
            st.latex(r"d_i=x_{1i}-x_{2i}")
            st.latex(fr"\bar d={mean_d:.{decimals}f},\; s_d={sd_d:.{decimals}f},\; n={n}")

            step_box("**Step 2:** Compute test statistic.")
            st.latex(r"t=\frac{\bar d}{s_d/\sqrt{n}}")
            st.latex(fr"t={tstat:.{decimals}f},\; df={dfree}")

            step_box("**Step 3:** Compute p-value and critical value(s).")
            p_val, reject, crit_str = t_tail_metrics(tstat, dfree, alpha, tails)

            decision = "✔ Reject H₀" if reject else "✖ Fail to reject H₀"

            st.markdown(f"""
### **Result Summary**
- Mean difference: {mean_d:.{decimals}f}  
- SD of differences: {sd_d:.{decimals}f}  
- Test Statistic (t): {tstat:.{decimals}f}  
- df = {dfree}  
- Critical Value(s): {crit_str}  
- P-value: {p_val:.{decimals}f}  
- **Decision:** {decision}
""")

    # ==========================================================
    # 3) PAIRED t-TEST USING SUMMARY STATISTICS (WITH TEACHING TABLE)
    # ==========================================================
    if test_choice == "Paired t-Test using Summary Statistics":
        st.subheader("📋 Paired t-Test (Summary Statistics)")

        mean_d = st.number_input("Mean of differences (d̄)", value=0.0)
        sd_d = st.number_input("Std dev of differences (s_d)", value=1.0)
        n = st.number_input("Sample size (n)", min_value=2, step=1)

        if st.button("Calculate"):
            dfree = n - 1
            se = sd_d / np.sqrt(n)
            tstat = mean_d / se

            st.markdown("### 📘 Step-by-Step")
            step_box("**Step 1:** Understanding d-values (teaching illustration).")

            st.markdown("""
| i | dᵢ |
|---|----|
| 1 | d₁ |
| 2 | d₂ |
| … | … |
| n | dₙ |
""")

            step_box("**Step 2:** Compute SE and test statistic.")
            st.latex(r"SE=\frac{s_d}{\sqrt{n}}")
            st.latex(r"t=\frac{\bar d}{SE}")
            st.latex(fr"SE={se:.{decimals}f},\; t={tstat:.{decimals}f},\; df={dfree}")

            p_val, reject, crit_str = t_tail_metrics(tstat, dfree, alpha, tails)

            decision = "✔ Reject H₀" if reject else "✖ Fail to reject H₀"

            st.markdown(f"""
### **Result Summary**
- d̄ = {mean_d:.{decimals}f}  
- s_d = {sd_d:.{decimals}f}  
- Test Statistic (t): {tstat:.{decimals}f}  
- df = {dfree}  
- Critical Value(s): {crit_str}  
- P-value: {p_val:.{decimals}f}  
- **Decision:** {decision}
""")

    # ==========================================================
    # 4–7) (Remaining tests unchanged in logic, but with identical styling)
    # ==========================================================
    # NOTE: To save message space, I will deliver the remaining tests
    # (Welch t-tests and F-tests, both data + summary forms)
    # in the **next message**, in the same single code block,
    # seamlessly continuing (no repetition, no breaks).
    # Your final file will be complete and ready to paste.

# ==========================================================
# Run App
# ==========================================================
if __name__ == "__main__":
    run_two_sample_tool()
