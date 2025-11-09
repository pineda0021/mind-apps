# ==========================================================
# two_sample_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

# ---------- UI Helper ----------
def step_box(text: str):
    st.markdown(
        f"""
        <div style="background-color:#f0f6ff;padding:10px;border-radius:10px;
        border-left:5px solid #007acc;margin-bottom:10px;">
        <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_two_sample_tool():
    st.header("👨🏻‍🔬 Two-Sample Inference")

    # --- Updated Dropdown Style ---
    test_choice = st.selectbox(
        "Choose a Two-Sample Test:",
        [
            "Two-Proportion Z-Test",
            "Confidence Interval for Proportion Difference",
            "Paired t-Test using Data",
            "Paired Confidence Interval using Data",
            "Paired t-Test using Summary Statistics",
            "Paired Confidence Interval using Summary Statistics",
            "Independent t-Test using Data",
            "Independent Confidence Interval using Data",
            "Independent t-Test using Summary Statistics",
            "Independent Confidence Interval using Summary Statistics",
            "F-Test for Standard Deviation using Data",
            "F-Test for Standard Deviation using Summary Statistics"
        ],
        index=None,
        placeholder="Select a two-sample test to begin..."
    )

    if not test_choice:
        st.info("👆 Please select a two-sample test to begin.")
        return

    alpha = st.number_input("Significance level (α)", value=0.05, min_value=0.001, max_value=0.5, step=0.01)

    # ==========================================================
    # TWO-PROPORTION: Z-Test & CI (two-sided, as in your original)
    # ==========================================================
    if test_choice in ["Two-Proportion Z-Test", "Confidence Interval for Proportion Difference"]:
        st.subheader("📊 Enter Sample Data")
        x1 = st.number_input("Number of successes in Sample 1 (x₁)", min_value=0, step=1)
        n1 = st.number_input("Sample size 1 (n₁)", min_value=1, step=1)
        x2 = st.number_input("Number of successes in Sample 2 (x₂)", min_value=0, step=1)
        n2 = st.number_input("Sample size 2 (n₂)", min_value=1, step=1)

        if st.button("👨‍💻 Calculate"):
            p1 = x1 / n1
            p2 = x2 / n2

            st.markdown("### 📘 Step-by-Step Solution")

            # Z-Test (pooled SE, two-sided)
            if test_choice == "Two-Proportion Z-Test":
                step_box("**Step 1:** Compute sample proportions and pooled proportion.")
                st.latex(r"\hat{p}_1=\frac{x_1}{n_1},\quad \hat{p}_2=\frac{x_2}{n_2},\quad \hat{p}=\frac{x_1+x_2}{n_1+n_2}")
                p_pool = (x1 + x2) / (n1 + n2)
                st.latex(fr"\hat p_1=\frac{{{x1}}}{{{n1}}}={p1:.4f},\;\;\hat p_2=\frac{{{x2}}}{{{n2}}}={p2:.4f},\;\;\hat p=\frac{{{x1}+{x2}}}{{{n1}+{n2}}}={p_pool:.4f}")

                step_box("**Step 2:** Compute standard error and test statistic (two-sided).")
                st.latex(r"z=\frac{(\hat p_1-\hat p_2)}{\sqrt{\hat p(1-\hat p)\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}}")
                se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
                z_stat = (p1 - p2) / se
                st.latex(fr"\text{{SE}} = \sqrt{{{p_pool:.4f}(1-{p_pool:.4f})\!\left(\frac1{{{n1}}}+\frac1{{{n2}}}\right)}} = {se:.6f}")
                st.latex(fr"z=\frac{{{p1:.4f}-{p2:.4f}}}{{{se:.6f}}} = {z_stat:.4f}")

                step_box("**Step 3:** Compute p-value and critical value (two-sided).")
                z_crit = stats.norm.ppf(1 - alpha/2)
                p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                st.latex(fr"z_{{\alpha/2}} = z_{{{1-alpha/2:.3f}}} = {z_crit:.4f},\quad p\text{{-value}}=2\big(1-\Phi(|z|)\big) = {p_val:.4f}")

                step_box(f"**Step 4:** Compare p-value = {p_val:.4f} with α = {alpha:.2f} and conclude.")
                decision = "✅ Reject the null hypothesis." if abs(z_stat) > z_crit else "❌ Do not reject the null hypothesis."

                st.markdown(f"""
**Result Summary**

- p̂₁ = {p1:.4f}, p̂₂ = {p2:.4f}, pooled p̂ = {p_pool:.4f}  
- Test Statistic (z): {z_stat:.4f}  
- Critical Value: ±{z_crit:.4f}  
- P-value: {p_val:.4f}  
- Decision: **{decision}**

**Interpretation:**  
If p-value < α → Reject H₀; otherwise, fail to reject H₀.
""")

            # CI (unpooled SE, two-sided)
            else:
                step_box("**Step 1:** Compute sample proportions and unpooled standard error.")
                st.latex(r"(\hat p_1-\hat p_2)\ \pm\ z_{\alpha/2}\ \sqrt{\frac{\hat p_1(1-\hat p_1)}{n_1}+\frac{\hat p_2(1-\hat p_2)}{n_2}}")
                z_crit = stats.norm.ppf(1 - alpha/2)
                se_unpooled = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
                diff = p1 - p2
                ci_lower = diff - z_crit * se_unpooled
                ci_upper = diff + z_crit * se_unpooled
                st.latex(fr"z_{{\alpha/2}} = {z_crit:.4f},\;\; \text{{SE}} = {se_unpooled:.6f}")

                step_box("**Step 2:** Construct the confidence interval.")
                st.latex(fr"\big({diff:.4f} - {z_crit:.4f}\cdot {se_unpooled:.6f},\; {diff:.4f} + {z_crit:.4f}\cdot {se_unpooled:.6f}\big) = ({ci_lower:.4f},\; {ci_upper:.4f})")

                st.markdown(f"**Confidence Interval ({100*(1-alpha):.0f}%):**  ({ci_lower:.4f}, {ci_upper:.4f})")

    # ==========================================================
    # PAIRED t: Data & Summary
    # ==========================================================
    elif test_choice in ["Paired t-Test using Data", "Paired Confidence Interval using Data"]:
        st.subheader("📊 Enter Paired Data")
        st.write("Option 1: Upload CSV with two columns: **Sample1**, **Sample2**")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")

        st.write("Option 2: Enter data manually (comma-separated)")
        sample1_input = st.text_area("Sample 1", placeholder="1.2, 2.3, 3.1, 4.5")
        sample2_input = st.text_area("Sample 2", placeholder="0.9, 2.1, 3.0, 4.2")

        if st.button("👨‍💻 Calculate"):
            # Load
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                if {"Sample1", "Sample2"} - set(df.columns):
                    st.error("CSV must include columns named 'Sample1' and 'Sample2'.")
                    return
                x1 = df["Sample1"].to_numpy(dtype=float)
                x2 = df["Sample2"].to_numpy(dtype=float)
            else:
                try:
                    x1 = np.array([float(i.strip()) for i in sample1_input.split(",") if i.strip()!=""])
                    x2 = np.array([float(i.strip()) for i in sample2_input.split(",") if i.strip()!=""])
                except:
                    st.error("❌ Invalid data. Make sure values are numeric and comma-separated.")
                    return

            if len(x1) != len(x2):
                st.error("❌ Paired samples must have the same length.")
                return

            d = x1 - x2
            n = len(d)
            mean_d = np.mean(d)
            sd_d = np.std(d, ddof=1)
            se = sd_d / np.sqrt(n)
            t_stat = mean_d / se
            df = n - 1
            t_crit = stats.t.ppf(1 - alpha/2, df)
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Define paired differences and compute summary stats.")
            st.latex(r"d_i = x_{1i} - x_{2i},\quad \bar d=\frac1n\sum d_i,\quad s_d=\sqrt{\frac{\sum(d_i-\bar d)^2}{n-1}}")
            st.latex(fr"\bar d={mean_d:.4f},\;\; s_d={sd_d:.4f},\;\; n={n}")

            if "t-Test" in test_choice:
                step_box("**Step 2:** Compute the test statistic (two-sided).")
                st.latex(r"t=\frac{\bar d - 0}{s_d/\sqrt{n}}")
                st.latex(fr"t=\frac{{{mean_d:.4f}}}{{{sd_d:.4f}/\sqrt{{{n}}}}} = {t_stat:.4f}")

                step_box("**Step 3:** Compute p-value and critical value.")
                st.latex(fr"t_{{\alpha/2,\,{df}}}={t_crit:.4f},\;\; p\text{{-value}}=2\big(1-F_T(|t|;{df})\big)={p_val:.4f}")

                step_box(f"**Step 4:** Compare p-value = {p_val:.4f} with α = {alpha:.2f} and conclude.")
                decision = "✅ Reject the null hypothesis." if abs(t_stat) > t_crit else "❌ Do not reject the null hypothesis."

                st.markdown(f"""
**Result Summary**

- Mean difference (d̄): {mean_d:.4f}  
- SD of differences (s_d): {sd_d:.4f}  
- Test Statistic (t): {t_stat:.4f} (df = {df})  
- Critical Value: ±{t_crit:.4f}  
- P-value: {p_val:.4f}  
- Decision: **{decision}**
""")

            else:
                step_box("**Step 2:** Construct the confidence interval for μ_d.")
                st.latex(r"\bar d \pm t_{\alpha/2,\,n-1}\ \frac{s_d}{\sqrt{n}}")
                ci_lower = mean_d - t_crit * se
                ci_upper = mean_d + t_crit * se
                st.latex(fr"({mean_d:.4f} \pm {t_crit:.4f}\cdot {se:.6f}) = ({ci_lower:.4f},\; {ci_upper:.4f})")
                st.markdown(f"**Confidence Interval ({100*(1-alpha):.0f}%):**  ({ci_lower:.4f}, {ci_upper:.4f})")

    elif test_choice in ["Paired t-Test using Summary Statistics", "Paired Confidence Interval using Summary Statistics"]:
        st.subheader("📋 Enter Summary Statistics of Differences (d = x₁ - x₂)")
        mean_diff = st.number_input("Mean of differences (d̄)", value=0.0, format="%.6f")
        sd_diff = st.number_input("Std Dev of differences (s_d)", value=1.0, format="%.6f")
        n = st.number_input("Sample size (n)", min_value=2, step=1)

        if st.button("👨‍💻 Calculate"):
            df = n - 1
            se = sd_diff / np.sqrt(n)
            t_stat = mean_diff / se
            t_crit = stats.t.ppf(1 - alpha/2, df)
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Compute test statistic / standard error.")
            st.latex(r"t=\frac{\bar d - 0}{s_d/\sqrt{n}},\qquad \text{SE}=\frac{s_d}{\sqrt{n}}")
            st.latex(fr"t=\frac{{{mean_diff:.4f}}}{{{sd_diff:.4f}/\sqrt{{{n}}}}}={t_stat:.4f},\quad \text{{SE}}={se:.6f}")

            if "t-Test" in test_choice:
                step_box("**Step 2:** Compute p-value and critical value (two-sided).")
                st.latex(fr"t_{{\alpha/2,\,{df}}}={t_crit:.4f},\;\; p\text{{-value}}={p_val:.4f}")
                step_box(f"**Step 3:** Compare p-value = {p_val:.4f} with α = {alpha:.2f} and conclude.")
                decision = "✅ Reject the null hypothesis." if abs(t_stat) > t_crit else "❌ Do not reject the null hypothesis."

                st.markdown(f"""
**Result Summary**

- d̄ = {mean_diff:.4f}, s_d = {sd_diff:.4f}, n = {n}  
- Test Statistic (t): {t_stat:.4f} (df = {df})  
- Critical Value: ±{t_crit:.4f}  
- P-value: {p_val:.4f}  
- Decision: **{decision}**
""")
            else:
                step_box("**Step 2:** Construct the confidence interval for μ_d.")
                st.latex(r"\bar d \pm t_{\alpha/2,\,n-1}\ \frac{s_d}{\sqrt{n}}")
                ci_lower = mean_diff - t_crit * se
                ci_upper = mean_diff + t_crit * se
                st.latex(fr"({mean_diff:.4f} \pm {t_crit:.4f}\cdot {se:.6f}) = ({ci_lower:.4f},\; {ci_upper:.4f})")
                st.markdown(f"**Confidence Interval ({100*(1-alpha):.0f}%):**  ({ci_lower:.4f}, {ci_upper:.4f})")

    # ==========================================================
    # INDEPENDENT t: Data & Summary (Welch by default)
    # ==========================================================
    elif test_choice in ["Independent t-Test using Data", "Independent Confidence Interval using Data"]:
        st.subheader("📊 Enter Independent Samples Data")
        st.write("Option 1: Upload CSV with columns: **Sample1**, **Sample2**")
        uploaded_file = st.file_uploader("Upload CSV", type="csv", key="indep_csv")

        st.write("Option 2: Enter data manually (comma-separated)")
        sample1_input = st.text_area("Sample 1", placeholder="1.2, 2.3, 3.1, 4.5", key="indep1")
        sample2_input = st.text_area("Sample 2", placeholder="0.9, 2.1, 3.0, 4.2", key="indep2")

        if st.button("👨‍💻 Calculate", key="indep_calc"):
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                if {"Sample1", "Sample2"} - set(df.columns):
                    st.error("CSV must include columns named 'Sample1' and 'Sample2'.")
                    return
                x1 = df["Sample1"].to_numpy(dtype=float)
                x2 = df["Sample2"].to_numpy(dtype=float)
            else:
                try:
                    x1 = np.array([float(i.strip()) for i in sample1_input.split(",") if i.strip()!=""])
                    x2 = np.array([float(i.strip()) for i in sample2_input.split(",") if i.strip()!=""])
                except:
                    st.error("❌ Invalid data format. Make sure values are numeric and comma-separated.")
                    return

            n1, n2 = len(x1), len(x2)
            mean1, mean2 = np.mean(x1), np.mean(x2)
            s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            # Welch-Satterthwaite df
            df_deg = ((s1**2/n1 + s2**2/n2)**2) / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
            t_stat = (mean1 - mean2) / se
            t_crit = stats.t.ppf(1 - alpha/2, df=df_deg)
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_deg))

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Compute sample means, SDs, and the standard error (Welch).")
            st.latex(r"\text{SE}=\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}")
            st.latex(fr"\bar x_1={mean1:.4f},\; \bar x_2={mean2:.4f},\; s_1={s1:.4f},\; s_2={s2:.4f},\; \text{{SE}}={se:.6f}")

            if "t-Test" in test_choice:
                step_box("**Step 2:** Compute test statistic and degrees of freedom (Welch).")
                st.latex(r"t=\frac{(\bar x_1-\bar x_2)}{\text{SE}}")
                st.latex(fr"t=\frac{{{mean1:.4f}-{mean2:.4f}}}{{{se:.6f}}} = {t_stat:.4f}")
                st.latex(r"""
                \nu \approx 
                \frac{\left(\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}\right)^2}
                     {\frac{\left(\frac{s_1^2}{n_1}\right)^2}{n_1-1} + \frac{\left(\frac{s_2^2}{n_2}\right)^2}{n_2-1}}
                """)
                step_box("**Step 3:** Compute p-value and critical value (two-sided).")
                st.latex(fr"t_{{\alpha/2,\,\nu}}={t_crit:.4f},\;\; p\text{{-value}}={p_val:.4f}")

                step_box(f"**Step 4:** Compare p-value = {p_val:.4f} with α = {alpha:.2f} and conclude.")
                decision = "✅ Reject the null hypothesis." if abs(t_stat) > t_crit else "❌ Do not reject the null hypothesis."

                st.markdown(f"""
**Result Summary**

- \u0305x₁ = {mean1:.4f}, \u0305x₂ = {mean2:.4f}, s₁ = {s1:.4f}, s₂ = {s2:.4f}  
- Test Statistic (t): {t_stat:.4f} (df ≈ {df_deg:.2f})  
- Critical Value: ±{t_crit:.4f}  
- P-value: {p_val:.4f}  
- Decision: **{decision}**
""")
            else:
                step_box("**Step 2:** Construct the confidence interval for (μ₁−μ₂).")
                st.latex(r"(\bar x_1-\bar x_2)\ \pm\ t_{\alpha/2,\,\nu}\ \text{SE}")
                ci_lower = (mean1 - mean2) - t_crit * se
                ci_upper = (mean1 - mean2) + t_crit * se
                st.latex(fr"(({mean1:.4f}-{mean2:.4f}) \pm {t_crit:.4f}\cdot {se:.6f}) = ({ci_lower:.4f},\; {ci_upper:.4f})")
                st.markdown(f"**Confidence Interval ({100*(1-alpha):.0f}%):**  ({ci_lower:.4f}, {ci_upper:.4f})")

    elif test_choice in ["Independent t-Test using Summary Statistics", "Independent Confidence Interval using Summary Statistics"]:
        st.subheader("📋 Enter Summary Statistics")
        mean1 = st.number_input("Mean of Sample 1 (x̄₁)", value=0.0, format="%.6f")
        s1 = st.number_input("Std Dev of Sample 1 (s₁)", value=1.0, format="%.6f")
        n1 = st.number_input("Sample size 1 (n₁)", min_value=2, step=1)
        mean2 = st.number_input("Mean of Sample 2 (x̄₂)", value=0.0, format="%.6f")
        s2 = st.number_input("Std Dev of Sample 2 (s₂)", value=1.0, format="%.6f")
        n2 = st.number_input("Sample size 2 (n₂)", min_value=2, step=1)

        if st.button("👨‍💻 Calculate"):
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            df_deg = ((s1**2/n1 + s2**2/n2)**2) / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
            diff = mean1 - mean2
            t_stat = diff / se
            t_crit = stats.t.ppf(1 - alpha/2, df_deg)
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_deg))

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Compute standard error and test statistic (Welch).")
            st.latex(r"\text{SE}=\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}},\quad t=\frac{(\bar x_1-\bar x_2)}{\text{SE}}")
            st.latex(fr"\text{{SE}}={se:.6f},\quad t=\frac{{{diff:.4f}}}{{{se:.6f}}}={t_stat:.4f}")

            if "t-Test" in test_choice:
                step_box("**Step 2:** Compute p-value and critical value (two-sided).")
                st.latex(fr"t_{{\alpha/2,\,\nu}}={t_crit:.4f},\;\; p\text{{-value}}={p_val:.4f}")
                step_box(f"**Step 3:** Compare p-value = {p_val:.4f} with α = {alpha:.2f} and conclude.")
                decision = "✅ Reject the null hypothesis." if abs(t_stat) > t_crit else "❌ Do not reject the null hypothesis."

                st.markdown(f"""
**Result Summary**

- Provided: x̄₁ = {mean1:.4f}, x̄₂ = {mean2:.4f}, s₁ = {s1:.4f}, s₂ = {s2:.4f}, n₁ = {n1}, n₂ = {n2}  
- Test Statistic (t): {t_stat:.4f} (df ≈ {df_deg:.2f})  
- Critical Value: ±{t_crit:.4f}  
- P-value: {p_val:.4f}  
- Decision: **{decision}**
""")
            else:
                step_box("**Step 2:** Construct the confidence interval for (μ₁−μ₂).")
                st.latex(r"(\bar x_1-\bar x_2)\ \pm\ t_{\alpha/2,\,\nu}\ \text{SE}")
                ci_lower = diff - t_crit * se
                ci_upper = diff + t_crit * se
                st.latex(fr"({diff:.4f} \pm {t_crit:.4f}\cdot {se:.6f}) = ({ci_lower:.4f},\; {ci_upper:.4f})")
                st.markdown(f"**Confidence Interval ({100*(1-alpha):.0f}%):**  ({ci_lower:.4f}, {ci_upper:.4f})")

    # ==========================================================
    # F-Test (two-sided) : Data & Summary
    # ==========================================================
    elif test_choice in ["F-Test for Standard Deviation using Data", "F-Test for Standard Deviation using Summary Statistics"]:
        st.subheader("📊 F-Test Input")
        use_data = test_choice.endswith("Data")

        if use_data:
            st.write("Option 1: Upload CSV with columns: **Sample1**, **Sample2**")
            uploaded_file = st.file_uploader("Upload CSV", type="csv", key="f_csv")

            st.write("Option 2: Enter data manually (comma-separated)")
            sample1_input = st.text_area("Sample 1", placeholder="1.2, 2.3, 3.1, 4.5", key="f1")
            sample2_input = st.text_area("Sample 2", placeholder="0.9, 2.1, 3.0, 4.2", key="f2")
        else:
            n1 = st.number_input("Sample size 1 (n₁)", min_value=2, step=1)
            s1 = st.number_input("Std Dev of Sample 1 (s₁)", value=1.0, format="%.6f")
            n2 = st.number_input("Sample size 2 (n₂)", min_value=2, step=1)
            s2 = st.number_input("Std Dev of Sample 2 (s₂)", value=1.0, format="%.6f")

        if st.button("👨‍💻 Calculate", key="f_calc"):
            if use_data:
                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    if {"Sample1", "Sample2"} - set(df.columns):
                        st.error("CSV must include columns named 'Sample1' and 'Sample2'.")
                        return
                    x1 = df["Sample1"].to_numpy(dtype=float)
                    x2 = df["Sample2"].to_numpy(dtype=float)
                else:
                    try:
                        x1 = np.array([float(i.strip()) for i in sample1_input.split(",") if i.strip()!=""])
                        x2 = np.array([float(i.strip()) for i in sample2_input.split(",") if i.strip()!=""])
                    except:
                        st.error("❌ Invalid manual data format.")
                        return
                n1, n2 = len(x1), len(x2)
                s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)

            F = (s1**2) / (s2**2)
            df1, df2 = n1 - 1, n2 - 1
            F_crit_low = stats.f.ppf(alpha / 2, df1, df2)
            F_crit_high = stats.f.ppf(1 - alpha / 2, df1, df2)
            p_val = 2 * min(stats.f.cdf(F, df1, df2), 1 - stats.f.cdf(F, df1, df2))

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Compute the F statistic and degrees of freedom.")
            st.latex(r"F=\frac{s_1^2}{s_2^2},\quad \text{df}_1=n_1-1,\ \text{df}_2=n_2-1")
            st.latex(fr"F=\frac{{{s1:.4f}^2}}{{{s2:.4f}^2}}={F:.4f},\quad \text{{df}}_1={df1},\ \text{{df}}_2={df2}")

            step_box("**Step 2:** Determine two-sided critical values and p-value.")
            st.latex(fr"F_{{\alpha/2}}={F_crit_low:.4f},\;\; F_{{1-\alpha/2}}={F_crit_high:.4f},\;\; p\text{{-value}}={p_val:.4f}")

            step_box(f"**Step 3:** Compare p-value = {p_val:.4f} with α = {alpha:.2f} and conclude.")
            decision = "✅ Reject the null hypothesis." if F < F_crit_low or F > F_crit_high else "❌ Do not reject the null hypothesis."

            st.markdown(f"""
**Result Summary**

- F statistic: {F:.4f} (df₁ = {df1}, df₂ = {df2})  
- Critical Values: ({F_crit_low:.4f}, {F_crit_high:.4f})  
- P-value: {p_val:.4f}  
- Decision: **{decision}**
""")


# ---------- Run ----------
if __name__ == "__main__":
    run_two_sample_tool()

