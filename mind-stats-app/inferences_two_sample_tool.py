# ==========================================================
# two_sample_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# Built with the students in MIND
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
    else:  # two
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
    else:  # two
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
    else:  # two
        crit_low = stats.f.ppf(alpha/2, df1, df2)
        crit_high = stats.f.ppf(1 - alpha/2, df1, df2)
        p = 2 * min(stats.f.cdf(F, df1, df2), 1 - stats.f.cdf(F, df1, df2))
        p = float(min(1.0, p))
        reject = (F < crit_low) or (F > crit_high)
        crit_str = f"({crit_low:.4f}, {crit_high:.4f})"
    return p, reject, crit_str

# ---------- Main ----------
def run_two_sample_tool():
    st.header("👨🏻‍🔬 Two-Sample Inference")
    st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — Built with the students in MIND")

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
        placeholder="Select a two-sample test to begin..."
    )

    if not test_choice:
        st.info("👆 Please select a two-sample test to begin.")
        return

    alpha = st.number_input("Significance level (α)", value=0.05, min_value=0.001, max_value=0.5, step=0.01)
    tails = st.selectbox("Tail type:", ["two", "left", "right"])
    show_ci = st.checkbox("Show Confidence Interval (two-sided)")

    # ==========================================================
    # TWO-PROPORTION Z-TEST
    # ==========================================================
    if test_choice == "Two-Proportion Z-Test":
        st.subheader("📊 Enter Sample Data (Counts)")
        x1 = st.number_input("Successes in Sample 1 (x₁)", min_value=0, step=1)
        n1 = st.number_input("Sample size 1 (n₁)", min_value=1, step=1)
        x2 = st.number_input("Successes in Sample 2 (x₂)", min_value=0, step=1)
        n2 = st.number_input("Sample size 2 (n₂)", min_value=1, step=1)

        if st.button("👨‍💻 Calculate"):
            p1, p2 = x1 / n1, x2 / n2
            p_pool = (x1 + x2) / (n1 + n2)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            z_stat = (p1 - p2) / se

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Compute sample proportions and pooled proportion.")
            st.latex(r"\hat{p}_1=\frac{x_1}{n_1},\;\hat{p}_2=\frac{x_2}{n_2},\;\hat{p}=\frac{x_1+x_2}{n_1+n_2}")
            st.latex(fr"\hat p_1={p1:.4f},\;\hat p_2={p2:.4f},\;\hat p={p_pool:.4f}")

            step_box("**Step 2:** Compute standard error and test statistic.")
            st.latex(r"z = \frac{\hat p_1-\hat p_2}{\sqrt{\hat p(1-\hat p)\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}}")
            st.latex(fr"\text{{SE}}={se:.6f},\;\; z={z_stat:.4f}")

            step_box("**Step 3:** Tail-specific p-value and critical region.")
            if tails == "left":
                st.latex(r"H_1: p_1-p_2 < 0")
            elif tails == "right":
                st.latex(r"H_1: p_1-p_2 > 0")
            else:
                st.latex(r"H_1: p_1-p_2 \ne 0")

            p_val, reject, crit_str = z_tail_metrics(z_stat, alpha, tails)

            step_box(f"**Step 4:** Compare p-value = {p_val:.4f} with α = {alpha:.2f} and conclude.")
            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"

            st.markdown(f"""
**Result Summary**

- p̂₁ = {p1:.4f}, p̂₂ = {p2:.4f}, pooled p̂ = {p_pool:.4f}  
- Test Statistic (z): {z_stat:.4f}  
- Critical Value(s): {crit_str}  
- P-value: {p_val:.4f}  
- Decision: **{decision}**
""")

            # Confidence Interval (two-sided, unpooled SE)
            if show_ci:
                z_crit = stats.norm.ppf(1 - alpha/2)
                se_unpooled = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
                diff = p1 - p2
                ci_lower = diff - z_crit * se_unpooled
                ci_upper = diff + z_crit * se_unpooled
                st.markdown("### 🧾 Confidence Interval (two-sided)")
                st.latex(r"(\hat p_1-\hat p_2) \pm z_{\alpha/2}\sqrt{\frac{\hat p_1(1-\hat p_1)}{n_1}+\frac{\hat p_2(1-\hat p_2)}{n_2}}")
                st.markdown(f"**CI ({100*(1-alpha):.0f}%):** ({ci_lower:.4f}, {ci_upper:.4f})")

    # ==========================================================
    # PAIRED t-TEST (DATA)
    # ==========================================================
    elif test_choice == "Paired t-Test using Data":
        st.subheader("📊 Enter Paired Data")
        st.write("Upload CSV with two columns: **Sample1**, **Sample2**, or enter manually.")
        up = st.file_uploader("Upload CSV", type="csv")
        s1 = st.text_area("Sample 1 (comma-separated)", "1.2, 2.3, 3.1, 4.5")
        s2 = st.text_area("Sample 2 (comma-separated)", "0.9, 2.1, 3.0, 4.2")

        if st.button("👨‍💻 Calculate"):
            if up is not None:
                df = pd.read_csv(up)
                if {"Sample1","Sample2"} - set(df.columns):
                    st.error("CSV must include 'Sample1' and 'Sample2'.")
                    return
                x1 = df["Sample1"].to_numpy(dtype=float)
                x2 = df["Sample2"].to_numpy(dtype=float)
            else:
                try:
                    x1 = np.array([float(i.strip()) for i in s1.split(",") if i.strip()!=""])
                    x2 = np.array([float(i.strip()) for i in s2.split(",") if i.strip()!=""])
                except:
                    st.error("Invalid manual data.")
                    return
            if len(x1) != len(x2):
                st.error("Paired samples must have the same length.")
                return

            d = x1 - x2
            n = len(d)
            mean_d = np.mean(d)
            sd_d = np.std(d, ddof=1)
            se = sd_d / np.sqrt(n)
            t_stat = mean_d / se
            dfree = n - 1

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Define differences and compute summary stats.")
            st.latex(r"d_i = x_{1i}-x_{2i},\;\;\bar d,\; s_d,\; n")
            st.latex(fr"\bar d={mean_d:.4f},\; s_d={sd_d:.4f},\; n={n}")

            step_box("**Step 2:** Test statistic for paired t.")
            st.latex(r"t=\frac{\bar d - 0}{s_d/\sqrt{n}}")
            st.latex(fr"t={t_stat:.4f},\; \text{{df}}={dfree}")

            step_box("**Step 3:** Tail-specific p-value and critical region.")
            if tails == "left":
                st.latex(r"H_1:\ \mu_d<0")
            elif tails == "right":
                st.latex(r"H_1:\ \mu_d>0")
            else:
                st.latex(r"H_1:\ \mu_d\ne 0")

            p_val, reject, crit_str = t_tail_metrics(t_stat, dfree, alpha, tails)

            step_box(f"**Step 4:** Compare p-value = {p_val:.4f} with α = {alpha:.2f}.")
            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"

            st.markdown(f"""
**Result Summary**

- Mean diff (d̄): {mean_d:.4f}, SD of diffs (s_d): {sd_d:.4f}  
- Test Statistic (t): {t_stat:.4f} (df = {dfree})  
- Critical Value(s): {crit_str}  
- P-value: {p_val:.4f}  
- Decision: **{decision}**
""")

            if show_ci:
                t_crit = stats.t.ppf(1 - alpha/2, dfree)
                ci_lower = mean_d - t_crit * se
                ci_upper = mean_d + t_crit * se
                st.markdown("### 🧾 Confidence Interval (two-sided)")
                st.latex(r"\bar d \pm t_{\alpha/2,\,n-1}\frac{s_d}{\sqrt{n}}")
                st.markdown(f"**CI ({100*(1-alpha):.0f}%):** ({ci_lower:.4f}, {ci_upper:.4f})")

    # ==========================================================
    # PAIRED t-TEST (SUMMARY)
    # ==========================================================
    elif test_choice == "Paired t-Test using Summary Statistics":
        st.subheader("📋 Enter Summary Statistics of Differences (d = x₁ - x₂)")
        mean_d = st.number_input("Mean of differences (d̄)", value=0.0, format="%.6f")
        sd_d = st.number_input("Std Dev of differences (s_d)", value=1.0, format="%.6f")
        n = st.number_input("Sample size (n)", min_value=2, step=1)

        if st.button("👨‍💻 Calculate"):
            dfree = n - 1
            se = sd_d / np.sqrt(n)
            t_stat = mean_d / se

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Test statistic and SE.")
            st.latex(r"t=\frac{\bar d - 0}{s_d/\sqrt{n}},\qquad \text{SE}=\frac{s_d}{\sqrt{n}}")
            st.latex(fr"t={t_stat:.4f},\; \text{{SE}}={se:.6f},\; \text{{df}}={dfree}")

            step_box("**Step 2:** Tail-specific p-value and critical region.")
            if tails == "left":
                st.latex(r"H_1:\ \mu_d<0")
            elif tails == "right":
                st.latex(r"H_1:\ \mu_d>0")
            else:
                st.latex(r"H_1:\ \mu_d\ne 0")

            p_val, reject, crit_str = t_tail_metrics(t_stat, dfree, alpha, tails)
            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"

            st.markdown(f"""
**Result Summary**

- d̄ = {mean_d:.4f}, s_d = {sd_d:.4f}, n = {n}  
- Test Statistic (t): {t_stat:.4f} (df = {dfree})  
- Critical Value(s): {crit_str}  
- P-value: {p_val:.4f}  
- Decision: **{decision}**
""")

            if show_ci:
                t_crit = stats.t.ppf(1 - alpha/2, dfree)
                ci_lower = mean_d - t_crit * se
                ci_upper = mean_d + t_crit * se
                st.markdown("### 🧾 Confidence Interval (two-sided)")
                st.latex(r"\bar d \pm t_{\alpha/2,\,n-1}\frac{s_d}{\sqrt{n}}")
                st.markdown(f"**CI ({100*(1-alpha):.0f}%):** ({ci_lower:.4f}, {ci_upper:.4f})")

    # ==========================================================
    # INDEPENDENT t-TEST (DATA, WELCH)
    # ==========================================================
    elif test_choice == "Independent t-Test using Data (Welch)":
        st.subheader("📊 Enter Independent Samples Data")
        st.write("Upload CSV with **Sample1**, **Sample2** or enter manually.")
        up = st.file_uploader("Upload CSV", type="csv", key="indep_csv")
        s1 = st.text_area("Sample 1 (comma-separated)", "1.2, 2.3, 3.1, 4.5", key="indep1")
        s2 = st.text_area("Sample 2 (comma-separated)", "0.9, 2.1, 3.0, 4.2", key="indep2")

        if st.button("👨‍💻 Calculate", key="indep_calc"):
            if up is not None:
                df = pd.read_csv(up)
                if {"Sample1","Sample2"} - set(df.columns):
                    st.error("CSV must include 'Sample1' and 'Sample2'.")
                    return
                x1 = df["Sample1"].to_numpy(dtype=float)
                x2 = df["Sample2"].to_numpy(dtype=float)
            else:
                try:
                    x1 = np.array([float(i.strip()) for i in s1.split(",") if i.strip()!=""])
                    x2 = np.array([float(i.strip()) for i in s2.split(",") if i.strip()!=""])
                except:
                    st.error("Invalid manual data.")
                    return

            n1, n2 = len(x1), len(x2)
            mean1, mean2 = np.mean(x1), np.mean(x2)
            s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            t_stat = (mean1 - mean2) / se
            # Welch-Satterthwaite df
            df_deg = ((s1**2/n1 + s2**2/n2)**2) / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Compute means, SDs, and Welch SE.")
            st.latex(r"\text{SE}=\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}")
            st.latex(fr"\bar x_1={mean1:.4f},\; \bar x_2={mean2:.4f},\; s_1={s1:.4f},\; s_2={s2:.4f},\; \text{{SE}}={se:.6f}")

            step_box("**Step 2:** Test statistic and df (Welch).")
            st.latex(r"t=\frac{\bar x_1-\bar x_2}{\text{SE}}")
            st.latex(fr"t={t_stat:.4f},\; \nu\approx {df_deg:.2f}")

            step_box("**Step 3:** Tail-specific p-value and critical region.")
            if tails == "left":
                st.latex(r"H_1:\ \mu_1-\mu_2<0")
            elif tails == "right":
                st.latex(r"H_1:\ \mu_1-\mu_2>0")
            else:
                st.latex(r"H_1:\ \mu_1-\mu_2\ne 0")

            p_val, reject, crit_str = t_tail_metrics(t_stat, df_deg, alpha, tails)
            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"

            st.markdown(f"""
**Result Summary**

- \u0305x₁ = {mean1:.4f}, \u0305x₂ = {mean2:.4f}, s₁ = {s1:.4f}, s₂ = {s2:.4f}  
- Test Statistic (t): {t_stat:.4f} (df ≈ {df_deg:.2f})  
- Critical Value(s): {crit_str}  
- P-value: {p_val:.4f}  
- Decision: **{decision}**
""")

            if show_ci:
                t_crit = stats.t.ppf(1 - alpha/2, df_deg)
                diff = mean1 - mean2
                ci_lower = diff - t_crit * se
                ci_upper = diff + t_crit * se
                st.markdown("### 🧾 Confidence Interval (two-sided)")
                st.latex(r"(\bar x_1-\bar x_2)\ \pm\ t_{\alpha/2,\,\nu}\ \text{SE}")
                st.markdown(f"**CI ({100*(1-alpha):.0f}%):** ({ci_lower:.4f}, {ci_upper:.4f})")

    # ==========================================================
    # INDEPENDENT t-TEST (SUMMARY, WELCH)
    # ==========================================================
    elif test_choice == "Independent t-Test using Summary Statistics (Welch)":
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

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Compute SE and test statistic (Welch).")
            st.latex(r"\text{SE}=\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}},\quad t=\frac{\bar x_1-\bar x_2}{\text{SE}}")
            st.latex(fr"\text{{SE}}={se:.6f},\; t={t_stat:.4f},\; \nu\approx {df_deg:.2f}")

            step_box("**Step 2:** Tail-specific p-value and critical region.")
            if tails == "left":
                st.latex(r"H_1:\ \mu_1-\mu_2<0")
            elif tails == "right":
                st.latex(r"H_1:\ \mu_1-\mu_2>0")
            else:
                st.latex(r"H_1:\ \mu_1-\mu_2\ne 0")

            p_val, reject, crit_str = t_tail_metrics(t_stat, df_deg, alpha, tails)
            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"

            st.markdown(f"""
**Result Summary**

- Provided: x̄₁ = {mean1:.4f}, x̄₂ = {mean2:.4f}, s₁ = {s1:.4f}, s₂ = {s2:.4f}, n₁ = {n1}, n₂ = {n2}  
- Test Statistic (t): {t_stat:.4f} (df ≈ {df_deg:.2f})  
- Critical Value(s): {crit_str}  
- P-value: {p_val:.4f}  
- Decision: **{decision}**
""")

            if show_ci:
                t_crit = stats.t.ppf(1 - alpha/2, df_deg)
                ci_lower = diff - t_crit * se
                ci_upper = diff + t_crit * se
                st.markdown("### 🧾 Confidence Interval (two-sided)")
                st.latex(r"(\bar x_1-\bar x_2)\ \pm\ t_{\alpha/2,\,\nu}\ \text{SE}")
                st.markdown(f"**CI ({100*(1-alpha):.0f}%):** ({ci_lower:.4f}, {ci_upper:.4f})")

    # ==========================================================
    # F-TEST FOR STANDARD DEVIATIONS (DATA)
    # ==========================================================
    elif test_choice == "F-Test for Standard Deviations using Data":
        st.subheader("📊 Enter Independent Samples Data")
        st.write("Upload CSV with **Sample1**, **Sample2** or enter manually.")
        up = st.file_uploader("Upload CSV", type="csv", key="f_csv")
        s1 = st.text_area("Sample 1 (comma-separated)", "1.2, 2.3, 3.1, 4.5", key="f1")
        s2 = st.text_area("Sample 2 (comma-separated)", "0.9, 2.1, 3.0, 4.2", key="f2")

        if st.button("👨‍💻 Calculate", key="f_calc"):
            if up is not None:
                df = pd.read_csv(up)
                if {"Sample1","Sample2"} - set(df.columns):
                    st.error("CSV must include 'Sample1' and 'Sample2'.")
                    return
                x1 = df["Sample1"].to_numpy(dtype=float)
                x2 = df["Sample2"].to_numpy(dtype=float)
            else:
                try:
                    x1 = np.array([float(i.strip()) for i in s1.split(",") if i.strip()!=""])
                    x2 = np.array([float(i.strip()) for i in s2.split(",") if i.strip()!=""])
                except:
                    st.error("Invalid manual data.")
                    return

            n1, n2 = len(x1), len(x2)
            s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
            F = (s1**2) / (s2**2)
            df1, df2 = n1 - 1, n2 - 1

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Compute F and degrees of freedom.")
            st.latex(r"F=\frac{s_1^2}{s_2^2},\quad \text{df}_1=n_1-1,\ \text{df}_2=n_2-1")
            st.latex(fr"F={F:.4f},\; \text{{df}}_1={df1},\; \text{{df}}_2={df2}")

            step_box("**Step 2:** Tail-specific hypothesis and p-value.")
            if tails == "left":
                st.latex(r"H_1:\ \sigma_1^2<\sigma_2^2")
            elif tails == "right":
                st.latex(r"H_1:\ \sigma_1^2>\sigma_2^2")
            else:
                st.latex(r"H_1:\ \sigma_1^2\ne \sigma_2^2")

            p_val, reject, crit_str = f_tail_metrics(F, df1, df2, alpha, tails)

            step_box(f"**Step 3:** Compare p-value = {p_val:.4f} with α = {alpha:.2f}.")
            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"

            st.markdown(f"""
**Result Summary**

- F statistic: {F:.4f} (df₁ = {df1}, df₂ = {df2})  
- Critical Value(s): {crit_str}  
- P-value: {p_val:.4f}  
- Decision: **{decision}**
""")

            if show_ci:
                # CI for variance ratio σ1^2/σ2^2 (two-sided)
                f_low = stats.f.ppf(1 - alpha/2, df1, df2)
                f_high = stats.f.ppf(alpha/2, df1, df2)  # reciprocal bound form
                ratio = (s1**2) / (s2**2)
                ci_lower = ratio / f_low
                ci_upper = ratio / stats.f.ppf(alpha/2, df1, df2) if alpha/2 > 0 else np.inf
                # safer: classic CI uses ( (s1^2/s2^2) / F_{1-α/2,df1,df2}, (s1^2/s2^2) / F_{α/2,df1,df2} )
                ci_lower = ratio / stats.f.ppf(1 - alpha/2, df1, df2)
                ci_upper = ratio / stats.f.ppf(alpha/2, df1, df2)
                st.markdown("### 🧾 Confidence Interval for Variance Ratio (two-sided)")
                st.latex(r"\left(\frac{s_1^2/s_2^2}{F_{1-\alpha/2,\ df_1, df_2}},\ \frac{s_1^2/s_2^2}{F_{\alpha/2,\ df_1, df_2}}\right)")
                st.markdown(f"**CI ({100*(1-alpha):.0f}%):** ({ci_lower:.4f}, {ci_upper:.4f})")

    # ==========================================================
    # F-TEST FOR STANDARD DEVIATIONS (SUMMARY)
    # ==========================================================
    elif test_choice == "F-Test for Standard Deviations using Summary Statistics":
        st.subheader("📋 Enter Summary Statistics")
        n1 = st.number_input("Sample size 1 (n₁)", min_value=2, step=1)
        s1 = st.number_input("Std Dev of Sample 1 (s₁)", value=1.0, format="%.6f")
        n2 = st.number_input("Sample size 2 (n₂)", min_value=2, step=1)
        s2 = st.number_input("Std Dev of Sample 2 (s₂)", value=1.0, format="%.6f")

        if st.button("👨‍💻 Calculate", key="f_summary"):
            F = (s1**2) / (s2**2)
            df1, df2 = n1 - 1, n2 - 1

            st.markdown("### 📘 Step-by-Step Solution")
            step_box("**Step 1:** Compute F and degrees of freedom.")
            st.latex(r"F=\frac{s_1^2}{s_2^2},\quad \text{df}_1=n_1-1,\ \text{df}_2=n_2-1")
            st.latex(fr"F={F:.4f},\; \text{{df}}_1={df1},\; \text{{df}}_2={df2}")

            step_box("**Step 2:** Tail-specific hypothesis and p-value.")
            if tails == "left":
                st.latex(r"H_1:\ \sigma_1^2<\sigma_2^2")
            elif tails == "right":
                st.latex(r"H_1:\ \sigma_1^2>\sigma_2^2")
            else:
                st.latex(r"H_1:\ \sigma_1^2\ne \sigma_2^2")

            p_val, reject, crit_str = f_tail_metrics(F, df1, df2, alpha, tails)
            decision = "✅ Reject H₀" if reject else "❌ Do not reject H₀"

            st.markdown(f"""
**Result Summary**

- F statistic: {F:.4f} (df₁ = {df1}, df₂ = {df2})  
- Critical Value(s): {crit_str}  
- P-value: {p_val:.4f}  
- Decision: **{decision}**
""")

            if show_ci:
                ratio = (s1**2) / (s2**2)
                ci_lower = ratio / stats.f.ppf(1 - alpha/2, df1, df2)
                ci_upper = ratio / stats.f.ppf(alpha/2, df1, df2)
                st.markdown("### 🧾 Confidence Interval for Variance Ratio (two-sided)")
                st.latex(r"\left(\frac{s_1^2/s_2^2}{F_{1-\alpha/2,\ df_1, df_2}},\ \frac{s_1^2/s_2^2}{F_{\alpha/2,\ df_1, df_2}}\right)")
                st.markdown(f"**CI ({100*(1-alpha):.0f}%):** ({ci_lower:.4f}, {ci_upper:.4f})")


# ---------- Run ----------
if __name__ == "__main__":
    run_two_sample_tool()


