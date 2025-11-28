# ==========================================================
# two_sample_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# Rebuilt "Perfect Final Version"
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

# ==========================================================
# UNIVERSAL INTERPRETATION BOX (Dark/Light Adaptive)
# ==========================================================
def interp_box(text: str):
    st.markdown(
        f"""
        <div style="
            padding:12px;
            border-radius:10px;
            border-left:5px solid #4A90E2;
            margin-top:10px;
            margin-bottom:12px;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# STEP BOX
# ==========================================================
def step_box(text: str):
    st.markdown(
        f"""
        <div style="
            background-color:rgba(74, 144, 226, 0.10);
            padding:10px;
            border-radius:10px;
            border-left:5px solid #4A90E2;
            margin-bottom:10px;
        ">
            <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# TAIL HANDLERS
# ==========================================================
def t_tail(tval, df, alpha, tail):
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
    else:  # two-tail
        crit = stats.t.ppf(1 - alpha/2, df)
        p = 2 * (1 - stats.t.cdf(abs(tval), df))
        reject = abs(tval) > crit
        crit_str = f"±{crit:.4f}"

    return p, reject, crit_str


def z_tail(z, alpha, tail):
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


# ==========================================================
# MAIN APP
# ==========================================================
def run_two_sample_tool():

    st.header("👨‍🔬 MIND: Two-Sample Inference Tool")
    st.markdown("Built with the students in **MIND — Los Angeles City College**")
    st.markdown("---")

    # Decimal place selector
    dec = st.number_input("Decimal places for output:", 0, 10, 4)

    test_choice = st.selectbox(
        "Choose a Two-Sample Test:",
        [
            "Two-Proportion Z-Test",
            "Paired t-Test (Raw Data)",
            "Paired t-Test (Summary Statistics)",
            "Independent t-Test (Raw Data, Welch)",
            "Independent t-Test (Summary Statistics, Welch)",
            "F-Test for Standard Deviations (Raw Data)",
            "F-Test for Standard Deviations (Summary Statistics)"
        ],
        index=None,
        placeholder="Select a test to begin..."
    )

    if not test_choice:
        st.info("👆 Select a two-sample test to begin.")
        return

    alpha = st.number_input("Significance level (α):", 0.001, 0.5, 0.05, 0.01)
    tails = st.selectbox("Tail type:", ["two", "left", "right"])


    # ==========================================================
    # TWO-PROPORTION Z-TEST
    # ==========================================================
    if test_choice == "Two-Proportion Z-Test":

        st.subheader("📊 Enter Counts")

        x1 = st.number_input("Successes in Sample 1 (x₁)", 0, step=1)
        n1 = st.number_input("Sample Size 1 (n₁)", 1, step=1)

        x2 = st.number_input("Successes in Sample 2 (x₂)", 0, step=1)
        n2 = st.number_input("Sample Size 2 (n₂)", 1, step=1)

        if st.button("Calculate"):

            p1 = x1 / n1
            p2 = x2 / n2
            pooled = (x1 + x2) / (n1 + n2)

            se = np.sqrt(pooled * (1 - pooled) * (1/n1 + 1/n2))
            z = (p1 - p2) / se

            st.markdown("### 📘 Step-by-Step Solution")

            step_box("**Step 1:** Compute sample proportions and pooled estimate.")
            st.latex(r"\hat p_1=\frac{x_1}{n_1},\;\hat p_2=\frac{x_2}{n_2},\;\hat p=\frac{x_1+x_2}{n_1+n_2}")
            st.write(f"p̂₁ = {p1:.{dec}f}, p̂₂ = {p2:.{dec}f}, pooled = {pooled:.{dec}f}")

            step_box("**Step 2:** Compute SE and z-value.")
            st.latex(r"z=\frac{\hat p_1-\hat p_2}{\sqrt{\hat p(1-\hat p)\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}}")
            st.write(f"SE = {se:.{dec}f}, z = {z:.{dec}f}")

            step_box("**Step 3:** Tail-specific p-value.")
            p_val, reject, crit = z_tail(z, alpha, tails)

            interp_box(f"""
            <b>Result Summary</b><br><br>
            z = {z:.{dec}f} <br>
            Critical value(s): {crit} <br>
            p-value = {p_val:.{dec}f} <br><br>
            <b>Decision:</b> {"Reject H₀" if reject else "Fail to reject H₀"}
            """)


    # ==========================================================
    # PAIRED t-TEST (RAW DATA)
    # ==========================================================
    elif test_choice == "Paired t-Test (Raw Data)":

        st.subheader("📊 Enter Paired Samples")

        col1, col2 = st.columns(2)
        with col1:
            s1 = st.text_area("Sample 1 (comma-separated):", "5, 6, 8, 9")
        with col2:
            s2 = st.text_area("Sample 2 (comma-separated):", "3, 7, 6, 10")

        if st.button("Calculate"):

            try:
                x1 = np.array([float(i) for i in s1.split(",")])
                x2 = np.array([float(i) for i in s2.split(",")])
            except:
                st.error("Invalid data.")
                return

            if len(x1) != len(x2):
                st.error("Samples must be the same length.")
                return

            d = x1 - x2
            n = len(d)
            mean_d = np.mean(d)
            sd_d = np.std(d, ddof=1)
            se = sd_d / np.sqrt(n)
            tval = mean_d / se
            df = n - 1

            st.markdown("### 📘 Step-by-Step Solution")

            # SHOW DIFFERENCE TABLE
            step_box("**Step 1: Compute differences** (d = x₁ − x₂)")
            df_table = pd.DataFrame({"x₁": x1, "x₂": x2, "d = x₁−x₂": d})
            st.dataframe(df_table, use_container_width=True)

            # Summary
            step_box("**Step 2: Summary statistics**")
            st.write(f"Mean(d) = {mean_d:.{dec}f}, SD(d) = {sd_d:.{dec}f}, n = {n}")

            # Formula
            step_box("**Step 3: Compute t-statistic**")
            st.latex(r"t=\frac{\bar d}{s_d/\sqrt{n}}")
            st.write(f"t = {tval:.{dec}f}, df = {df}")

            # P-value
            step_box("**Step 4: Compute p-value**")
            p_val, reject, crit = t_tail(tval, df, alpha, tails)

            interp_box(f"""
            <b>Result Summary</b><br><br>
            t = {tval:.{dec}f} (df = {df}) <br>
            Critical value(s): {crit} <br>
            p-value = {p_val:.{dec}f} <br><br>
            <b>Decision:</b> {"Reject H₀" if reject else "Fail to reject H₀"}
            """)


    # ==========================================================
    # PAIRED t-TEST (SUMMARY)
    # ==========================================================
    elif test_choice == "Paired t-Test (Summary Statistics)":

        mean_d = st.number_input("Mean of differences (ḋ)", format="%.6f")
        sd_d = st.number_input("SD of differences (s_d)", format="%.6f")
        n = st.number_input("Sample size n:", 2, step=1)

        if st.button("Calculate"):

            df = n - 1
            se = sd_d / np.sqrt(n)
            tval = mean_d / se

            st.markdown("### 📘 Step-by-Step Solution")

            step_box("**Step 1: Test statistic**")
            st.latex(r"t=\frac{\bar d}{s_d/\sqrt{n}}")
            st.write(f"t = {tval:.{dec}f}, df = {df}")

            step_box("**Step 2: P-value**")
            p_val, reject, crit = t_tail(tval, df, alpha, tails)

            interp_box(f"""
            <b>Result Summary</b><br><br>
            t = {tval:.{dec}f} (df = {df})<br>
            Critical value(s): {crit} <br>
            p-value = {p_val:.{dec}f} <br><br>
            <b>Decision:</b> {"Reject H₀" if reject else "Fail to reject H₀"}
            """)


    # ==========================================================
    # INDEPENDENT t-TEST (RAW DATA, WELCH)
    # ==========================================================
    elif test_choice == "Independent t-Test (Raw Data, Welch)":

        st.subheader("📊 Enter Independent Samples")

        s1 = st.text_area("Sample 1:", "5,6,7,8")
        s2 = st.text_area("Sample 2:", "3,2,4,5")

        if st.button("Calculate"):

            try:
                x1 = np.array([float(i) for i in s1.split(",")])
                x2 = np.array([float(i) for i in s2.split(",")])
            except:
                st.error("Invalid input.")
                return

            n1, n2 = len(x1), len(x2)
            mean1, mean2 = np.mean(x1), np.mean(x2)
            s1_, s2_ = np.std(x1, ddof=1), np.std(x2, ddof=1)

            se = np.sqrt(s1_**2/n1 + s2_**2/n2)
            tval = (mean1 - mean2) / se

            df = (se**4) / ((s1_**4)/(n1**2*(n1-1)) + (s2_**4)/(n2**2*(n2-1)))

            st.markdown("### 📘 Step-by-Step Solution")

            step_box("**Step 1: Summary Statistics**")
            st.write(f"x̄₁ = {mean1:.{dec}f}, x̄₂ = {mean2:.{dec}f}")
            st.write(f"s₁ = {s1_:.{dec}f}, s₂ = {s2_:.{dec}f}")

            step_box("**Step 2: Compute SE and t**")
            st.latex(r"t=\frac{\bar x_1-\bar x_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}")
            st.write(f"t = {tval:.{dec}f}, df ≈ {df:.2f}")

            p_val, reject, crit = t_tail(tval, df, alpha, tails)

            interp_box(f"""
            <b>Result Summary</b><br><br>
            t = {tval:.{dec}f} (df ≈ {df:.2f})<br>
            Critical value(s): {crit} <br>
            p-value = {p_val:.{dec}f} <br><br>
            <b>Decision:</b> {"Reject H₀" if reject else "Fail to reject H₀"}
            """)


    # ==========================================================
    # INDEPENDENT t-TEST (SUMMARY STATISTICS, WELCH)
    # ==========================================================
    elif test_choice == "Independent t-Test (Summary Statistics, Welch)":

        mean1 = st.number_input("Mean 1", format="%.6f")
        s1_ = st.number_input("Std Dev 1", format="%.6f")
        n1 = st.number_input("n₁", 2, step=1)

        mean2 = st.number_input("Mean 2", format="%.6f")
        s2_ = st.number_input("Std Dev 2", format="%.6f")
        n2 = st.number_input("n₂", 2, step=1)

        if st.button("Calculate"):

            se = np.sqrt(s1_**2/n1 + s2_**2/n2)
            tval = (mean1 - mean2) / se

            df = (se**4) / ((s1_**4)/(n1**2*(n1-1)) + (s2_**4)/(n2**2*(n2-1)))

            step_box("**Step 1: Compute SE and t**")
            st.write(f"t = {tval:.{dec}f}, df ≈ {df:.2f}")

            p_val, reject, crit = t_tail(tval, df, alpha, tails)

            interp_box(f"""
            <b>Result Summary</b><br><br>
            t = {tval:.{dec}f} (df ≈ {df:.2f})<br>
            Critical value(s): {crit} <br>
            p-value = {p_val:.{dec}f} <br><br>
            <b>Decision:</b> {"Reject H₀" if reject else "Fail to reject H₀"}
            """)


    # ==========================================================
    # F-TEST RAW DATA
    # ==========================================================
    elif test_choice == "F-Test for Standard Deviations (Raw Data)":

        st.subheader("📊 Enter Independent Samples")

        s1 = st.text_area("Sample 1:", "5,6,7,8")
        s2 = st.text_area("Sample 2:", "3,4,6,5")

        if st.button("Calculate"):

            x1 = np.array([float(i) for i in s1.split(",")])
            x2 = np.array([float(i) for i in s2.split(",")])

            n1, n2 = len(x1), len(x2)
            s1_, s2_ = np.std(x1, ddof=1), np.std(x2, ddof=1)

            F = (s1_**2) / (s2_**2)
            df1, df2 = n1 - 1, n2 - 1

            if tails == "left":
                crit = stats.f.ppf(alpha, df1, df2)
                p_val = stats.f.cdf(F, df1, df2)
                reject = F < crit
                crit_str = f"{crit:.4f}"
            elif tails == "right":
                crit = stats.f.ppf(1 - alpha, df1, df2)
                p_val = 1 - stats.f.cdf(F, df1, df2)
                reject = F > crit
                crit_str = f"{crit:.4f}"
            else:
                crit_low = stats.f.ppf(alpha/2, df1, df2)
                crit_high = stats.f.ppf(1 - alpha/2, df1, df2)
                p_val = 2 * min(stats.f.cdf(F, df1, df2), 1 - stats.f.cdf(F, df1, df2))
                reject = F < crit_low or F > crit_high
                crit_str = f"({crit_low:.4f}, {crit_high:.4f})"

            interp_box(f"""
            <b>Result Summary</b><br><br>
            F = {F:.{dec}f} (df₁={df1}, df₂={df2})<br>
            Critical region: {crit_str}<br>
            p-value = {p_val:.{dec}f}<br><br>
            <b>Decision:</b> {"Reject H₀" if reject else "Fail to reject H₀"}
            """)


    # ==========================================================
    # F-TEST SUMMARY STATS
    # ==========================================================
    elif test_choice == "F-Test for Standard Deviations (Summary Statistics)":

        n1 = st.number_input("n₁", 2, step=1)
        s1_ = st.number_input("s₁", value=1.0, format="%.6f")

        n2 = st.number_input("n₂", 2, step=1)
        s2_ = st.number_input("s₂", value=1.0, format="%.6f")

        if st.button("Calculate"):

            F = (s1_**2) / (s2_**2)
            df1, df2 = n1 - 1, n2 - 1

            if tails == "left":
                crit = stats.f.ppf(alpha, df1, df2)
                p_val = stats.f.cdf(F, df1, df2)
                reject = F < crit
                crit_str = f"{crit:.4f}"
            elif tails == "right":
                crit = stats.f.ppf(1 - alpha, df1, df2)
                p_val = 1 - stats.f.cdf(F, df1, df2)
                reject = F > crit
                crit_str = f"{crit:.4f}"
            else:
                crit_low = stats.f.ppf(alpha/2, df1, df2)
                crit_high = stats.f.ppf(1 - alpha/2, df1, df2)
                p_val = 2 * min(stats.f.cdf(F, df1, df2), 1 - stats.f.cdf(F, df1, df2))
                reject = F < crit_low or F > crit_high
                crit_str = f"({crit_low:.4f}, {crit_high:.4f})"

            interp_box(f"""
            <b>Result Summary</b><br><br>
            F = {F:.{dec}f} (df₁={df1}, df₂={df2})<br>
            Critical region: {crit_str}<br>
            p-value = {p_val:.{dec}f}<br><br>
            <b>Decision:</b> {"Reject H₀" if reject else "Fail to reject H₀"}
            """)


# ==========================================================
# RUN APP
# ==========================================================
if __name__ == "__main__":
    run_two_sample_tool()

