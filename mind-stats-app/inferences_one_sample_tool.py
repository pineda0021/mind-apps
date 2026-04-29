# ==========================================================
# inferences_one_sample_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# MIND: Statistics Visualizer Suite
# Updated with 5-Step Hypothesis Testing Format
# ==========================================================

import streamlit as st
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t, chi2, binom

# ==========================================================
# Auto Light/Dark Mode Box
# ==========================================================
def themed_box(text):
    st.markdown(f"""
        <style>
            .themed-box {{
                padding: 12px;
                border-radius: 10px;
                margin-bottom: 10px;
                border-left: 5px solid #007acc;
            }}
            @media (prefers-color-scheme: light) {{
                .themed-box {{
                    background-color: #e6f3ff;
                    color: black;
                }}
            }}
            @media (prefers-color-scheme: dark) {{
                .themed-box {{
                    background-color: #2b2b2b;
                    color: white;
                }}
            }}
        </style>
        <div class="themed-box">{text}</div>
    """, unsafe_allow_html=True)


# ==========================================================
# Helper: Upload Numeric Data
# ==========================================================
def load_uploaded_data():
    uploaded_file = st.file_uploader(
        "📂 Upload CSV or Excel file with a single column of numeric data",
        type=["csv", "xlsx"]
    )
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    return df[col].dropna().to_numpy()
            st.error("No numeric column found in file.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    return None


# ==========================================================
# Formatting Helpers
# ==========================================================
def get_tail_hypotheses(parameter_symbol, null_value, tails):
    if tails == "left":
        h0 = fr"H_0: {parameter_symbol} = {null_value}"
        h1 = fr"H_1: {parameter_symbol} < {null_value}"
    elif tails == "right":
        h0 = fr"H_0: {parameter_symbol} = {null_value}"
        h1 = fr"H_1: {parameter_symbol} > {null_value}"
    else:
        h0 = fr"H_0: {parameter_symbol} = {null_value}"
        h1 = fr"H_1: {parameter_symbol} \ne {null_value}"
    return h0, h1


def conclusion_text(reject, context_reject, context_fail):
    if reject:
        return f"Since the p-value is less than α, we reject H₀. {context_reject}"
    return f"Since the p-value is greater than or equal to α, we do not reject H₀. {context_fail}"


# ==========================================================
# Plot Helpers
# ==========================================================
def plot_continuous_rejection_region(dist_name, df, stat, alpha, tails, crit_vals):
    fig, ax = plt.subplots(figsize=(8, 4))

    if dist_name == "z":
        x = np.linspace(-4, 4, 500)
        y = norm.pdf(x)
        ax.plot(x, y)
        ax.set_title("Classical Method: Rejection Region (Z Distribution)")

        # Green base = do not reject region
        ax.fill_between(x, y, color="green", alpha=0.15)

        # Red = rejection region
        if tails == "left":
            crit = crit_vals
            xx = np.linspace(-4, crit, 200)
            ax.fill_between(xx, norm.pdf(xx), color="red", alpha=0.35)
            ax.axvline(crit, linestyle="--")
        elif tails == "right":
            crit = crit_vals
            xx = np.linspace(crit, 4, 200)
            ax.fill_between(xx, norm.pdf(xx), color="red", alpha=0.35)
            ax.axvline(crit, linestyle="--")
        else:
            left, right = crit_vals
            xx1 = np.linspace(-4, left, 200)
            xx2 = np.linspace(right, 4, 200)
            ax.fill_between(xx1, norm.pdf(xx1), color="red", alpha=0.35)
            ax.fill_between(xx2, norm.pdf(xx2), color="red", alpha=0.35)
            ax.axvline(left, linestyle="--")
            ax.axvline(right, linestyle="--")

        ax.axvline(stat, linestyle="-", linewidth=2)
        ax.set_xlabel("z")

    elif dist_name == "t":
        x = np.linspace(-5, 5, 500)
        y = t.pdf(x, df)
        ax.plot(x, y)
        ax.set_title(f"Classical Method: Rejection Region (t Distribution, df={df})")

        # Green base = do not reject region
        ax.fill_between(x, y, color="green", alpha=0.15)

        # Red = rejection region
        if tails == "left":
            crit = crit_vals
            xx = np.linspace(-5, crit, 200)
            ax.fill_between(xx, t.pdf(xx, df), color="red", alpha=0.35)
            ax.axvline(crit, linestyle="--")
        elif tails == "right":
            crit = crit_vals
            xx = np.linspace(crit, 5, 200)
            ax.fill_between(xx, t.pdf(xx, df), color="red", alpha=0.35)
            ax.axvline(crit, linestyle="--")
        else:
            left, right = crit_vals
            xx1 = np.linspace(-5, left, 200)
            xx2 = np.linspace(right, 5, 200)
            ax.fill_between(xx1, t.pdf(xx1, df), color="red", alpha=0.35)
            ax.fill_between(xx2, t.pdf(xx2, df), color="red", alpha=0.35)
            ax.axvline(left, linestyle="--")
            ax.axvline(right, linestyle="--")

        ax.axvline(stat, linestyle="-", linewidth=2)
        ax.set_xlabel("t")

    elif dist_name == "chi2":
        x = np.linspace(0, max(20, stat + 5), 500)
        y = chi2.pdf(x, df)
        ax.plot(x, y)
        ax.set_title(f"Classical Method: Rejection Region (Chi-Square Distribution, df={df})")

        # Green base = do not reject region
        ax.fill_between(x, y, color="green", alpha=0.15)

        # Red = rejection region
        if tails == "left":
            crit = crit_vals
            xx = np.linspace(0, crit, 200)
            ax.fill_between(xx, chi2.pdf(xx, df), color="red", alpha=0.35)
            ax.axvline(crit, linestyle="--")
        elif tails == "right":
            crit = crit_vals
            xx = np.linspace(crit, max(20, stat + 5), 200)
            ax.fill_between(xx, chi2.pdf(xx, df), color="red", alpha=0.35)
            ax.axvline(crit, linestyle="--")
        else:
            left, right = crit_vals
            xx1 = np.linspace(0, left, 200)
            xx2 = np.linspace(right, max(20, stat + 5), 200)
            ax.fill_between(xx1, chi2.pdf(xx1, df), color="red", alpha=0.35)
            ax.fill_between(xx2, chi2.pdf(xx2, df), color="red", alpha=0.35)
            ax.axvline(left, linestyle="--")
            ax.axvline(right, linestyle="--")

        ax.axvline(stat, linestyle="-", linewidth=2)
        ax.set_xlabel(r"$\chi^2$")

    st.pyplot(fig)
    st.caption("🟥 Red = Reject H₀ region   |   🟩 Green = Do not reject H₀ region")


def plot_binomial_tail(n, p0, x_obs, tails):
    xs = np.arange(0, n + 1)
    probs = binom.pmf(xs, n, p0)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Default all bars GREEN
    colors = ["green"] * len(xs)

    # Rejection region RED
    if tails == "left":
        for i in range(len(xs)):
            if xs[i] <= x_obs:
                colors[i] = "red"
    elif tails == "right":
        for i in range(len(xs)):
            if xs[i] >= x_obs:
                colors[i] = "red"
    else:
        center_prob = binom.pmf(x_obs, n, p0)
        for i in range(len(xs)):
            if probs[i] <= center_prob + 1e-15:
                colors[i] = "red"

    ax.bar(xs, probs, color=colors)
    ax.set_title("Classical Method / Exact Binomial Picture")
    ax.set_xlabel("x")
    ax.set_ylabel("P(X = x)")

    st.pyplot(fig)
    st.caption("🟥 Red = Reject H₀ region   |   🟩 Green = Do not reject H₀ region")


# ==========================================================
# Main App
# ==========================================================
def run_hypothesis_tool():
    st.header("🔎 Inferences on One Sample")

    decimals = st.number_input(
        "Decimal places for output:",
        min_value=0,
        max_value=10,
        value=4,
        step=1
    )

    fmt = f"{{:.{decimals}f}}"

    test_options = [
        "Proportion test (large sample)",
        "Proportion test (small sample, binomial)",
        "t-test for population mean (summary stats)",
        "t-test for population mean (raw data)",
        "Chi-squared test for std dev (summary stats)",
        "Chi-squared test for std dev (raw data)"
    ]

    test_choice = st.selectbox(
        "Choose a hypothesis test:",
        test_options,
        index=None,
        placeholder="Select a hypothesis test to begin..."
    )

    if not test_choice:
        st.info("👆 Please select a hypothesis test to begin.")
        return

    alpha = st.number_input(
        "Significance level (α)",
        value=0.05,
        min_value=0.001,
        max_value=0.5,
        step=0.01
    )

    tails = st.selectbox("Tail type:", ["two", "left", "right"])
    show_picture = st.checkbox("Show Classical Method picture", value=True)

    # ==========================================================
    # PROPORTION TESTS
    # ==========================================================
    if test_choice in ["Proportion test (large sample)", "Proportion test (small sample, binomial)"]:
        x = st.number_input("Number of successes (x)", min_value=0, step=1)
        n = st.number_input("Sample size (n)", min_value=1, step=1)
        p0 = st.number_input("Null proportion (p₀)", min_value=0.0, max_value=1.0, format="%.6f")

        if st.button("👨‍💻 Calculate"):
            if x > n:
                st.error("x cannot be greater than n.")
                return

            p_hat = x / n
            h0, h1 = get_tail_hypotheses("p", p0, tails)

            st.markdown("### 📘 Step-by-Step Solution")

            # ------------------------------------------------------
            # LARGE SAMPLE: Z TEST
            # ------------------------------------------------------
            if test_choice == "Proportion test (large sample)":
                se = math.sqrt(p0 * (1 - p0) / n)
                if se == 0:
                    st.error("Standard error is zero. Check p₀ and n.")
                    return

                z_stat = (p_hat - p0) / se

                # Step 1
                themed_box("**Step 1: Hypotheses**")
                st.latex(h0)
                st.latex(h1)

                # Step 2
                themed_box("**Step 2: Test Statistic**")
                st.latex(r"z = \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}}")
                st.latex(fr"\hat{{p}} = \frac{{x}}{{n}} = \frac{{{x}}}{{{n}}} = {fmt.format(p_hat)}")
                st.latex(fr"\text{{SE}} = \sqrt{{\frac{{p_0(1-p_0)}}{{n}}}} = {fmt.format(se)}")
                st.latex(fr"z = {fmt.format(z_stat)}")

                # Step 3
                themed_box("**Step 3: Classical Method**")
                if tails == "left":
                    z_crit = norm.ppf(alpha)
                    reject_classical = z_stat < z_crit
                    crit_str = fmt.format(z_crit)
                    st.latex(fr"\text{{Critical value}} = {fmt.format(z_crit)}")
                    st.markdown(
                        f"Decision rule: Reject H₀ if z < {fmt.format(z_crit)}."
                    )
                    crit_vals = z_crit
                elif tails == "right":
                    z_crit = norm.ppf(1 - alpha)
                    reject_classical = z_stat > z_crit
                    crit_str = fmt.format(z_crit)
                    st.latex(fr"\text{{Critical value}} = {fmt.format(z_crit)}")
                    st.markdown(
                        f"Decision rule: Reject H₀ if z > {fmt.format(z_crit)}."
                    )
                    crit_vals = z_crit
                else:
                    z_left = norm.ppf(alpha / 2)
                    z_right = norm.ppf(1 - alpha / 2)
                    reject_classical = (z_stat < z_left) or (z_stat > z_right)
                    crit_str = f"{fmt.format(z_left)}, {fmt.format(z_right)}"
                    st.latex(fr"\text{{Critical values}} = {fmt.format(z_left)},\ {fmt.format(z_right)}")
                    st.markdown(
                        f"Decision rule: Reject H₀ if z < {fmt.format(z_left)} or z > {fmt.format(z_right)}."
                    )
                    crit_vals = (z_left, z_right)

                st.markdown(
                    f"Observed test statistic: z = **{fmt.format(z_stat)}**"
                )
                st.markdown(
                    f"Classical method decision: **{'Reject H₀' if reject_classical else 'Do not reject H₀'}**"
                )

                if show_picture:
                    plot_continuous_rejection_region("z", None, z_stat, alpha, tails, crit_vals)

                # Step 4
                themed_box("**Step 4: P-value Approach**")
                if tails == "left":
                    p_val = norm.cdf(z_stat)
                elif tails == "right":
                    p_val = 1 - norm.cdf(z_stat)
                else:
                    p_val = 2 * (1 - norm.cdf(abs(z_stat)))

                reject = p_val < alpha
                st.markdown(f"P-value = **{fmt.format(p_val)}**")
                st.markdown(f"α = **{fmt.format(alpha)}**")
                st.markdown(
                    f"P-value approach decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**"
                )

                # Step 5
                themed_box("**Step 5: Conclusion**")
                final_text = conclusion_text(
                    reject,
                    f"There is sufficient evidence to support the claim in H₁ that the population proportion differs from {fmt.format(p0)}."
                    if tails == "two" else
                    f"There is sufficient evidence to support the claim in H₁ about the population proportion.",
                    f"There is not sufficient evidence to support the claim in H₁ about the population proportion."
                )
                st.write(final_text)

                st.markdown(f"""
### **Result Summary**
- Sample proportion: {fmt.format(p_hat)}
- Test Statistic (z): {fmt.format(z_stat)}
- Critical Value(s): {crit_str}
- P-value: {fmt.format(p_val)}
- Decision: **{'✅ Reject H₀' if reject else '❌ Do not reject H₀'}**
""")

            # ------------------------------------------------------
            # SMALL SAMPLE: BINOMIAL EXACT TEST
            # ------------------------------------------------------
            else:
                themed_box("**Step 1: Hypotheses**")
                st.latex(h0)
                st.latex(h1)

                themed_box("**Step 2: Test Statistic**")
                st.markdown(
                    "For the exact binomial test, we use the observed number of successes as the test statistic."
                )
                st.latex(fr"x = {x}, \quad n = {n}, \quad \hat{{p}} = \frac{{{x}}}{{{n}}} = {fmt.format(p_hat)}")

                themed_box("**Step 3: Classical Method**")
                st.markdown(
                    "For an exact binomial test, the rejection region is determined from the binomial distribution under the null hypothesis."
                )
                st.markdown(f"Null model: X ~ Binomial(n = {n}, p = {p0})")

                if show_picture:
                    plot_binomial_tail(n, p0, x, tails)

                themed_box("**Step 4: P-value Approach**")
                if tails == "left":
                    p_val = binom.cdf(x, n, p0)
                elif tails == "right":
                    p_val = 1 - binom.cdf(x - 1, n, p0)
                else:
                    left = binom.cdf(x, n, p0)
                    right = 1 - binom.cdf(x - 1, n, p0)
                    p_val = float(min(1, 2 * min(left, right)))

                reject = p_val < alpha
                st.markdown(f"P-value = **{fmt.format(p_val)}**")
                st.markdown(f"α = **{fmt.format(alpha)}**")
                st.markdown(
                    f"P-value approach decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**"
                )

                themed_box("**Step 5: Conclusion**")
                final_text = conclusion_text(
                    reject,
                    "There is sufficient evidence to support the claim in H₁ about the population proportion.",
                    "There is not sufficient evidence to support the claim in H₁ about the population proportion."
                )
                st.write(final_text)

                st.markdown(f"""
### **Result Summary**
- Sample proportion: {fmt.format(p_hat)}
- P-value: {fmt.format(p_val)}
- Decision: **{'✅ Reject H₀' if reject else '❌ Do not reject H₀'}**
""")

    # ==========================================================
    # T-TESTS
    # ==========================================================
    elif test_choice in [
        "t-test for population mean (summary stats)",
        "t-test for population mean (raw data)"
    ]:

        if test_choice == "t-test for population mean (summary stats)":
            mean = st.number_input("Sample mean (x̄)", format="%.6f")
            sd = st.number_input("Sample standard deviation (s)", min_value=0.0, format="%.6f")
            n = st.number_input("Sample size (n)", min_value=2, step=1)

        else:
            st.markdown("### 📊 Provide Sample Data")
            uploaded_data = load_uploaded_data()
            raw_input = st.text_area("Or enter comma-separated values:")

        mu0 = st.number_input("Null hypothesis mean (μ₀)", format="%.6f")

        if st.button("👨‍💻 Calculate"):
            if test_choice == "t-test for population mean (raw data)":
                if uploaded_data is not None:
                    data = uploaded_data
                elif raw_input:
                    try:
                        data = np.array([float(i.strip()) for i in raw_input.split(",") if i.strip() != ""])
                    except ValueError:
                        st.error("Please enter valid numeric data.")
                        return
                else:
                    st.warning("⚠ Please provide sample data.")
                    return

                if len(data) < 2:
                    st.error("At least two observations are required.")
                    return

                mean = np.mean(data)
                sd = np.std(data, ddof=1)
                n = len(data)

            if sd == 0:
                st.error("The sample standard deviation must be greater than 0.")
                return

            df = n - 1
            se = sd / math.sqrt(n)
            t_stat = (mean - mu0) / se
            h0, h1 = get_tail_hypotheses(r"\mu", mu0, tails)

            st.markdown("### 📘 Step-by-Step Solution")

            # Step 1
            themed_box("**Step 1: Hypotheses**")
            st.latex(h0)
            st.latex(h1)

            # Step 2
            themed_box("**Step 2: Test Statistic**")
            st.latex(r"t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}")
            st.latex(fr"\bar{{x}} = {fmt.format(mean)}, \quad s = {fmt.format(sd)}, \quad n = {n}")
            st.latex(fr"\text{{SE}} = \frac{{s}}{{\sqrt{{n}}}} = {fmt.format(se)}")
            st.latex(fr"t = {fmt.format(t_stat)}")

            # Step 3
            themed_box("**Step 3: Classical Method**")
            if tails == "left":
                t_crit = t.ppf(alpha, df)
                reject_classical = t_stat < t_crit
                crit_str = fmt.format(t_crit)
                st.latex(fr"\text{{Critical value}} = {fmt.format(t_crit)}")
                st.markdown(f"Decision rule: Reject H₀ if t < {fmt.format(t_crit)}.")
                crit_vals = t_crit
            elif tails == "right":
                t_crit = t.ppf(1 - alpha, df)
                reject_classical = t_stat > t_crit
                crit_str = fmt.format(t_crit)
                st.latex(fr"\text{{Critical value}} = {fmt.format(t_crit)}")
                st.markdown(f"Decision rule: Reject H₀ if t > {fmt.format(t_crit)}.")
                crit_vals = t_crit
            else:
                t_left = t.ppf(alpha / 2, df)
                t_right = t.ppf(1 - alpha / 2, df)
                reject_classical = (t_stat < t_left) or (t_stat > t_right)
                crit_str = f"{fmt.format(t_left)}, {fmt.format(t_right)}"
                st.latex(fr"\text{{Critical values}} = {fmt.format(t_left)},\ {fmt.format(t_right)}")
                st.markdown(
                    f"Decision rule: Reject H₀ if t < {fmt.format(t_left)} or t > {fmt.format(t_right)}."
                )
                crit_vals = (t_left, t_right)

            st.markdown(f"Observed test statistic: t = **{fmt.format(t_stat)}**")
            st.markdown(
                f"Classical method decision: **{'Reject H₀' if reject_classical else 'Do not reject H₀'}**"
            )

            if show_picture:
                plot_continuous_rejection_region("t", df, t_stat, alpha, tails, crit_vals)

            # Step 4
            themed_box("**Step 4: P-value Approach**")
            if tails == "left":
                p_val = t.cdf(t_stat, df)
            elif tails == "right":
                p_val = 1 - t.cdf(t_stat, df)
            else:
                p_val = 2 * (1 - t.cdf(abs(t_stat), df))

            reject = p_val < alpha
            st.markdown(f"P-value = **{fmt.format(p_val)}**")
            st.markdown(f"α = **{fmt.format(alpha)}**")
            st.markdown(
                f"P-value approach decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**"
            )

            # Step 5
            themed_box("**Step 5: Conclusion**")
            final_text = conclusion_text(
                reject,
                "There is sufficient evidence to support the claim in H₁ about the population mean.",
                "There is not sufficient evidence to support the claim in H₁ about the population mean."
            )
            st.write(final_text)

            st.markdown(f"""
### **Result Summary**
- df = {df}
- t-statistic = {fmt.format(t_stat)}
- Critical Value(s): {crit_str}
- P-value = {fmt.format(p_val)}
- Decision: **{'✅ Reject H₀' if reject else '❌ Do not reject H₀'}**
""")

    # ==========================================================
    # CHI-SQUARED TESTS
    # ==========================================================
    else:

        if test_choice == "Chi-squared test for std dev (summary stats)":
            sd = st.number_input("Sample standard deviation (s)", min_value=0.0, format="%.6f")
            n = st.number_input("Sample size (n)", min_value=2, step=1)

        else:
            st.markdown("### 📊 Provide Sample Data")
            uploaded_data = load_uploaded_data()
            raw_input = st.text_area("Or enter comma-separated values:")

        sigma0 = st.number_input("Population standard deviation (σ₀)", min_value=0.000001, format="%.6f")

        if st.button("👨‍💻 Calculate"):
            if test_choice == "Chi-squared test for std dev (raw data)":
                if uploaded_data is not None:
                    data = uploaded_data
                elif raw_input:
                    try:
                        data = np.array([float(i.strip()) for i in raw_input.split(",") if i.strip() != ""])
                    except ValueError:
                        st.error("Please enter valid numeric data.")
                        return
                else:
                    st.warning("⚠ Please provide sample data.")
                    return

                if len(data) < 2:
                    st.error("At least two observations are required.")
                    return

                sd = np.std(data, ddof=1)
                n = len(data)

            if sd == 0:
                st.error("The sample standard deviation must be greater than 0.")
                return

            df = n - 1
            chi2_stat = (df * sd**2) / sigma0**2
            h0, h1 = get_tail_hypotheses(r"\sigma", sigma0, tails)

            st.markdown("### 📘 Step-by-Step Solution")

            # Step 1
            themed_box("**Step 1: Hypotheses**")
            st.latex(h0)
            st.latex(h1)

            # Step 2
            themed_box("**Step 2: Test Statistic**")
            st.latex(r"\chi^2 = \frac{(n-1)s^2}{\sigma_0^2}")
            st.latex(fr"s = {fmt.format(sd)}, \quad n = {n}, \quad df = {df}")
            st.latex(fr"\chi^2 = {fmt.format(chi2_stat)}")

            # Step 3
            themed_box("**Step 3: Classical Method**")
            if tails == "left":
                chi_crit = chi2.ppf(alpha, df)
                reject_classical = chi2_stat < chi_crit
                crit_str = fmt.format(chi_crit)
                st.latex(fr"\text{{Critical value}} = {fmt.format(chi_crit)}")
                st.markdown(f"Decision rule: Reject H₀ if χ² < {fmt.format(chi_crit)}.")
                crit_vals = chi_crit
            elif tails == "right":
                chi_crit = chi2.ppf(1 - alpha, df)
                reject_classical = chi2_stat > chi_crit
                crit_str = fmt.format(chi_crit)
                st.latex(fr"\text{{Critical value}} = {fmt.format(chi_crit)}")
                st.markdown(f"Decision rule: Reject H₀ if χ² > {fmt.format(chi_crit)}.")
                crit_vals = chi_crit
            else:
                left = chi2.ppf(alpha / 2, df)
                right = chi2.ppf(1 - alpha / 2, df)
                reject_classical = (chi2_stat < left) or (chi2_stat > right)
                crit_str = f"{fmt.format(left)}, {fmt.format(right)}"
                st.latex(fr"\text{{Critical values}} = {fmt.format(left)},\ {fmt.format(right)}")
                st.markdown(
                    f"Decision rule: Reject H₀ if χ² < {fmt.format(left)} or χ² > {fmt.format(right)}."
                )
                crit_vals = (left, right)

            st.markdown(f"Observed test statistic: χ² = **{fmt.format(chi2_stat)}**")
            st.markdown(
                f"Classical method decision: **{'Reject H₀' if reject_classical else 'Do not reject H₀'}**"
            )

            if show_picture:
                plot_continuous_rejection_region("chi2", df, chi2_stat, alpha, tails, crit_vals)

            # Step 4
            themed_box("**Step 4: P-value Approach**")
            if tails == "left":
                p_val = chi2.cdf(chi2_stat, df)
            elif tails == "right":
                p_val = 1 - chi2.cdf(chi2_stat, df)
            else:
                p_val = 2 * min(chi2.cdf(chi2_stat, df), 1 - chi2.cdf(chi2_stat, df))
                p_val = min(1.0, p_val)

            reject = p_val < alpha
            st.markdown(f"P-value = **{fmt.format(p_val)}**")
            st.markdown(f"α = **{fmt.format(alpha)}**")
            st.markdown(
                f"P-value approach decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**"
            )

            # Step 5
            themed_box("**Step 5: Conclusion**")
            final_text = conclusion_text(
                reject,
                "There is sufficient evidence to support the claim in H₁ about the population standard deviation.",
                "There is not sufficient evidence to support the claim in H₁ about the population standard deviation."
            )
            st.write(final_text)

            st.markdown(f"""
### **Result Summary**
- df = {df}
- χ² statistic = {fmt.format(chi2_stat)}
- Critical Value(s): {crit_str}
- P-value = {fmt.format(p_val)}
- Decision: **{'✅ Reject H₀' if reject else '❌ Do not reject H₀'}**
""")


# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":
    run_hypothesis_tool()
