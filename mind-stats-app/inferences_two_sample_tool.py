# ==========================================================
# two_sample_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# MIND: Statistics Visualizer Suite
# Updated with 5-Step Hypothesis Testing Format
# ==========================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

# ---------- Input Helper ----------
def parse_sample_text(sample_text: str):
    try:
        values = [float(i.strip()) for i in sample_text.split(",") if i.strip() != ""]
        return np.array(values, dtype=float)
    except ValueError:
        return None

def ci_message():
    st.info(
        "Confidence intervals always use a two-tailed critical value because "
        "they estimate a range of plausible values above and below the sample "
        "statistic, regardless of the hypothesis test direction."
    )

# ---------- Tail utilities ----------
def z_tail_metrics(z, alpha, tail):
    if tail == "left":
        crit = stats.norm.ppf(alpha)
        p = stats.norm.cdf(z)
        reject = z < crit
    elif tail == "right":
        crit = stats.norm.ppf(1 - alpha)
        p = 1 - stats.norm.cdf(z)
        reject = z > crit
    else:
        crit = stats.norm.ppf(1 - alpha / 2)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        reject = abs(z) > crit
    return p, reject, crit

def t_tail_metrics(tval, df, alpha, tail):
    if tail == "left":
        crit = stats.t.ppf(alpha, df)
        p = stats.t.cdf(tval, df)
        reject = tval < crit
    elif tail == "right":
        crit = stats.t.ppf(1 - alpha, df)
        p = 1 - stats.t.cdf(tval, df)
        reject = tval > crit
    else:
        crit = stats.t.ppf(1 - alpha / 2, df)
        p = 2 * (1 - stats.t.cdf(abs(tval), df))
        reject = abs(tval) > crit
    return p, reject, crit

def f_tail_metrics(F, df1, df2, alpha, tail):
    if tail == "left":
        crit = stats.f.ppf(alpha, df1, df2)
        p = stats.f.cdf(F, df1, df2)
        reject = F < crit
        return p, reject, crit
    elif tail == "right":
        crit = stats.f.ppf(1 - alpha, df1, df2)
        p = 1 - stats.f.cdf(F, df1, df2)
        reject = F > crit
        return p, reject, crit
    else:
        crit_low = stats.f.ppf(alpha / 2, df1, df2)
        crit_high = stats.f.ppf(1 - alpha / 2, df1, df2)
        p = 2 * min(stats.f.cdf(F, df1, df2), 1 - stats.f.cdf(F, df1, df2))
        p = min(1.0, p)
        reject = (F < crit_low) or (F > crit_high)
        return p, reject, (crit_low, crit_high)

# ---------- Hypothesis Helpers ----------
def get_tail_hypotheses(parameter_symbol, tails):
    if tails == "left":
        h0 = fr"H_0: {parameter_symbol} = 0"
        h1 = fr"H_1: {parameter_symbol} < 0"
    elif tails == "right":
        h0 = fr"H_0: {parameter_symbol} = 0"
        h1 = fr"H_1: {parameter_symbol} > 0"
    else:
        h0 = fr"H_0: {parameter_symbol} = 0"
        h1 = fr"H_1: {parameter_symbol} \ne 0"
    return h0, h1

def get_f_hypotheses(tails):
    if tails == "left":
        h0 = r"H_0: \sigma_1^2 / \sigma_2^2 = 1"
        h1 = r"H_1: \sigma_1^2 / \sigma_2^2 < 1"
    elif tails == "right":
        h0 = r"H_0: \sigma_1^2 / \sigma_2^2 = 1"
        h1 = r"H_1: \sigma_1^2 / \sigma_2^2 > 1"
    else:
        h0 = r"H_0: \sigma_1^2 / \sigma_2^2 = 1"
        h1 = r"H_1: \sigma_1^2 / \sigma_2^2 \ne 1"
    return h0, h1

def conclusion_text(reject, context_reject, context_fail):
    if reject:
        return f"Since the p-value is less than α, we reject H₀. {context_reject}"
    return f"Since the p-value is greater than or equal to α, we do not reject H₀. {context_fail}"

# ---------- Plot Helpers ----------
def plot_normal_rejection(stat, alpha, tails, crit_vals, xlabel="z"):
    fig, ax = plt.subplots(figsize=(8, 4))

    reject_color = "#d62728"
    accept_color = "#2ca02c"
    curve_color = "black"
    stat_color = "#1f77b4"

    x = np.linspace(-4.5, 4.5, 600)
    y = stats.norm.pdf(x)
    ax.set_title("Classical Method: Rejection Region")
    ax.set_xlabel(xlabel)

    if tails == "left":
        crit = crit_vals

        x_reject = np.linspace(-4.5, crit, 300)
        y_reject = stats.norm.pdf(x_reject)
        ax.fill_between(x_reject, y_reject, 0, color=reject_color, alpha=0.75, zorder=2)

        x_accept = np.linspace(crit, 4.5, 300)
        y_accept = stats.norm.pdf(x_accept)
        ax.fill_between(x_accept, y_accept, 0, color=accept_color, alpha=0.55, zorder=1)

        ax.axvline(crit, color="black", linestyle="--", linewidth=2, zorder=4)

    elif tails == "right":
        crit = crit_vals

        x_accept = np.linspace(-4.5, crit, 300)
        y_accept = stats.norm.pdf(x_accept)
        ax.fill_between(x_accept, y_accept, 0, color=accept_color, alpha=0.55, zorder=1)

        x_reject = np.linspace(crit, 4.5, 300)
        y_reject = stats.norm.pdf(x_reject)
        ax.fill_between(x_reject, y_reject, 0, color=reject_color, alpha=0.75, zorder=2)

        ax.axvline(crit, color="black", linestyle="--", linewidth=2, zorder=4)

    else:
        left, right = crit_vals

        x_reject_left = np.linspace(-4.5, left, 300)
        y_reject_left = stats.norm.pdf(x_reject_left)
        ax.fill_between(x_reject_left, y_reject_left, 0, color=reject_color, alpha=0.75, zorder=2)

        x_accept = np.linspace(left, right, 300)
        y_accept = stats.norm.pdf(x_accept)
        ax.fill_between(x_accept, y_accept, 0, color=accept_color, alpha=0.55, zorder=1)

        x_reject_right = np.linspace(right, 4.5, 300)
        y_reject_right = stats.norm.pdf(x_reject_right)
        ax.fill_between(x_reject_right, y_reject_right, 0, color=reject_color, alpha=0.75, zorder=2)

        ax.axvline(left, color="black", linestyle="--", linewidth=2, zorder=4)
        ax.axvline(right, color="black", linestyle="--", linewidth=2, zorder=4)

    ax.plot(x, y, color=curve_color, linewidth=2, zorder=3)
    ax.axvline(stat, color=stat_color, linestyle="-", linewidth=3, zorder=5)
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    st.caption("🟥 Red = Reject H₀ region   |   🟩 Green = Do not reject H₀ region")

def plot_t_rejection(stat, df, alpha, tails, crit_vals):
    fig, ax = plt.subplots(figsize=(8, 4))

    reject_color = "#d62728"
    accept_color = "#2ca02c"
    curve_color = "black"
    stat_color = "#1f77b4"

    x = np.linspace(-5.5, 5.5, 600)
    y = stats.t.pdf(x, df)
    ax.set_title(f"Classical Method: Rejection Region (t, df={df:.2f})")
    ax.set_xlabel("t")

    if tails == "left":
        crit = crit_vals

        x_reject = np.linspace(-5.5, crit, 300)
        y_reject = stats.t.pdf(x_reject, df)
        ax.fill_between(x_reject, y_reject, 0, color=reject_color, alpha=0.75, zorder=2)

        x_accept = np.linspace(crit, 5.5, 300)
        y_accept = stats.t.pdf(x_accept, df)
        ax.fill_between(x_accept, y_accept, 0, color=accept_color, alpha=0.55, zorder=1)

        ax.axvline(crit, color="black", linestyle="--", linewidth=2, zorder=4)

    elif tails == "right":
        crit = crit_vals

        x_accept = np.linspace(-5.5, crit, 300)
        y_accept = stats.t.pdf(x_accept, df)
        ax.fill_between(x_accept, y_accept, 0, color=accept_color, alpha=0.55, zorder=1)

        x_reject = np.linspace(crit, 5.5, 300)
        y_reject = stats.t.pdf(x_reject, df)
        ax.fill_between(x_reject, y_reject, 0, color=reject_color, alpha=0.75, zorder=2)

        ax.axvline(crit, color="black", linestyle="--", linewidth=2, zorder=4)

    else:
        left, right = crit_vals

        x_reject_left = np.linspace(-5.5, left, 300)
        y_reject_left = stats.t.pdf(x_reject_left, df)
        ax.fill_between(x_reject_left, y_reject_left, 0, color=reject_color, alpha=0.75, zorder=2)

        x_accept = np.linspace(left, right, 300)
        y_accept = stats.t.pdf(x_accept, df)
        ax.fill_between(x_accept, y_accept, 0, color=accept_color, alpha=0.55, zorder=1)

        x_reject_right = np.linspace(right, 5.5, 300)
        y_reject_right = stats.t.pdf(x_reject_right, df)
        ax.fill_between(x_reject_right, y_reject_right, 0, color=reject_color, alpha=0.75, zorder=2)

        ax.axvline(left, color="black", linestyle="--", linewidth=2, zorder=4)
        ax.axvline(right, color="black", linestyle="--", linewidth=2, zorder=4)

    ax.plot(x, y, color=curve_color, linewidth=2, zorder=3)
    ax.axvline(stat, color=stat_color, linestyle="-", linewidth=3, zorder=5)
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    st.caption("🟥 Red = Reject H₀ region   |   🟩 Green = Do not reject H₀ region")

def plot_f_rejection(stat, df1, df2, alpha, tails, crit_vals):
    xmax = max(6, stat + 2)
    fig, ax = plt.subplots(figsize=(8, 4))

    reject_color = "#d62728"
    accept_color = "#2ca02c"
    curve_color = "black"
    stat_color = "#1f77b4"

    x = np.linspace(0.001, xmax, 600)
    y = stats.f.pdf(x, df1, df2)
    ax.set_title(f"Classical Method: Rejection Region (F, df1={df1}, df2={df2})")
    ax.set_xlabel("F")

    if tails == "left":
        crit = crit_vals

        x_reject = np.linspace(0.001, crit, 300)
        y_reject = stats.f.pdf(x_reject, df1, df2)
        ax.fill_between(x_reject, y_reject, 0, color=reject_color, alpha=0.75, zorder=2)

        x_accept = np.linspace(crit, xmax, 300)
        y_accept = stats.f.pdf(x_accept, df1, df2)
        ax.fill_between(x_accept, y_accept, 0, color=accept_color, alpha=0.55, zorder=1)

        ax.axvline(crit, color="black", linestyle="--", linewidth=2, zorder=4)

    elif tails == "right":
        crit = crit_vals

        x_accept = np.linspace(0.001, crit, 300)
        y_accept = stats.f.pdf(x_accept, df1, df2)
        ax.fill_between(x_accept, y_accept, 0, color=accept_color, alpha=0.55, zorder=1)

        x_reject = np.linspace(crit, xmax, 300)
        y_reject = stats.f.pdf(x_reject, df1, df2)
        ax.fill_between(x_reject, y_reject, 0, color=reject_color, alpha=0.75, zorder=2)

        ax.axvline(crit, color="black", linestyle="--", linewidth=2, zorder=4)

    else:
        left, right = crit_vals

        x_reject_left = np.linspace(0.001, left, 300)
        y_reject_left = stats.f.pdf(x_reject_left, df1, df2)
        ax.fill_between(x_reject_left, y_reject_left, 0, color=reject_color, alpha=0.75, zorder=2)

        x_accept = np.linspace(left, right, 300)
        y_accept = stats.f.pdf(x_accept, df1, df2)
        ax.fill_between(x_accept, y_accept, 0, color=accept_color, alpha=0.55, zorder=1)

        x_reject_right = np.linspace(right, xmax, 300)
        y_reject_right = stats.f.pdf(x_reject_right, df1, df2)
        ax.fill_between(x_reject_right, y_reject_right, 0, color=reject_color, alpha=0.75, zorder=2)

        ax.axvline(left, color="black", linestyle="--", linewidth=2, zorder=4)
        ax.axvline(right, color="black", linestyle="--", linewidth=2, zorder=4)

    ax.plot(x, y, color=curve_color, linewidth=2, zorder=3)
    ax.axvline(stat, color=stat_color, linestyle="-", linewidth=3, zorder=5)
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    st.caption("🟥 Red = Reject H₀ region   |   🟩 Green = Do not reject H₀ region")

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

    dec = st.number_input("Decimal places for output:", 0, 10, 4)
    alpha = st.number_input("Significance level (α):", 0.001, 0.5, 0.05, step=0.01)
    tails = st.selectbox("Tail type:", ["two", "left", "right"])
    show_ci = st.checkbox("Show Confidence Interval (always two-tailed)")
    show_picture = st.checkbox("Show Classical Method picture", value=True)

    # ==========================================================
    # TWO-PROPORTION Z-TEST
    # ==========================================================
    if test_choice == "Two-Proportion Z-Test":
        x1 = st.number_input("x₁ (successes in Sample 1):", min_value=0, step=1)
        n1 = st.number_input("n₁ (size of Sample 1):", min_value=1, step=1)
        x2 = st.number_input("x₂ (successes in Sample 2):", min_value=0, step=1)
        n2 = st.number_input("n₂ (size of Sample 2):", min_value=1, step=1)

        if st.button("Calculate"):
            if x1 > n1 or x2 > n2:
                st.error("Each number of successes must be less than or equal to its sample size.")
                return

            p1 = x1 / n1
            p2 = x2 / n2
            p_pool = (x1 + x2) / (n1 + n2)
            se_test = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

            if se_test == 0:
                st.error("Standard error is zero. Check your input values.")
                return

            z = (p1 - p2) / se_test
            h0, h1 = get_tail_hypotheses(r"p_1 - p_2", tails)

            st.markdown("### 📘 Step-by-Step Solution")

            themed_box("**Step 1: Hypotheses**")
            st.latex(h0)
            st.latex(h1)

            themed_box("**Step 2: Test Statistic**")
            st.latex(r"z = \frac{(\hat{p}_1 - \hat{p}_2) - 0}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}}")
            st.latex(fr"\hat{{p}}_1 = \frac{{{x1}}}{{{n1}}} = {p1:.{dec}f}")
            st.latex(fr"\hat{{p}}_2 = \frac{{{x2}}}{{{n2}}} = {p2:.{dec}f}")
            st.latex(fr"\hat{{p}} = \frac{{x_1+x_2}}{{n_1+n_2}} = \frac{{{x1}+{x2}}}{{{n1}+{n2}}} = {p_pool:.{dec}f}")
            st.latex(fr"z = {z:.{dec}f}")

            themed_box("**Step 3: Classical Method**")
            if tails == "left":
                crit = stats.norm.ppf(alpha)
                reject_classical = z < crit
                crit_str = f"{crit:.{dec}f}"
                st.latex(fr"\text{{Critical value}} = {crit:.{dec}f}")
                st.markdown(f"Decision rule: Reject H₀ if z < {crit:.{dec}f}.")
                crit_vals = crit
            elif tails == "right":
                crit = stats.norm.ppf(1 - alpha)
                reject_classical = z > crit
                crit_str = f"{crit:.{dec}f}"
                st.latex(fr"\text{{Critical value}} = {crit:.{dec}f}")
                st.markdown(f"Decision rule: Reject H₀ if z > {crit:.{dec}f}.")
                crit_vals = crit
            else:
                left = stats.norm.ppf(alpha / 2)
                right = stats.norm.ppf(1 - alpha / 2)
                reject_classical = (z < left) or (z > right)
                crit_str = f"{left:.{dec}f}, {right:.{dec}f}"
                st.latex(fr"\text{{Critical values}} = {left:.{dec}f},\ {right:.{dec}f}")
                st.markdown(f"Decision rule: Reject H₀ if z < {left:.{dec}f} or z > {right:.{dec}f}.")
                crit_vals = (left, right)

            st.markdown(f"Observed test statistic: z = **{z:.{dec}f}**")
            st.markdown(f"Classical method decision: **{'Reject H₀' if reject_classical else 'Do not reject H₀'}**")

            if show_picture:
                plot_normal_rejection(z, alpha, tails, crit_vals, xlabel="z")

            themed_box("**Step 4: P-value Approach**")
            p_val, reject, _ = z_tail_metrics(z, alpha, tails)
            st.markdown(f"P-value = **{p_val:.{dec}f}**")
            st.markdown(f"α = **{alpha:.{dec}f}**")
            st.markdown(f"P-value approach decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**")

            themed_box("**Step 5: Conclusion**")
            st.write(
                conclusion_text(
                    reject,
                    "There is sufficient evidence to support the claim in H₁ about the difference between the two population proportions.",
                    "There is not sufficient evidence to support the claim in H₁ about the difference between the two population proportions."
                )
            )

            st.markdown("### 📝 Result Summary")
            st.markdown(f"• Sample Proportion 1 (p̂₁): {p1:.{dec}f}")
            st.markdown(f"• Sample Proportion 2 (p̂₂): {p2:.{dec}f}")
            st.markdown(f"• Pooled Proportion (p̂): {p_pool:.{dec}f}")
            st.markdown(f"• Test Statistic (z): {z:.{dec}f}")
            st.markdown(f"• Critical Value(s): {crit_str}")
            st.markdown(f"• P-value: {p_val:.{dec}f}")

            if show_ci:
                ci_message()
                zcrit = stats.norm.ppf(1 - alpha / 2)
                se_ci = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
                ci_low = (p1 - p2) - zcrit * se_ci
                ci_high = (p1 - p2) + zcrit * se_ci
                st.markdown(
                    f"• Confidence Interval ({100*(1-alpha):.0f}%): "
                    f"({ci_low:.{dec}f}, {ci_high:.{dec}f})"
                )

            st.markdown(f"• Decision: {'✅ Reject H₀' if reject else '❌ Do not reject H₀'}")

    # ==========================================================
    # PAIRED t-TEST (DATA)
    # ==========================================================
    elif test_choice == "Paired t-Test (Data)":
        s1 = st.text_area("Sample 1:", "1,2,3")
        s2 = st.text_area("Sample 2:", "1,2,3")

        if st.button("Calculate"):
            x1 = parse_sample_text(s1)
            x2 = parse_sample_text(s2)

            if x1 is None or x2 is None:
                st.error("Please enter valid numeric data separated by commas.")
                return
            if len(x1) != len(x2):
                st.error("For a paired t-test, both samples must have the same length.")
                return
            if len(x1) < 2:
                st.error("At least two paired observations are required.")
                return

            d = x1 - x2
            mean_d = np.mean(d)
            sd_d = np.std(d, ddof=1)
            se = sd_d / np.sqrt(len(d))

            if se == 0:
                st.error("Standard error is zero. The paired differences may all be identical.")
                return

            tstat = mean_d / se
            df = len(d) - 1
            h0, h1 = get_tail_hypotheses(r"\mu_d", tails)

            st.markdown("### 📘 Step-by-Step Solution")

            themed_box("**Step 1: Hypotheses**")
            st.latex(h0)
            st.latex(h1)

            themed_box("**Step 2: Test Statistic**")
            st.latex(r"t = \frac{\bar{d} - 0}{s_d / \sqrt{n}}")
            st.latex(fr"\bar{{d}} = {mean_d:.{dec}f}, \quad s_d = {sd_d:.{dec}f}, \quad n = {len(d)}")
            st.latex(fr"t = {tstat:.{dec}f}")

            themed_box("**Step 3: Classical Method**")
            if tails == "left":
                crit = stats.t.ppf(alpha, df)
                reject_classical = tstat < crit
                crit_str = f"{crit:.{dec}f}"
                st.latex(fr"\text{{Critical value}} = {crit:.{dec}f}")
                st.markdown(f"Decision rule: Reject H₀ if t < {crit:.{dec}f}.")
                crit_vals = crit
            elif tails == "right":
                crit = stats.t.ppf(1 - alpha, df)
                reject_classical = tstat > crit
                crit_str = f"{crit:.{dec}f}"
                st.latex(fr"\text{{Critical value}} = {crit:.{dec}f}")
                st.markdown(f"Decision rule: Reject H₀ if t > {crit:.{dec}f}.")
                crit_vals = crit
            else:
                left = stats.t.ppf(alpha / 2, df)
                right = stats.t.ppf(1 - alpha / 2, df)
                reject_classical = (tstat < left) or (tstat > right)
                crit_str = f"{left:.{dec}f}, {right:.{dec}f}"
                st.latex(fr"\text{{Critical values}} = {left:.{dec}f},\ {right:.{dec}f}")
                st.markdown(f"Decision rule: Reject H₀ if t < {left:.{dec}f} or t > {right:.{dec}f}.")
                crit_vals = (left, right)

            st.markdown(f"Observed test statistic: t = **{tstat:.{dec}f}**")
            st.markdown(f"Classical method decision: **{'Reject H₀' if reject_classical else 'Do not reject H₀'}**")

            if show_picture:
                plot_t_rejection(tstat, df, alpha, tails, crit_vals)

            themed_box("**Step 4: P-value Approach**")
            p_val, reject, _ = t_tail_metrics(tstat, df, alpha, tails)
            st.markdown(f"P-value = **{p_val:.{dec}f}**")
            st.markdown(f"α = **{alpha:.{dec}f}**")
            st.markdown(f"P-value approach decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**")

            themed_box("**Step 5: Conclusion**")
            st.write(
                conclusion_text(
                    reject,
                    "There is sufficient evidence to support the claim in H₁ about the mean paired difference.",
                    "There is not sufficient evidence to support the claim in H₁ about the mean paired difference."
                )
            )

            st.markdown("### 📝 Result Summary")
            st.markdown(f"• Mean Difference: {mean_d:.{dec}f}")
            st.markdown(f"• Test Statistic (t): {tstat:.{dec}f}")
            st.markdown(f"• Critical Value(s): {crit_str}")
            st.markdown(f"• P-value: {p_val:.{dec}f}")

            if show_ci:
                ci_message()
                tcrit = stats.t.ppf(1 - alpha / 2, df)
                ci_low = mean_d - tcrit * se
                ci_high = mean_d + tcrit * se
                st.markdown(
                    f"• Confidence Interval ({100*(1-alpha):.0f}%): "
                    f"({ci_low:.{dec}f}, {ci_high:.{dec}f})"
                )

            st.markdown(f"• Decision: {'✅ Reject H₀' if reject else '❌ Do not reject H₀'}")

    # ==========================================================
    # PAIRED t-TEST (SUMMARY)
    # ==========================================================
    elif test_choice == "Paired t-Test (Summary)":
        n = st.number_input("n (number of pairs):", min_value=2, step=1)
        mean_d = st.number_input("Mean of differences (d̄):", value=0.0)
        sd_d = st.number_input("Standard deviation of differences (s_d):", min_value=0.0, value=1.0)

        if st.button("Calculate"):
            if sd_d == 0:
                st.error("The standard deviation of differences must be greater than 0.")
                return

            se = sd_d / np.sqrt(n)
            tstat = mean_d / se
            df = n - 1
            h0, h1 = get_tail_hypotheses(r"\mu_d", tails)

            st.markdown("### 📘 Step-by-Step Solution")

            themed_box("**Step 1: Hypotheses**")
            st.latex(h0)
            st.latex(h1)

            themed_box("**Step 2: Test Statistic**")
            st.latex(r"t = \frac{\bar{d} - 0}{s_d / \sqrt{n}}")
            st.latex(fr"\bar{{d}} = {mean_d:.{dec}f}, \quad s_d = {sd_d:.{dec}f}, \quad n = {n}")
            st.latex(fr"t = {tstat:.{dec}f}")

            themed_box("**Step 3: Classical Method**")
            if tails == "left":
                crit = stats.t.ppf(alpha, df)
                reject_classical = tstat < crit
                crit_str = f"{crit:.{dec}f}"
                st.latex(fr"\text{{Critical value}} = {crit:.{dec}f}")
                st.markdown(f"Decision rule: Reject H₀ if t < {crit:.{dec}f}.")
                crit_vals = crit
            elif tails == "right":
                crit = stats.t.ppf(1 - alpha, df)
                reject_classical = tstat > crit
                crit_str = f"{crit:.{dec}f}"
                st.latex(fr"\text{{Critical value}} = {crit:.{dec}f}")
                st.markdown(f"Decision rule: Reject H₀ if t > {crit:.{dec}f}.")
                crit_vals = crit
            else:
                left = stats.t.ppf(alpha / 2, df)
                right = stats.t.ppf(1 - alpha / 2, df)
                reject_classical = (tstat < left) or (tstat > right)
                crit_str = f"{left:.{dec}f}, {right:.{dec}f}"
                st.latex(fr"\text{{Critical values}} = {left:.{dec}f},\ {right:.{dec}f}")
                st.markdown(f"Decision rule: Reject H₀ if t < {left:.{dec}f} or t > {right:.{dec}f}.")
                crit_vals = (left, right)

            st.markdown(f"Observed test statistic: t = **{tstat:.{dec}f}**")
            st.markdown(f"Classical method decision: **{'Reject H₀' if reject_classical else 'Do not reject H₀'}**")

            if show_picture:
                plot_t_rejection(tstat, df, alpha, tails, crit_vals)

            themed_box("**Step 4: P-value Approach**")
            p_val, reject, _ = t_tail_metrics(tstat, df, alpha, tails)
            st.markdown(f"P-value = **{p_val:.{dec}f}**")
            st.markdown(f"α = **{alpha:.{dec}f}**")
            st.markdown(f"P-value approach decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**")

            themed_box("**Step 5: Conclusion**")
            st.write(
                conclusion_text(
                    reject,
                    "There is sufficient evidence to support the claim in H₁ about the mean paired difference.",
                    "There is not sufficient evidence to support the claim in H₁ about the mean paired difference."
                )
            )

            st.markdown("### 📝 Result Summary")
            st.markdown(f"• Mean Difference: {mean_d:.{dec}f}")
            st.markdown(f"• Test Statistic (t): {tstat:.{dec}f}")
            st.markdown(f"• Critical Value(s): {crit_str}")
            st.markdown(f"• P-value: {p_val:.{dec}f}")

            if show_ci:
                ci_message()
                tcrit = stats.t.ppf(1 - alpha / 2, df)
                ci_low = mean_d - tcrit * se
                ci_high = mean_d + tcrit * se
                st.markdown(
                    f"• Confidence Interval ({100*(1-alpha):.0f}%): "
                    f"({ci_low:.{dec}f}, {ci_high:.{dec}f})"
                )

            st.markdown(f"• Decision: {'✅ Reject H₀' if reject else '❌ Do not reject H₀'}")

    # ==========================================================
    # INDEPENDENT t-TEST (DATA, WELCH)
    # ==========================================================
    elif test_choice == "Independent t-Test (Data, Welch)":
        a = st.text_area("Sample 1:", "1,2,3")
        b = st.text_area("Sample 2:", "4,5,6")

        if st.button("Calculate"):
            x1 = parse_sample_text(a)
            x2 = parse_sample_text(b)

            if x1 is None or x2 is None:
                st.error("Please enter valid numeric data separated by commas.")
                return
            if len(x1) < 2 or len(x2) < 2:
                st.error("Each sample must contain at least two observations.")
                return

            m1, m2 = np.mean(x1), np.mean(x2)
            s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
            n1, n2 = len(x1), len(x2)

            se = np.sqrt(s1**2 / n1 + s2**2 / n2)
            if se == 0:
                st.error("Standard error is zero. Check your sample data.")
                return

            tstat = (m1 - m2) / se
            df = (s1**2 / n1 + s2**2 / n2)**2 / (
                ((s1**2 / n1)**2) / (n1 - 1) + ((s2**2 / n2)**2) / (n2 - 1)
            )
            h0, h1 = get_tail_hypotheses(r"\mu_1 - \mu_2", tails)

            st.markdown("### 📘 Step-by-Step Solution")

            themed_box("**Step 1: Hypotheses**")
            st.latex(h0)
            st.latex(h1)

            themed_box("**Step 2: Test Statistic**")
            st.latex(r"t = \frac{(\bar{x}_1 - \bar{x}_2) - 0}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}")
            st.latex(fr"\bar{{x}}_1 = {m1:.{dec}f}, \quad \bar{{x}}_2 = {m2:.{dec}f}")
            st.latex(fr"s_1 = {s1:.{dec}f}, \quad s_2 = {s2:.{dec}f}")
            st.latex(fr"n_1 = {n1}, \quad n_2 = {n2}")
            st.latex(fr"t = {tstat:.{dec}f}")

            themed_box("**Step 3: Classical Method**")
            if tails == "left":
                crit = stats.t.ppf(alpha, df)
                reject_classical = tstat < crit
                crit_str = f"{crit:.{dec}f}"
                st.latex(fr"\text{{Critical value}} = {crit:.{dec}f}")
                st.markdown(f"Decision rule: Reject H₀ if t < {crit:.{dec}f}.")
                crit_vals = crit
            elif tails == "right":
                crit = stats.t.ppf(1 - alpha, df)
                reject_classical = tstat > crit
                crit_str = f"{crit:.{dec}f}"
                st.latex(fr"\text{{Critical value}} = {crit:.{dec}f}")
                st.markdown(f"Decision rule: Reject H₀ if t > {crit:.{dec}f}.")
                crit_vals = crit
            else:
                left = stats.t.ppf(alpha / 2, df)
                right = stats.t.ppf(1 - alpha / 2, df)
                reject_classical = (tstat < left) or (tstat > right)
                crit_str = f"{left:.{dec}f}, {right:.{dec}f}"
                st.latex(fr"\text{{Critical values}} = {left:.{dec}f},\ {right:.{dec}f}")
                st.markdown(f"Decision rule: Reject H₀ if t < {left:.{dec}f} or t > {right:.{dec}f}.")
                crit_vals = (left, right)

            st.markdown(f"Observed test statistic: t = **{tstat:.{dec}f}**")
            st.markdown(f"Classical method decision: **{'Reject H₀' if reject_classical else 'Do not reject H₀'}**")

            if show_picture:
                plot_t_rejection(tstat, df, alpha, tails, crit_vals)

            themed_box("**Step 4: P-value Approach**")
            p_val, reject, _ = t_tail_metrics(tstat, df, alpha, tails)
            st.markdown(f"P-value = **{p_val:.{dec}f}**")
            st.markdown(f"α = **{alpha:.{dec}f}**")
            st.markdown(f"P-value approach decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**")

            themed_box("**Step 5: Conclusion**")
            st.write(
                conclusion_text(
                    reject,
                    "There is sufficient evidence to support the claim in H₁ about the difference between the two population means.",
                    "There is not sufficient evidence to support the claim in H₁ about the difference between the two population means."
                )
            )

            st.markdown("### 📝 Result Summary")
            st.markdown(f"• Sample Mean 1: {m1:.{dec}f}")
            st.markdown(f"• Sample Mean 2: {m2:.{dec}f}")
            st.markdown(f"• Test Statistic (t): {tstat:.{dec}f}")
            st.markdown(f"• Degrees of Freedom: {df:.{dec}f}")
            st.markdown(f"• Critical Value(s): {crit_str}")
            st.markdown(f"• P-value: {p_val:.{dec}f}")

            if show_ci:
                ci_message()
                tcrit = stats.t.ppf(1 - alpha / 2, df)
                ci_low = (m1 - m2) - tcrit * se
                ci_high = (m1 - m2) + tcrit * se
                st.markdown(
                    f"• Confidence Interval ({100*(1-alpha):.0f}%): "
                    f"({ci_low:.{dec}f}, {ci_high:.{dec}f})"
                )

            st.markdown(f"• Decision: {'✅ Reject H₀' if reject else '❌ Do not reject H₀'}")

    # ==========================================================
    # INDEPENDENT t-TEST (SUMMARY, WELCH)
    # ==========================================================
    elif test_choice == "Independent t-Test (Summary, Welch)":
        n1 = st.number_input("n₁:", min_value=2, step=1)
        mean1 = st.number_input("x̄₁:", value=0.0)
        s1 = st.number_input("s₁:", min_value=0.0, value=1.0)

        n2 = st.number_input("n₂:", min_value=2, step=1)
        mean2 = st.number_input("x̄₂:", value=0.0)
        s2 = st.number_input("s₂:", min_value=0.0, value=1.0)

        if st.button("Calculate"):
            if s1 == 0 and s2 == 0:
                st.error("At least one sample standard deviation must be greater than 0.")
                return

            se = np.sqrt(s1**2 / n1 + s2**2 / n2)
            if se == 0:
                st.error("Standard error is zero. Check your summary values.")
                return

            tstat = (mean1 - mean2) / se
            df = (s1**2 / n1 + s2**2 / n2)**2 / (
                ((s1**2 / n1)**2) / (n1 - 1) + ((s2**2 / n2)**2) / (n2 - 1)
            )
            h0, h1 = get_tail_hypotheses(r"\mu_1 - \mu_2", tails)

            st.markdown("### 📘 Step-by-Step Solution")

            themed_box("**Step 1: Hypotheses**")
            st.latex(h0)
            st.latex(h1)

            themed_box("**Step 2: Test Statistic**")
            st.latex(r"t = \frac{(\bar{x}_1 - \bar{x}_2) - 0}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}")
            st.latex(fr"\bar{{x}}_1 = {mean1:.{dec}f}, \quad \bar{{x}}_2 = {mean2:.{dec}f}")
            st.latex(fr"s_1 = {s1:.{dec}f}, \quad s_2 = {s2:.{dec}f}")
            st.latex(fr"n_1 = {n1}, \quad n_2 = {n2}")
            st.latex(fr"t = {tstat:.{dec}f}")

            themed_box("**Step 3: Classical Method**")
            if tails == "left":
                crit = stats.t.ppf(alpha, df)
                reject_classical = tstat < crit
                crit_str = f"{crit:.{dec}f}"
                st.latex(fr"\text{{Critical value}} = {crit:.{dec}f}")
                st.markdown(f"Decision rule: Reject H₀ if t < {crit:.{dec}f}.")
                crit_vals = crit
            elif tails == "right":
                crit = stats.t.ppf(1 - alpha, df)
                reject_classical = tstat > crit
                crit_str = f"{crit:.{dec}f}"
                st.latex(fr"\text{{Critical value}} = {crit:.{dec}f}")
                st.markdown(f"Decision rule: Reject H₀ if t > {crit:.{dec}f}.")
                crit_vals = crit
            else:
                left = stats.t.ppf(alpha / 2, df)
                right = stats.t.ppf(1 - alpha / 2, df)
                reject_classical = (tstat < left) or (tstat > right)
                crit_str = f"{left:.{dec}f}, {right:.{dec}f}"
                st.latex(fr"\text{{Critical values}} = {left:.{dec}f},\ {right:.{dec}f}")
                st.markdown(f"Decision rule: Reject H₀ if t < {left:.{dec}f} or t > {right:.{dec}f}.")
                crit_vals = (left, right)

            st.markdown(f"Observed test statistic: t = **{tstat:.{dec}f}**")
            st.markdown(f"Classical method decision: **{'Reject H₀' if reject_classical else 'Do not reject H₀'}**")

            if show_picture:
                plot_t_rejection(tstat, df, alpha, tails, crit_vals)

            themed_box("**Step 4: P-value Approach**")
            p_val, reject, _ = t_tail_metrics(tstat, df, alpha, tails)
            st.markdown(f"P-value = **{p_val:.{dec}f}**")
            st.markdown(f"α = **{alpha:.{dec}f}**")
            st.markdown(f"P-value approach decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**")

            themed_box("**Step 5: Conclusion**")
            st.write(
                conclusion_text(
                    reject,
                    "There is sufficient evidence to support the claim in H₁ about the difference between the two population means.",
                    "There is not sufficient evidence to support the claim in H₁ about the difference between the two population means."
                )
            )

            st.markdown("### 📝 Result Summary")
            st.markdown(f"• Difference in Means: {(mean1 - mean2):.{dec}f}")
            st.markdown(f"• Test Statistic (t): {tstat:.{dec}f}")
            st.markdown(f"• Degrees of Freedom: {df:.{dec}f}")
            st.markdown(f"• Critical Value(s): {crit_str}")
            st.markdown(f"• P-value: {p_val:.{dec}f}")

            if show_ci:
                ci_message()
                tcrit = stats.t.ppf(1 - alpha / 2, df)
                ci_low = (mean1 - mean2) - tcrit * se
                ci_high = (mean1 - mean2) + tcrit * se
                st.markdown(
                    f"• Confidence Interval ({100*(1-alpha):.0f}%): "
                    f"({ci_low:.{dec}f}, {ci_high:.{dec}f})"
                )

            st.markdown(f"• Decision: {'✅ Reject H₀' if reject else '❌ Do not reject H₀'}")

    # ==========================================================
    # F-TESTS (DATA & SUMMARY)
    # ==========================================================
    elif test_choice in ["F-Test (Data)", "F-Test (Summary)"]:
        if test_choice == "F-Test (Data)":
            a = st.text_area("Sample 1:", "1,2,3")
            b = st.text_area("Sample 2:", "4,5,6")

            if st.button("Calculate"):
                x1 = parse_sample_text(a)
                x2 = parse_sample_text(b)

                if x1 is None or x2 is None:
                    st.error("Please enter valid numeric data separated by commas.")
                    return
                if len(x1) < 2 or len(x2) < 2:
                    st.error("Each sample must contain at least two observations.")
                    return

                s1 = np.std(x1, ddof=1)
                s2 = np.std(x2, ddof=1)
                df1, df2 = len(x1) - 1, len(x2) - 1

                if s2 == 0:
                    st.error("The second sample standard deviation is zero, so F cannot be computed.")
                    return

                F = (s1**2) / (s2**2)
                h0, h1 = get_f_hypotheses(tails)

                st.markdown("### 📘 Step-by-Step Solution")

                themed_box("**Step 1: Hypotheses**")
                st.latex(h0)
                st.latex(h1)

                themed_box("**Step 2: Test Statistic**")
                st.latex(r"F = \frac{s_1^2}{s_2^2}")
                st.latex(fr"s_1 = {s1:.{dec}f}, \quad s_2 = {s2:.{dec}f}")
                st.latex(fr"F = {F:.{dec}f}")

                themed_box("**Step 3: Classical Method**")
                p_val, reject, crit = f_tail_metrics(F, df1, df2, alpha, tails)

                if tails == "two":
                    crit_str = f"{crit[0]:.{dec}f}, {crit[1]:.{dec}f}"
                    st.latex(fr"\text{{Critical values}} = {crit[0]:.{dec}f},\ {crit[1]:.{dec}f}")
                    st.markdown(
                        f"Decision rule: Reject H₀ if F < {crit[0]:.{dec}f} or F > {crit[1]:.{dec}f}."
                    )
                    reject_classical = (F < crit[0]) or (F > crit[1])
                    crit_vals = crit
                else:
                    crit_str = f"{crit:.{dec}f}"
                    st.latex(fr"\text{{Critical value}} = {crit:.{dec}f}")
                    if tails == "left":
                        st.markdown(f"Decision rule: Reject H₀ if F < {crit:.{dec}f}.")
                        reject_classical = F < crit
                    else:
                        st.markdown(f"Decision rule: Reject H₀ if F > {crit:.{dec}f}.")
                        reject_classical = F > crit
                    crit_vals = crit

                st.markdown(f"Observed test statistic: F = **{F:.{dec}f}**")
                st.markdown(f"Classical method decision: **{'Reject H₀' if reject_classical else 'Do not reject H₀'}**")

                if show_picture:
                    plot_f_rejection(F, df1, df2, alpha, tails, crit_vals)

                themed_box("**Step 4: P-value Approach**")
                st.markdown(f"P-value = **{p_val:.{dec}f}**")
                st.markdown(f"α = **{alpha:.{dec}f}**")
                st.markdown(f"P-value approach decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**")

                themed_box("**Step 5: Conclusion**")
                st.write(
                    conclusion_text(
                        reject,
                        "There is sufficient evidence to support the claim in H₁ about the population variances.",
                        "There is not sufficient evidence to support the claim in H₁ about the population variances."
                    )
                )

                st.markdown("### 📝 Result Summary")
                st.markdown(f"• Test Statistic (F): {F:.{dec}f}")
                st.markdown(f"• Critical Value(s): {crit_str}")
                st.markdown(f"• P-value: {p_val:.{dec}f}")
                st.markdown(f"• Decision: {'✅ Reject H₀' if reject else '❌ Do not reject H₀'}")

        else:
            n1 = st.number_input("n₁:", min_value=2, step=1)
            s1 = st.number_input("s₁:", min_value=0.0, value=1.0)
            n2 = st.number_input("n₂:", min_value=2, step=1)
            s2 = st.number_input("s₂:", min_value=0.0, value=1.0)

            if st.button("Calculate"):
                if s2 == 0:
                    st.error("The second sample standard deviation must be greater than 0.")
                    return

                df1, df2 = n1 - 1, n2 - 1
                F = (s1**2) / (s2**2)
                h0, h1 = get_f_hypotheses(tails)

                st.markdown("### 📘 Step-by-Step Solution")

                themed_box("**Step 1: Hypotheses**")
                st.latex(h0)
                st.latex(h1)

                themed_box("**Step 2: Test Statistic**")
                st.latex(r"F = \frac{s_1^2}{s_2^2}")
                st.latex(fr"s_1 = {s1:.{dec}f}, \quad s_2 = {s2:.{dec}f}")
                st.latex(fr"F = {F:.{dec}f}")

                themed_box("**Step 3: Classical Method**")
                p_val, reject, crit = f_tail_metrics(F, df1, df2, alpha, tails)

                if tails == "two":
                    crit_str = f"{crit[0]:.{dec}f}, {crit[1]:.{dec}f}"
                    st.latex(fr"\text{{Critical values}} = {crit[0]:.{dec}f},\ {crit[1]:.{dec}f}")
                    st.markdown(
                        f"Decision rule: Reject H₀ if F < {crit[0]:.{dec}f} or F > {crit[1]:.{dec}f}."
                    )
                    reject_classical = (F < crit[0]) or (F > crit[1])
                    crit_vals = crit
                else:
                    crit_str = f"{crit:.{dec}f}"
                    st.latex(fr"\text{{Critical value}} = {crit:.{dec}f}")
                    if tails == "left":
                        st.markdown(f"Decision rule: Reject H₀ if F < {crit:.{dec}f}.")
                        reject_classical = F < crit
                    else:
                        st.markdown(f"Decision rule: Reject H₀ if F > {crit:.{dec}f}.")
                        reject_classical = F > crit
                    crit_vals = crit

                st.markdown(f"Observed test statistic: F = **{F:.{dec}f}**")
                st.markdown(f"Classical method decision: **{'Reject H₀' if reject_classical else 'Do not reject H₀'}**")

                if show_picture:
                    plot_f_rejection(F, df1, df2, alpha, tails, crit_vals)

                themed_box("**Step 4: P-value Approach**")
                st.markdown(f"P-value = **{p_val:.{dec}f}**")
                st.markdown(f"α = **{alpha:.{dec}f}**")
                st.markdown(f"P-value approach decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**")

                themed_box("**Step 5: Conclusion**")
                st.write(
                    conclusion_text(
                        reject,
                        "There is sufficient evidence to support the claim in H₁ about the population variances.",
                        "There is not sufficient evidence to support the claim in H₁ about the population variances."
                    )
                )

                st.markdown("### 📝 Result Summary")
                st.markdown(f"• Test Statistic (F): {F:.{dec}f}")
                st.markdown(f"• Critical Value(s): {crit_str}")
                st.markdown(f"• P-value: {p_val:.{dec}f}")
                st.markdown(f"• Decision: {'✅ Reject H₀' if reject else '❌ Do not reject H₀'}")

# ---------- RUN ----------
if __name__ == "__main__":
    run_two_sample_tool()
         
