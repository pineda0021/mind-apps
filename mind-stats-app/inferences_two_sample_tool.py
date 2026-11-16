
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ==========================================================
# inferences_two_sample_tool.py
# MIND: Statistics Visualizer Suite
# Two-Sample Hypothesis Tests
# ==========================================================

# ==========================================================
# AUTO LIGHT/DARK MODE BOX
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
# INPUT HELPERS
# ==========================================================
def parse_sample_text(sample_text):
    try:
        values = [
            float(i.strip())
            for i in sample_text.replace("\n", ",").split(",")
            if i.strip()
        ]
        if not values or not np.all(np.isfinite(values)):
            return None
        return np.array(values, dtype=float)
    except ValueError:
        return None


def ci_message():
    st.info(
        "Confidence intervals use two-tailed critical values "
        "to estimate a range of plausible parameter values, "
        "regardless of the hypothesis test direction."
    )


# ==========================================================
# HYPOTHESIS HELPERS
# ==========================================================
def get_tail_hypotheses(parameter_symbol, tails):

    if parameter_symbol == r"\mu_1 - \mu_2":
        a, b = r"\mu_1", r"\mu_2"
    elif parameter_symbol == r"p_1 - p_2":
        a, b = r"p_1", r"p_2"
    else:
        a, b = None, None

    if a is not None:
        h0 = fr"H_0: {a} = {b}"

        if tails == "left":
            h1 = fr"H_1: {a} < {b}"
        elif tails == "right":
            h1 = fr"H_1: {a} > {b}"
        else:
            h1 = fr"H_1: {a} \ne {b}"

    else:
        h0 = fr"H_0: {parameter_symbol} = 0"

        if tails == "left":
            h1 = fr"H_1: {parameter_symbol} < 0"
        elif tails == "right":
            h1 = fr"H_1: {parameter_symbol} > 0"
        else:
            h1 = fr"H_1: {parameter_symbol} \ne 0"

    return h0, h1


def get_f_hypotheses(tails):
    h0 = r"H_0: \sigma_1 = \sigma_2"

    if tails == "left":
        h1 = r"H_1: \sigma_1 < \sigma_2"
    elif tails == "right":
        h1 = r"H_1: \sigma_1 > \sigma_2"
    else:
        h1 = r"H_1: \sigma_1 \ne \sigma_2"

    return h0, h1


def conclusion_text(reject, context_reject, context_fail):
    if reject:
        return (
            "Since the p-value is less than α, we reject H₀. "
            + context_reject
        )
    return (
        "Since the p-value is greater than or equal to α, "
        "we do not reject H₀. " + context_fail
    )


# ==========================================================
# TAIL UTILITIES
# ==========================================================
def z_tail_metrics(z, alpha, tail):
    if tail == "left":
        crit = stats.norm.ppf(alpha)
        p = stats.norm.cdf(z)
        reject = z < crit
    elif tail == "right":
        crit = stats.norm.ppf(1 - alpha)
        p = stats.norm.sf(z)
        reject = z > crit
    else:
        low = stats.norm.ppf(alpha / 2)
        high = stats.norm.ppf(1 - alpha / 2)
        crit = (low, high)
        p = 2 * stats.norm.sf(abs(z))
        reject = z < low or z > high

    return p, reject, crit


def t_tail_metrics(tval, df, alpha, tail):
    if tail == "left":
        crit = stats.t.ppf(alpha, df)
        p = stats.t.cdf(tval, df)
        reject = tval < crit
    elif tail == "right":
        crit = stats.t.ppf(1 - alpha, df)
        p = stats.t.sf(tval, df)
        reject = tval > crit
    else:
        low = stats.t.ppf(alpha / 2, df)
        high = stats.t.ppf(1 - alpha / 2, df)
        crit = (low, high)
        p = 2 * stats.t.sf(abs(tval), df)
        reject = tval < low or tval > high

    return p, reject, crit


def f_tail_metrics(F, df1, df2, alpha, tail):
    if tail == "left":
        crit = stats.f.ppf(alpha, df1, df2)
        p = stats.f.cdf(F, df1, df2)
        reject = F < crit

    elif tail == "right":
        crit = stats.f.ppf(1 - alpha, df1, df2)
        p = stats.f.sf(F, df1, df2)
        reject = F > crit

    else:
        low = stats.f.ppf(alpha / 2, df1, df2)
        high = stats.f.ppf(1 - alpha / 2, df1, df2)
        crit = (low, high)
        p = min(
            1.0,
            2 * min(
                stats.f.cdf(F, df1, df2),
                stats.f.sf(F, df1, df2)
            )
        )
        reject = F < low or F > high

    return p, reject, crit


# ==========================================================
# CLASSICAL METHOD GRAPHS
# ==========================================================
def plot_rejection(stat, dist, alpha, tails, crit_vals,
                   xlabel="t", title="Classical Method"):

    if xlabel == "F":
        q = dist.ppf(0.995)
        xmax = max(6, stat + 2, q if np.isfinite(q) else 6)
        xmin = 0.0001
    else:
        if isinstance(crit_vals, tuple):
            xmin = min(-5.5, stat - 1, crit_vals[0] - 1)
            xmax = max(5.5, stat + 1, crit_vals[1] + 1)
        else:
            xmin = min(-5.5, stat - 1, crit_vals - 1)
            xmax = max(5.5, stat + 1, crit_vals + 1)

    x = np.linspace(xmin, xmax, 800)
    y = dist.pdf(x)

    fig, ax = plt.subplots(figsize=(8, 4))

    if tails == "left":
        mask = x < crit_vals
        ax.axvline(crit_vals, color="black", linestyle="--", linewidth=2)

    elif tails == "right":
        mask = x > crit_vals
        ax.axvline(crit_vals, color="black", linestyle="--", linewidth=2)

    else:
        low, high = crit_vals
        mask = (x < low) | (x > high)
        ax.axvline(low, color="black", linestyle="--", linewidth=2)
        ax.axvline(high, color="black", linestyle="--", linewidth=2)

    ax.fill_between(
        x, y, 0, where=mask,
        color="#d62728", alpha=0.75
    )
    ax.fill_between(
        x, y, 0, where=~mask,
        color="#2ca02c", alpha=0.55
    )

    ax.plot(x, y, color="black", linewidth=2)
    ax.axvline(
        stat, color="#1f77b4",
        linewidth=3, label=f"{xlabel} = {stat:.4f}"
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "🟥 Red = Reject H₀ region   |   "
        "🟩 Green = Do not reject H₀ region   |   "
        "Blue = Test statistic"
    )


def plot_normal_rejection(stat, alpha, tails, crit_vals, xlabel="z"):
    plot_rejection(
        stat, stats.norm(), alpha, tails, crit_vals,
        xlabel, "Classical Method: Rejection Region"
    )


def plot_t_rejection(stat, df, alpha, tails, crit_vals):
    plot_rejection(
        stat, stats.t(df), alpha, tails, crit_vals,
        "t", f"Classical Method: Rejection Region (t, df={df:.2f})"
    )


def plot_f_rejection(stat, df1, df2, alpha, tails, crit_vals):
    plot_rejection(
        stat, stats.f(df1, df2), alpha, tails, crit_vals,
        "F",
        f"Classical Method: Rejection Region (F, df1={df1}, df2={df2})"
    )


# ==========================================================
# CLASSICAL METHOD DISPLAY
# ==========================================================
def show_classical(statistic, symbol, alpha, tails,
                   critical, dec, plot_function=None):

    themed_box("**Step 3: Classical Method**")

    if tails == "left":
        st.latex(
            fr"\text{{Critical value}} = {critical:.{dec}f}"
        )
        st.markdown(
            f"Decision rule: Reject H₀ if "
            f"{symbol} < {critical:.{dec}f}."
        )
        reject = statistic < critical

    elif tails == "right":
        st.latex(
            fr"\text{{Critical value}} = {critical:.{dec}f}"
        )
        st.markdown(
            f"Decision rule: Reject H₀ if "
            f"{symbol} > {critical:.{dec}f}."
        )
        reject = statistic > critical

    else:
        low, high = critical

        st.latex(
            fr"\text{{Critical values}} = "
            fr"{low:.{dec}f},\ {high:.{dec}f}"
        )
        st.markdown(
            f"Decision rule: Reject H₀ if "
            f"{symbol} < {low:.{dec}f} or "
            f"{symbol} > {high:.{dec}f}."
        )
        reject = statistic < low or statistic > high

    st.markdown(
        f"Observed test statistic: "
        f"{symbol} = **{statistic:.{dec}f}**"
    )

    st.markdown(
        "Classical method decision: **"
        + ("Reject H₀" if reject else "Do not reject H₀")
        + "**"
    )

    if plot_function is not None:
        plot_function()

    return reject


# ==========================================================
# CONFIDENCE INTERVAL: MEANS AND PROPORTIONS
# ==========================================================
def show_confidence_interval(
    estimate, se, critical, confidence_level,
    dec, formula, substitution, parameter
):
    st.divider()
    st.markdown("### 📐 Confidence Interval")

    ci_message()

    st.markdown("**1. Confidence Level**")
    st.latex(
        fr"C = {100 * confidence_level:.2f}\%"
    )

    st.markdown("**2. Confidence Interval Formula**")
    st.latex(formula)

    st.markdown("**3. Critical Value**")
    st.latex(
        fr"\text{{Critical value}} = {critical:.{dec}f}"
    )

    st.markdown("**4. Substitute the Values**")
    st.latex(substitution)

    margin = critical * se
    lower = estimate - margin
    upper = estimate + margin

    st.markdown("**5. Margin of Error**")
    st.latex(r"E = \text{Critical Value}\times SE")
    st.latex(
        fr"E = ({critical:.{dec}f})"
        fr"({se:.{dec}f}) = {margin:.{dec}f}"
    )

    st.markdown("**6. Lower and Upper Endpoints**")
    st.latex(
        fr"L = {estimate:.{dec}f} - {margin:.{dec}f}"
        fr" = {lower:.{dec}f}"
    )
    st.latex(
        fr"U = {estimate:.{dec}f} + {margin:.{dec}f}"
        fr" = {upper:.{dec}f}"
    )

    st.markdown("**Final Confidence Interval**")
    st.latex(
        fr"\boxed{{({lower:.{dec}f},\ {upper:.{dec}f})}}"
    )

    st.write(
        f"We are {100 * confidence_level:.2f}% confident "
        f"that the interval contains {parameter}."
    )


# ==========================================================
# CONFIDENCE INTERVAL: F-TEST
# ==========================================================
def show_f_confidence_interval(F, df1, df2, confidence_level, dec):

    ci_alpha = 1 - confidence_level

    f_low = stats.f.ppf(ci_alpha / 2, df1, df2)
    f_high = stats.f.ppf(1 - ci_alpha / 2, df1, df2)

    lower = F / f_high
    upper = F / f_low

    st.divider()
    st.markdown("### 📐 Confidence Interval")

    ci_message()

    st.markdown("**1. Confidence Level**")
    st.latex(
        fr"C = {100 * confidence_level:.2f}\%"
    )

    st.markdown("**2. Confidence Interval Formula**")
    st.latex(
        r"\left("
        r"\frac{s_1^2/s_2^2}{F_{1-\alpha/2,df_1,df_2}},"
        r"\frac{s_1^2/s_2^2}{F_{\alpha/2,df_1,df_2}}"
        r"\right)"
    )

    st.markdown("**3. Critical Values**")
    st.latex(
        fr"F_{{\alpha/2}} = {f_low:.{dec}f}"
    )
    st.latex(
        fr"F_{{1-\alpha/2}} = {f_high:.{dec}f}"
    )

    st.markdown("**4. Substitute the Values**")
    st.latex(
        fr"\left("
        fr"\frac{{{F:.{dec}f}}}{{{f_high:.{dec}f}}},"
        fr"\frac{{{F:.{dec}f}}}{{{f_low:.{dec}f}}}"
        fr"\right)"
    )

    st.markdown("**5. Lower and Upper Endpoints**")
    st.latex(
        fr"L = \frac{{{F:.{dec}f}}}{{{f_high:.{dec}f}}}"
        fr" = {lower:.{dec}f}"
    )
    st.latex(
        fr"U = \frac{{{F:.{dec}f}}}{{{f_low:.{dec}f}}}"
        fr" = {upper:.{dec}f}"
    )

    st.markdown("**Final Confidence Interval**")
    st.latex(
        fr"\boxed{{({lower:.{dec}f},\ {upper:.{dec}f})}}"
    )

    st.write(
        f"We are {100 * confidence_level:.2f}% confident "
        "that the interval contains the ratio of population "
        "variances σ₁²/σ₂²."
    )


# ==========================================================
# FIVE-STEP HYPOTHESIS TEST
# ==========================================================
def show_five_steps(
    h0, h1, formula, substitutions,
    statistic, symbol, p_val, critical,
    alpha, tails, dec, show_picture,
    plot_function, context_reject, context_fail
):
    st.markdown("### 📘 Step-by-Step Solution")

    themed_box("**Step 1: Hypotheses**")
    st.latex(h0)
    st.latex(h1)

    themed_box("**Step 2: Test Statistic**")
    st.latex(formula)

    for expression in substitutions:
        st.latex(expression)

    st.latex(
        fr"{symbol} = {statistic:.{dec}f}"
    )

    show_classical(
        statistic, symbol, alpha, tails, critical, dec,
        plot_function if show_picture else None
    )

    themed_box("**Step 4: P-value Approach**")
    st.markdown(f"P-value = **{p_val:.{dec}f}**")
    st.markdown(f"α = **{alpha:.{dec}f}**")
    st.markdown(
        "P-value approach decision: **"
        + ("Reject H₀" if p_val < alpha else "Do not reject H₀")
        + "**"
    )

    themed_box("**Step 5: Conclusion**")
    st.write(
        conclusion_text(
            p_val < alpha, context_reject, context_fail
        )
    )


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

    dec = st.number_input(
        "Decimal places for output:",
        min_value=0, max_value=10, value=4
    )

    alpha = st.number_input(
        "Significance level (α):",
        min_value=0.001,
        max_value=0.5,
        value=0.005,
        step=0.001,
        format="%.3f"
    )

    tails = st.selectbox(
        "Tail type:", ["two", "left", "right"]
    )

    show_ci = st.checkbox(
        "Show Confidence Interval (always two-tailed)"
    )

    ci_conf_level = 0.995

    if show_ci:
        ci_conf_level = st.number_input(
            "Enter Confidence Interval confidence level (0–1):",
            min_value=0.001,
            max_value=0.999,
            value=0.995,
            step=0.001,
            format="%.3f"
        )

    show_picture = st.checkbox(
        "Show Classical Method picture",
        value=True
    )

    # ======================================================
    # TWO-PROPORTION Z-TEST
    # ======================================================
    if test_choice == "Two-Proportion Z-Test":

        x1 = st.number_input(
            "x₁ (successes in Sample 1):",
            min_value=0, step=1
        )
        n1 = st.number_input(
            "n₁ (size of Sample 1):",
            min_value=1, step=1
        )
        x2 = st.number_input(
            "x₂ (successes in Sample 2):",
            min_value=0, step=1
        )
        n2 = st.number_input(
            "n₂ (size of Sample 2):",
            min_value=1, step=1
        )

        if st.button("▶️ Run Test"):

            if x1 > n1 or x2 > n2:
                st.error(
                    "Each number of successes must be "
                    "less than or equal to its sample size."
                )
                return

            p1 = x1 / n1
            p2 = x2 / n2
            pooled = (x1 + x2) / (n1 + n2)

            se_test = np.sqrt(
                pooled * (1 - pooled) * (1/n1 + 1/n2)
            )

            if se_test <= 0:
                st.error("Standard error is zero.")
                return

            z = (p1 - p2) / se_test

            h0, h1 = get_tail_hypotheses(
                r"p_1 - p_2", tails
            )

            p_val, reject, critical = z_tail_metrics(
                z, alpha, tails
            )

            show_five_steps(
                h0, h1,
                r"z=\frac{\hat p_1-\hat p_2}"
                r"{\sqrt{\hat p(1-\hat p)"
                r"(\frac{1}{n_1}+\frac{1}{n_2})}}",
                [
                    fr"\hat p_1 = \frac{{{x1}}}{{{n1}}}"
                    fr" = {p1:.{dec}f}",
                    fr"\hat p_2 = \frac{{{x2}}}{{{n2}}}"
                    fr" = {p2:.{dec}f}",
                    fr"\hat p = {pooled:.{dec}f}"
                ],
                z, "z", p_val, critical,
                alpha, tails, dec, show_picture,
                lambda: plot_normal_rejection(
                    z, alpha, tails, critical
                ),
                "There is sufficient evidence to support "
                "the claim about the difference between "
                "the two population proportions.",
                "There is insufficient evidence to support "
                "the claim about the difference between "
                "the two population proportions."
            )

            if show_ci:
                se_ci = np.sqrt(
                    p1*(1-p1)/n1 + p2*(1-p2)/n2
                )
                zcrit = stats.norm.ppf(
                    1 - (1-ci_conf_level)/2
                )

                show_confidence_interval(
                    p1-p2, se_ci, zcrit,
                    ci_conf_level, dec,
                    r"(\hat{p}_1-\hat{p}_2)\pm "
                    r"z_{\alpha/2}"
                    r"\sqrt{\frac{\hat p_1(1-\hat p_1)}{n_1}"
                    r"+\frac{\hat p_2(1-\hat p_2)}{n_2}}",
                    fr"({p1:.{dec}f}-{p2:.{dec}f})"
                    fr"\pm({zcrit:.{dec}f})"
                    fr"({se_ci:.{dec}f})",
                    "the difference p₁ − p₂"
                )

    # ======================================================
    # PAIRED T-TEST (DATA)
    # ======================================================
    elif test_choice == "Paired t-Test (Data)":

        a = st.text_area("Sample 1:", "1,2,3")
        b = st.text_area("Sample 2:", "1,2,3")

        if st.button("▶️ Run Test"):

            x1 = parse_sample_text(a)
            x2 = parse_sample_text(b)

            if x1 is None or x2 is None:
                st.error("Enter valid numeric data.")
                return

            if len(x1) != len(x2):
                st.error(
                    "Both samples must have the same length."
                )
                return

            if len(x1) < 2:
                st.error(
                    "At least two paired observations are required."
                )
                return

            d = x1 - x2
            n = len(d)
            mean_d = np.mean(d)
            sd_d = np.std(d, ddof=1)
            se = sd_d / np.sqrt(n)

            st.markdown("### Paired Differences")
            st.latex(r"d_i = x_{1i} - x_{2i}")

            paired_df = pd.DataFrame({
                "Pair": np.arange(1, n+1),
                "Sample 1": x1,
                "Sample 2": x2,
                "Difference (d)": d
            })

            st.dataframe(
                paired_df,
                use_container_width=True,
                hide_index=True
            )

            if se <= 0:
                st.error(
                    "The standard error is zero. "
                    "The t-statistic cannot be calculated."
                )
                return

            tstat = mean_d / se
            df = n - 1

            h0, h1 = get_tail_hypotheses(
                r"\mu_d", tails
            )

            p_val, reject, critical = t_tail_metrics(
                tstat, df, alpha, tails
            )

            show_five_steps(
                h0, h1,
                r"t = \frac{\bar d-0}{s_d/\sqrt n}",
                [
                    fr"\bar d = {mean_d:.{dec}f}",
                    fr"s_d = {sd_d:.{dec}f},\quad n={n}",
                    fr"SE = {se:.{dec}f}"
                ],
                tstat, "t", p_val, critical,
                alpha, tails, dec, show_picture,
                lambda: plot_t_rejection(
                    tstat, df, alpha, tails, critical
                ),
                "There is sufficient evidence about "
                "the population mean paired difference.",
                "There is insufficient evidence about "
                "the population mean paired difference."
            )

            if show_ci:
                tcrit = stats.t.ppf(
                    1-(1-ci_conf_level)/2, df
                )

                show_confidence_interval(
                    mean_d, se, tcrit,
                    ci_conf_level, dec,
                    r"\bar d\pm t_{\alpha/2,n-1}"
                    r"\frac{s_d}{\sqrt n}",
                    fr"{mean_d:.{dec}f}"
                    fr"\pm({tcrit:.{dec}f})"
                    fr"({se:.{dec}f})",
                    "the population mean paired difference μd"
                )

    # ======================================================
    # PAIRED T-TEST (SUMMARY)
    # ======================================================
    elif test_choice == "Paired t-Test (Summary)":

        n = st.number_input(
            "n (number of pairs):", min_value=2, step=1
        )
        mean_d = st.number_input(
            "Mean of differences (d̄):", value=0.0
        )
        sd_d = st.number_input(
            "Standard deviation of differences (s_d):",
            min_value=0.0, value=1.0
        )

        if st.button("▶️ Run Test"):

            if sd_d <= 0:
                st.error("s_d must be greater than zero.")
                return

            se = sd_d / np.sqrt(n)
            tstat = mean_d / se
            df = n - 1

            h0, h1 = get_tail_hypotheses(
                r"\mu_d", tails
            )

            p_val, reject, critical = t_tail_metrics(
                tstat, df, alpha, tails
            )

            show_five_steps(
                h0, h1,
                r"t = \frac{\bar d-0}{s_d/\sqrt n}",
                [
                    fr"\bar d = {mean_d:.{dec}f}",
                    fr"s_d = {sd_d:.{dec}f},\quad n={n}",
                    fr"SE = {se:.{dec}f}"
                ],
                tstat, "t", p_val, critical,
                alpha, tails, dec, show_picture,
                lambda: plot_t_rejection(
                    tstat, df, alpha, tails, critical
                ),
                "There is sufficient evidence about "
                "the population mean paired difference.",
                "There is insufficient evidence about "
                "the population mean paired difference."
            )

            if show_ci:
                tcrit = stats.t.ppf(
                    1-(1-ci_conf_level)/2, df
                )

                show_confidence_interval(
                    mean_d, se, tcrit,
                    ci_conf_level, dec,
                    r"\bar d\pm t_{\alpha/2,n-1}"
                    r"\frac{s_d}{\sqrt n}",
                    fr"{mean_d:.{dec}f}"
                    fr"\pm({tcrit:.{dec}f})"
                    fr"({se:.{dec}f})",
                    "the population mean paired difference μd"
                )

    # ======================================================
    # INDEPENDENT T-TEST (DATA AND SUMMARY, WELCH)
    # ======================================================
    elif test_choice in [
        "Independent t-Test (Data, Welch)",
        "Independent t-Test (Summary, Welch)"
    ]:

        if test_choice == "Independent t-Test (Data, Welch)":
            a = st.text_area("Sample 1:", "1,2,3")
            b = st.text_area("Sample 2:", "4,5,6")

            if st.button("▶️ Run Test"):
                x1 = parse_sample_text(a)
                x2 = parse_sample_text(b)

                if x1 is None or x2 is None:
                    st.error("Enter valid numeric data.")
                    return

                if len(x1) < 2 or len(x2) < 2:
                    st.error(
                        "Each sample must have at least two values."
                    )
                    return

                n1, n2 = len(x1), len(x2)
                m1, m2 = np.mean(x1), np.mean(x2)
                s1 = np.std(x1, ddof=1)
                s2 = np.std(x2, ddof=1)
                run = True
            else:
                run = False

        else:
            n1 = st.number_input("n₁:", min_value=2, step=1)
            m1 = st.number_input("x̄₁:", value=0.0)
            s1 = st.number_input(
                "s₁:", min_value=0.0, value=1.0
            )
            n2 = st.number_input("n₂:", min_value=2, step=1)
            m2 = st.number_input("x̄₂:", value=0.0)
            s2 = st.number_input(
                "s₂:", min_value=0.0, value=1.0
            )
            run = st.button("▶️ Run Test")

        if run:
            v1 = s1**2/n1
            v2 = s2**2/n2
            se = np.sqrt(v1+v2)

            if se <= 0:
                st.error(
                    "Standard error must be greater than zero."
                )
                return

            df = (v1+v2)**2 / (
                v1**2/(n1-1) + v2**2/(n2-1)
            )

            tstat = (m1-m2)/se

            h0, h1 = get_tail_hypotheses(
                r"\mu_1 - \mu_2", tails
            )

            p_val, reject, critical = t_tail_metrics(
                tstat, df, alpha, tails
            )

            show_five_steps(
                h0, h1,
                r"t=\frac{(\bar x_1-\bar x_2)-0}"
                r"{\sqrt{\frac{s_1^2}{n_1}"
                r"+\frac{s_2^2}{n_2}}}",
                [
                    fr"\bar x_1={m1:.{dec}f},"
                    fr"\quad\bar x_2={m2:.{dec}f}",
                    fr"s_1={s1:.{dec}f},"
                    fr"\quad s_2={s2:.{dec}f}",
                    fr"n_1={n1},\quad n_2={n2}",
                    fr"df={df:.{dec}f}"
                ],
                tstat, "t", p_val, critical,
                alpha, tails, dec, show_picture,
                lambda: plot_t_rejection(
                    tstat, df, alpha, tails, critical
                ),
                "There is sufficient evidence about "
                "the difference between population means.",
                "There is insufficient evidence about "
                "the difference between population means."
            )

            if show_ci:
                tcrit = stats.t.ppf(
                    1-(1-ci_conf_level)/2, df
                )

                show_confidence_interval(
                    m1-m2, se, tcrit,
                    ci_conf_level, dec,
                    r"(\bar{x}_1-\bar{x}_2)"
                    r"\pm t_{\alpha/2,df}"
                    r"\sqrt{\frac{s_1^2}{n_1}"
                    r"+\frac{s_2^2}{n_2}}",
                    fr"({m1:.{dec}f}-{m2:.{dec}f})"
                    fr"\pm({tcrit:.{dec}f})"
                    fr"({se:.{dec}f})",
                    "the difference μ₁ − μ₂"
                )

    # ======================================================
    # F-TEST (DATA AND SUMMARY)
    # ======================================================
    elif test_choice in ["F-Test (Data)", "F-Test (Summary)"]:

        if test_choice == "F-Test (Data)":
            a = st.text_area("Sample 1:", "1,2,3")
            b = st.text_area("Sample 2:", "4,5,6")

            if st.button("▶️ Run Test"):
                x1 = parse_sample_text(a)
                x2 = parse_sample_text(b)

                if x1 is None or x2 is None:
                    st.error("Enter valid numeric data.")
                    return

                if len(x1) < 2 or len(x2) < 2:
                    st.error(
                        "Each sample must have at least two values."
                    )
                    return

                n1, n2 = len(x1), len(x2)
                s1 = np.std(x1, ddof=1)
                s2 = np.std(x2, ddof=1)
                run = True
            else:
                run = False

        else:
            n1 = st.number_input("n₁:", min_value=2, step=1)
            s1 = st.number_input(
                "s₁:", min_value=0.0, value=1.0
            )
            n2 = st.number_input("n₂:", min_value=2, step=1)
            s2 = st.number_input(
                "s₂:", min_value=0.0, value=1.0
            )
            run = st.button("▶️ Run Test")

        if run:
            if s2 <= 0:
                st.error(
                    "The second sample standard deviation "
                    "must be greater than zero."
                )
                return

            df1, df2 = n1-1, n2-1
            F = s1**2/s2**2

            h0, h1 = get_f_hypotheses(tails)

            p_val, reject, critical = f_tail_metrics(
                F, df1, df2, alpha, tails
            )

            show_five_steps(
                h0, h1,
                r"F=\frac{s_1^2}{s_2^2}",
                [
                    fr"s_1={s1:.{dec}f},"
                    fr"\quad s_2={s2:.{dec}f}",
                    fr"df_1={df1},\quad df_2={df2}",
                    fr"F=\frac{{{s1:.{dec}f}^2}}"
                    fr"{{{s2:.{dec}f}^2}}"
                ],
                F, "F", p_val, critical,
                alpha, tails, dec, show_picture,
                lambda: plot_f_rejection(
                    F, df1, df2, alpha, tails, critical
                ),
                "There is sufficient evidence about "
                "the population standard deviations.",
                "There is insufficient evidence about "
                "the population standard deviations."
            )

            if show_ci:
                show_f_confidence_interval(
                    F, df1, df2, ci_conf_level, dec
                )


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    run_two_sample_tool()

