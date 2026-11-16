
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ==========================================================
# two_sample_tool.py
# MIND: Statistics Visualizer Suite
# Two-Sample Hypothesis Testing
# ==========================================================

# -----------------------------
# Helpers
# -----------------------------
def parse_sample_text(text):
    try:
        values = [
            float(v.strip())
            for v in text.replace("\n", ",").split(",")
            if v.strip()
        ]
        if not np.all(np.isfinite(values)):
            return None
        return np.array(values, dtype=float)
    except ValueError:
        return None


def themed_box(text):
    st.markdown(f"**{text}**")


def get_tail_hypotheses(parameter, tails):
    if parameter == "means":
        a, b = r"\mu_1", r"\mu_2"
        h0 = fr"H_0: {a}={b}"
        if tails == "left":
            h1 = fr"H_1: {a}<{b}"
        elif tails == "right":
            h1 = fr"H_1: {a}>{b}"
        else:
            h1 = fr"H_1: {a}\ne {b}"

    elif parameter == "proportions":
        a, b = r"p_1", r"p_2"
        h0 = fr"H_0: {a}={b}"
        if tails == "left":
            h1 = fr"H_1: {a}<{b}"
        elif tails == "right":
            h1 = fr"H_1: {a}>{b}"
        else:
            h1 = fr"H_1: {a}\ne {b}"

    elif parameter == "variances":
        a, b = r"\sigma_1", r"\sigma_2"
        h0 = fr"H_0: {a}={b}"
        if tails == "left":
            h1 = fr"H_1: {a}<{b}"
        elif tails == "right":
            h1 = fr"H_1: {a}>{b}"
        else:
            h1 = fr"H_1: {a}\ne {b}"

    else:
        h0 = r"H_0: \mu_d=0"
        if tails == "left":
            h1 = r"H_1: \mu_d<0"
        elif tails == "right":
            h1 = r"H_1: \mu_d>0"
        else:
            h1 = r"H_1: \mu_d\ne 0"

    return h0, h1


def tail_metrics(value, dist, alpha, tails):
    if tails == "left":
        p = dist.cdf(value)
        critical = dist.ppf(alpha)
        reject = value < critical

    elif tails == "right":
        p = dist.sf(value)
        critical = dist.ppf(1 - alpha)
        reject = value > critical

    else:
        p = 2 * min(dist.cdf(value), dist.sf(value))
        low = dist.ppf(alpha / 2)
        high = dist.ppf(1 - alpha / 2)
        critical = (low, high)
        reject = value < low or value > high

    return min(1.0, float(p)), bool(reject), critical


def plot_rejection(statistic, dist, alpha, tails, critical,
                   symbol="t", df_label=""):
    if symbol == "F":
        upper = max(
            5.0, statistic * 1.25,
            dist.ppf(0.999) if np.isfinite(dist.ppf(0.999)) else 5
        )
        x = np.linspace(0.0001, upper, 900)
    else:
        left = min(-5, statistic - 1)
        right = max(5, statistic + 1)
        x = np.linspace(left, right, 900)

    y = dist.pdf(x)
    fig, ax = plt.subplots(figsize=(8, 4))

    if tails == "left":
        reject_mask = x < critical
        ax.axvline(critical, color="black", linestyle="--")

    elif tails == "right":
        reject_mask = x > critical
        ax.axvline(critical, color="black", linestyle="--")

    else:
        reject_mask = (x < critical[0]) | (x > critical[1])
        ax.axvline(critical[0], color="black", linestyle="--")
        ax.axvline(critical[1], color="black", linestyle="--")

    ax.fill_between(
        x, y, where=reject_mask,
        color="#d62728", alpha=0.65
    )
    ax.fill_between(
        x, y, where=~reject_mask,
        color="#2ca02c", alpha=0.45
    )

    ax.plot(x, y, color="black", linewidth=2)
    ax.axvline(
        statistic, color="#1f77b4",
        linewidth=2.5, label=f"{symbol} = {statistic:.4f}"
    )

    ax.set_title("Classical Method: Rejection Region " + df_label)
    ax.set_xlabel(symbol)
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Red = Reject H₀ region | "
        "Green = Do not reject H₀ region | "
        "Blue = Observed test statistic"
    )


def show_ci(estimate, se, critical, level, dec, formula,
            substitution, parameter):
    margin = critical * se
    lower = estimate - margin
    upper = estimate + margin

    st.markdown("### Confidence Interval")
    st.write(f"Confidence Level: **{100 * level:.1f}%**")
    st.latex(formula)
    st.markdown("**Substitute the sample values:**")
    st.latex(substitution)
    st.latex(
        fr"E=({critical:.{dec}f})({se:.{dec}f})"
        fr"={margin:.{dec}f}"
    )
    st.latex(
        fr"({estimate:.{dec}f}-{margin:.{dec}f},"
        fr"\ {estimate:.{dec}f}+{margin:.{dec}f})"
    )
    st.latex(
        fr"\boxed{{({lower:.{dec}f},\ {upper:.{dec}f})}}"
    )
    st.write(f"Lower endpoint: **{lower:.{dec}f}**")
    st.write(f"Upper endpoint: **{upper:.{dec}f}**")
    st.write(
        f"We are {100 * level:.1f}% confident that "
        f"the interval contains {parameter}."
    )


def show_f_ci(F, df1, df2, level, dec):
    a = 1 - level
    f_lower_critical = stats.f.ppf(a / 2, df1, df2)
    f_upper_critical = stats.f.ppf(1 - a / 2, df1, df2)

    lower = F / f_upper_critical
    upper = F / f_lower_critical

    st.markdown("### Confidence Interval")
    st.write(f"Confidence Level: **{100 * level:.1f}%**")
    st.latex(
        r"\left("
        r"\frac{s_1^2/s_2^2}{F_{1-\alpha/2,df_1,df_2}},"
        r"\frac{s_1^2/s_2^2}{F_{\alpha/2,df_1,df_2}}"
        r"\right)"
    )
    st.markdown("**Substitute the sample values:**")
    st.latex(
        fr"\left("
        fr"\frac{{{F:.{dec}f}}}{{{f_upper_critical:.{dec}f}}},"
        fr"\frac{{{F:.{dec}f}}}{{{f_lower_critical:.{dec}f}}}"
        fr"\right)"
    )
    st.latex(
        fr"\boxed{{({lower:.{dec}f},\ {upper:.{dec}f})}}"
    )
    st.write(f"Lower endpoint: **{lower:.{dec}f}**")
    st.write(f"Upper endpoint: **{upper:.{dec}f}**")
    st.write(
        f"We are {100 * level:.1f}% confident that "
        "this interval contains the ratio of population "
        "variances σ₁²/σ₂²."
    )


def show_five_steps(h0, h1, formula, details, statistic,
                    symbol, dist, alpha, tails, dec,
                    show_picture, conclusion, df_label=""):

    st.markdown("### 📘 Step-by-Step Solution")

    themed_box("Step 1: Hypotheses")
    st.latex(h0)
    st.latex(h1)

    themed_box("Step 2: Test Statistic")
    st.latex(formula)
    for item in details:
        st.latex(item)
    st.latex(fr"{symbol}={statistic:.{dec}f}")

    p, reject, critical = tail_metrics(
        statistic, dist, alpha, tails
    )

    themed_box("Step 3: Classical Method")
    if tails == "two":
        st.latex(
            fr"\text{{Critical values}}="
            fr"{critical[0]:.{dec}f},\ {critical[1]:.{dec}f}"
        )
        st.write(
            f"Reject H₀ if {symbol} < {critical[0]:.{dec}f} "
            f"or {symbol} > {critical[1]:.{dec}f}."
        )
    else:
        st.latex(
            fr"\text{{Critical value}}={critical:.{dec}f}"
        )
        sign = "<" if tails == "left" else ">"
        st.write(
            f"Reject H₀ if {symbol} {sign} {critical:.{dec}f}."
        )

    st.write(
        "Classical method decision: **" +
        ("Reject H₀" if reject else "Do not reject H₀") +
        "**"
    )

    if show_picture:
        plot_rejection(
            statistic, dist, alpha, tails,
            critical, symbol, df_label
        )

    themed_box("Step 4: P-value Approach")
    st.write(f"P-value = **{p:.{dec}f}**")
    st.write(f"α = **{alpha:.{dec}f}**")
    st.write(
        "P-value decision: **" +
        ("Reject H₀" if reject else "Do not reject H₀") +
        "**"
    )

    themed_box("Step 5: Conclusion")
    if reject:
        st.write(
            "Since the p-value is less than α, reject H₀. "
            "There is sufficient evidence to support H₁ "
            + conclusion
        )
    else:
        st.write(
            "Since the p-value is greater than or equal to α, "
            "do not reject H₀. There is insufficient evidence "
            "to support H₁ " + conclusion
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
        "Decimal places for output:", 0, 10, 4
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

    show_confidence = st.checkbox(
        "Show Confidence Interval (always two-tailed)"
    )

    ci_level = 0.995
    if show_confidence:
        ci_level = st.number_input(
            "Enter Confidence Interval confidence level (0–1):",
            min_value=0.001,
            max_value=0.999,
            value=0.995,
            step=0.001,
            format="%.3f"
        )

    show_picture = st.checkbox(
        "Show Classical Method picture", value=True
    )

    # ======================================================
    # TWO-PROPORTION Z-TEST
    # ======================================================
    if test_choice == "Two-Proportion Z-Test":
        x1 = st.number_input(
            "x₁ (successes in Sample 1):", min_value=0, step=1
        )
        n1 = st.number_input(
            "n₁ (size of Sample 1):", min_value=1, step=1
        )
        x2 = st.number_input(
            "x₂ (successes in Sample 2):", min_value=0, step=1
        )
        n2 = st.number_input(
            "n₂ (size of Sample 2):", min_value=1, step=1
        )

        if st.button("▶️ Run Test"):
            if x1 > n1 or x2 > n2:
                st.error("Successes cannot exceed sample size.")
                return

            p1 = x1 / n1
            p2 = x2 / n2
            pooled = (x1 + x2) / (n1 + n2)
            se = np.sqrt(
                pooled * (1 - pooled) * (1/n1 + 1/n2)
            )
            if se <= 0:
                st.error("The pooled standard error is zero.")
                return

            z = (p1 - p2) / se
            h0, h1 = get_tail_hypotheses("proportions", tails)

            show_five_steps(
                h0, h1,
                r"z=\frac{\hat p_1-\hat p_2}"
                r"{\sqrt{\hat p(1-\hat p)(1/n_1+1/n_2)}}",
                [
                    fr"\hat p_1={p1:.{dec}f},"
                    fr"\quad\hat p_2={p2:.{dec}f}",
                    fr"\hat p={pooled:.{dec}f}",
                    fr"SE={se:.{dec}f}"
                ],
                z, "z", stats.norm(),
                alpha, tails, dec, show_picture,
                "about the difference between population proportions."
            )

            if show_confidence:
                se_ci = np.sqrt(
                    p1*(1-p1)/n1 + p2*(1-p2)/n2
                )
                zcrit = stats.norm.ppf(
                    1 - (1-ci_level)/2
                )
                show_ci(
                    p1-p2, se_ci, zcrit, ci_level, dec,
                    r"(\hat p_1-\hat p_2)\pm"
                    r"z_{\alpha/2}"
                    r"\sqrt{\frac{\hat p_1(1-\hat p_1)}{n_1}"
                    r"+\frac{\hat p_2(1-\hat p_2)}{n_2}}",
                    fr"({p1:.{dec}f}-{p2:.{dec}f})"
                    fr"\pm({zcrit:.{dec}f})({se_ci:.{dec}f})",
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
                st.error("Both samples must have equal lengths.")
                return
            if len(x1) < 2:
                st.error("At least two pairs are required.")
                return

            d = x1 - x2
            mean_d = np.mean(d)
            sd_d = np.std(d, ddof=1)
            n = len(d)
            se = sd_d / np.sqrt(n)

            st.markdown("### Paired Differences")
            st.latex(r"d_i=x_{1i}-x_{2i}")
            paired_df = pd.DataFrame({
                "Pair": np.arange(1, n+1),
                "Sample 1": x1,
                "Sample 2": x2,
                "Difference (d)": d
            })
            st.dataframe(
                paired_df, use_container_width=True,
                hide_index=True
            )
            st.latex(
                fr"\bar d={mean_d:.{dec}f},"
                fr"\quad s_d={sd_d:.{dec}f},\quad n={n}"
            )

            if se <= 0:
                st.error(
                    "The standard error is zero. "
                    "A t-statistic cannot be calculated."
                )
                return

            tstat = mean_d / se
            df = n - 1
            h0, h1 = get_tail_hypotheses("paired", tails)

            show_five_steps(
                h0, h1,
                r"t=\frac{\bar d-0}{s_d/\sqrt n}",
                [
                    fr"\bar d={mean_d:.{dec}f}",
                    fr"s_d={sd_d:.{dec}f},\quad n={n}",
                    fr"SE={se:.{dec}f}"
                ],
                tstat, "t", stats.t(df),
                alpha, tails, dec, show_picture,
                "about the population mean paired difference.",
                f"(df={df})"
            )

            if show_confidence:
                tcrit = stats.t.ppf(
                    1-(1-ci_level)/2, df
                )
                show_ci(
                    mean_d, se, tcrit, ci_level, dec,
                    r"\bar d\pm t_{\alpha/2,n-1}"
                    r"\frac{s_d}{\sqrt n}",
                    fr"{mean_d:.{dec}f}"
                    fr"\pm({tcrit:.{dec}f})({se:.{dec}f})",
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
            df = n-1
            h0, h1 = get_tail_hypotheses("paired", tails)

            show_five_steps(
                h0, h1,
                r"t=\frac{\bar d-0}{s_d/\sqrt n}",
                [
                    fr"\bar d={mean_d:.{dec}f}",
                    fr"s_d={sd_d:.{dec}f},\quad n={n}",
                    fr"SE={se:.{dec}f}"
                ],
                tstat, "t", stats.t(df),
                alpha, tails, dec, show_picture,
                "about the population mean paired difference.",
                f"(df={df})"
            )

            if show_confidence:
                tcrit = stats.t.ppf(
                    1-(1-ci_level)/2, df
                )
                show_ci(
                    mean_d, se, tcrit, ci_level, dec,
                    r"\bar d\pm t_{\alpha/2,n-1}"
                    r"\frac{s_d}{\sqrt n}",
                    fr"{mean_d:.{dec}f}"
                    fr"\pm({tcrit:.{dec}f})({se:.{dec}f})",
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
                    st.error("Each sample needs at least two values.")
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
                st.error("Standard error must be positive.")
                return

            df = (v1+v2)**2 / (
                v1**2/(n1-1) + v2**2/(n2-1)
            )
            tstat = (m1-m2)/se
            h0, h1 = get_tail_hypotheses("means", tails)

            show_five_steps(
                h0, h1,
                r"t=\frac{(\bar x_1-\bar x_2)-0}"
                r"{\sqrt{s_1^2/n_1+s_2^2/n_2}}",
                [
                    fr"\bar x_1={m1:.{dec}f},"
                    fr"\quad\bar x_2={m2:.{dec}f}",
                    fr"s_1={s1:.{dec}f},"
                    fr"\quad s_2={s2:.{dec}f}",
                    fr"n_1={n1},\quad n_2={n2}",
                    fr"SE={se:.{dec}f}",
                    fr"df={df:.{dec}f}"
                ],
                tstat, "t", stats.t(df),
                alpha, tails, dec, show_picture,
                "about the difference between population means.",
                f"(df={df:.2f})"
            )

            if show_confidence:
                tcrit = stats.t.ppf(
                    1-(1-ci_level)/2, df
                )
                show_ci(
                    m1-m2, se, tcrit, ci_level, dec,
                    r"(\bar x_1-\bar x_2)\pm"
                    r"t_{\alpha/2,df}"
                    r"\sqrt{\frac{s_1^2}{n_1}"
                    r"+\frac{s_2^2}{n_2}}",
                    fr"({m1:.{dec}f}-{m2:.{dec}f})"
                    fr"\pm({tcrit:.{dec}f})({se:.{dec}f})",
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
                    st.error("Each sample needs at least two values.")
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
            h0, h1 = get_tail_hypotheses("variances", tails)

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
                F, "F", stats.f(df1, df2),
                alpha, tails, dec, show_picture,
                "about the population standard deviations.",
                f"(df1={df1}, df2={df2})"
            )

            if show_confidence:
                show_f_ci(F, df1, df2, ci_level, dec)


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    run_two_sample_tool()
