# ==========================================================
# confidence_intervals_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# MIND: Statistics Visualizer Suite
# Updated with Dark/Light Mode Safe Interpretation Boxes
# Improved Step-by-Step Pedagogical Format
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

# ==========================================================
# Helper Functions
# ==========================================================

def round_value(value, decimals=4):
    return round(float(value), decimals)

def load_uploaded_data():
    uploaded_file = st.file_uploader(
        "📂 Upload CSV or Excel file (single numeric column)",
        type=["csv", "xlsx"]
    )
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    return df[col].dropna().to_numpy()
            st.error("❌ No numeric column found in uploaded file.")
        except Exception as e:
            st.error(f"⚠️ Error reading file: {e}")
    return None


# ==========================================================
# Dark/Light Mode Safe Interpretation Box
# ==========================================================

def interpretation_box(html_text):
    st.markdown(
        f"""
        <div class="interp-box">
            {html_text}
        </div>

        <style>
        /* LIGHT MODE */
        @media (prefers-color-scheme: light) {{
            .interp-box {{
                background-color: #e6f3ff;
                color: #000000;
                padding: 12px;
                border-radius: 10px;
                border: 1px solid #bcdcff;
            }}
        }}

        /* DARK MODE */
        @media (prefers-color-scheme: dark) {{
            .interp-box {{
                background-color: #2b2b2b;
                color: #ffffff;
                padding: 12px;
                border-radius: 10px;
                border: 1px solid #444444;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# Main App
# ==========================================================
def run():
    st.header("🔮 MIND: Confidence Interval Calculator (Step-by-Step Edition)")
    st.markdown("---")

    categories = [
        "Confidence Interval for Proportion (p, z)",
        "Sample Size for Proportion (p, z, E)",
        "Confidence Interval for Mean (σ known, z)",
        "Confidence Interval for Mean (s given, t)",
        "Confidence Interval for Mean (with data, t)",
        "Sample Size for Mean (σ known, z, E)",
        "Confidence Interval for Variance & SD (χ²)",
        "Confidence Interval for Variance & SD (with data, χ²)"
    ]

    choice = st.selectbox(
        "Choose a category:",
        categories,
        index=None,
        placeholder="Select a confidence interval type..."
    )

    if not choice:
        st.info("👆 Please select a category to begin.")
        return

    decimal = st.number_input("Decimal places for output", min_value=0, max_value=10, value=4)

    # ==========================================================
    # 1) Confidence Interval for Proportion (p, z)
    # ==========================================================
    if choice == categories[0]:

        st.latex(r"""
            \text{CI: } 
            \hat{p} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
        """)

        x = st.number_input("Number of successes (x)", min_value=0, step=1)
        n = st.number_input("Sample size (n)", min_value=1, step=1)
        conf = st.number_input("Confidence level (0–1)", value=0.950, step=0.001, format="%.3f")

        if st.button("👨‍💻 Calculate"):

            if x > n:
                st.error("❌ Number of successes (x) cannot exceed sample size (n).")
                return

            p_hat = x / n
            alpha = 1 - conf
            z = stats.norm.ppf((1 + conf) / 2)
            se = np.sqrt(p_hat * (1 - p_hat) / n)
            moe = z * se
            lower, upper = p_hat - moe, p_hat + moe

            st.subheader("Step-by-Step Solution")

            st.markdown("**Step 1: Identify the confidence interval formula**")
            st.latex(r"\hat{p} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}")

            st.markdown("**Step 2: Compute the sample proportion**")
            st.latex(fr"\hat{{p}} = \frac{{x}}{{n}} = \frac{{{x}}}{{{n}}} = {p_hat:.{decimal}f}")

            st.markdown("**Step 3: Compute α and the critical value**")
            st.latex(fr"\alpha = 1 - {conf:.3f} = {alpha:.{decimal}f}")
            st.latex(fr"\alpha/2 = {alpha/2:.{decimal}f}")
            st.latex(fr"z_{{\alpha/2}} = {z:.{decimal}f}")

            st.markdown("**Step 4: Compute the standard error**")
            st.latex(fr"SE = \sqrt{{\frac{{\hat{{p}}(1-\hat{{p}})}}{{n}}}}")
            st.latex(
                fr"SE = \sqrt{{\frac{{({p_hat:.{decimal}f})(1-{p_hat:.{decimal}f})}}{{{n}}}}} = {se:.{decimal}f}"
            )

            st.markdown("**Step 5: Compute the margin of error**")
            st.latex(fr"E = z_{{\alpha/2}} \cdot SE")
            st.latex(fr"E = ({z:.{decimal}f})({se:.{decimal}f}) = {moe:.{decimal}f}")

            st.markdown("**Step 6: Construct the confidence interval**")
            st.latex(fr"\hat{{p}} \pm E = {p_hat:.{decimal}f} \pm {moe:.{decimal}f}")
            st.latex(fr"({lower:.{decimal}f},\; {upper:.{decimal}f})")

            interpretation_box(
                f"We are <b>{conf*100:.3f}% confident</b> that the true population "
                f"proportion lies between <b>{lower:.{decimal}f}</b> and "
                f"<b>{upper:.{decimal}f}</b>."
            )


    # ==========================================================
    # 2) Sample Size for Proportion (p, z, E)
    # ==========================================================
    elif choice == categories[1]:

        st.latex(r"""
            n = \hat{p}(1-\hat{p})\left(\frac{z_{\alpha/2}}{E}\right)^2
        """)

        conf = st.number_input("Confidence level", value=0.950, step=0.001, format="%.3f")
        p_est = st.number_input("Estimated proportion (p̂)", value=0.500, step=0.001, format="%.3f")
        E = st.number_input("Margin of error (E)", value=0.025, step=0.001, format="%.3f")

        if st.button("👨‍💻 Calculate"):

            alpha = 1 - conf
            z = stats.norm.ppf((1 + conf) / 2)
            n_req = p_est * (1 - p_est) * (z / E) ** 2
            n_round = int(np.ceil(n_req))

            st.subheader("Step-by-Step Solution")

            st.markdown("**Step 1: Identify the sample size formula**")
            st.latex(r"n = \hat{p}(1-\hat{p})\left(\frac{z_{\alpha/2}}{E}\right)^2")

            st.markdown("**Step 2: Compute α and the critical value**")
            st.latex(fr"\alpha = 1 - {conf:.3f} = {alpha:.{decimal}f}")
            st.latex(fr"\alpha/2 = {alpha/2:.{decimal}f}")
            st.latex(fr"z_{{\alpha/2}} = {z:.{decimal}f}")

            st.markdown("**Step 3: Substitute the known values**")
            st.latex(
                fr"n = ({p_est:.{decimal}f})(1-{p_est:.{decimal}f})\left(\frac{{{z:.{decimal}f}}}{{{E:.{decimal}f}}}\right)^2"
            )

            st.markdown("**Step 4: Compute the required sample size**")
            st.latex(fr"n = {n_req:.{decimal}f}")

            st.markdown("**Step 5: Round up to the next whole number**")
            st.latex(fr"n = {n_round}")

            interpretation_box(
                f"A minimum of <b>{n_round}</b> participants is needed to achieve "
                f"<b>{conf*100:.3f}% confidence</b> with margin of error <b>{E:.3f}</b>."
            )


    # ==========================================================
    # 3) CI for Mean (σ known, z)
    # ==========================================================
    elif choice == categories[2]:

        st.latex(r"""
            \bar{X} \pm z_{\alpha/2}\left(\frac{\sigma}{\sqrt{n}}\right)
        """)

        mean = st.number_input("Sample mean (x̄)")
        sigma = st.number_input("Population SD (σ)", min_value=0.0)
        n = st.number_input("Sample size (n)", min_value=1)
        conf = st.number_input("Confidence level", value=0.950, step=0.001, format="%.3f")

        if st.button("👨‍💻 Calculate"):

            alpha = 1 - conf
            z = stats.norm.ppf((1 + conf) / 2)
            se = sigma / np.sqrt(n)
            moe = z * se
            lower, upper = mean - moe, mean + moe

            st.subheader("Step-by-Step Solution")

            st.markdown("**Step 1: Identify the confidence interval formula**")
            st.latex(r"\bar{X} \pm z_{\alpha/2}\left(\frac{\sigma}{\sqrt{n}}\right)")

            st.markdown("**Step 2: List the known values**")
            st.latex(fr"\bar{{x}} = {mean:.{decimal}f}, \quad \sigma = {sigma:.{decimal}f}, \quad n = {n}")

            st.markdown("**Step 3: Compute α and the critical value**")
            st.latex(fr"\alpha = 1 - {conf:.3f} = {alpha:.{decimal}f}")
            st.latex(fr"\alpha/2 = {alpha/2:.{decimal}f}")
            st.latex(fr"z_{{\alpha/2}} = {z:.{decimal}f}")

            st.markdown("**Step 4: Compute the standard error**")
            st.latex(r"SE = \frac{\sigma}{\sqrt{n}}")
            st.latex(fr"SE = \frac{{{sigma:.{decimal}f}}}{{\sqrt{{{n}}}}} = {se:.{decimal}f}")

            st.markdown("**Step 5: Compute the margin of error**")
            st.latex(r"E = z_{\alpha/2}\left(\frac{\sigma}{\sqrt{n}}\right)")
            st.latex(fr"E = ({z:.{decimal}f})({se:.{decimal}f}) = {moe:.{decimal}f}")

            st.markdown("**Step 6: Construct the confidence interval**")
            st.latex(fr"\bar{{x}} \pm E = {mean:.{decimal}f} \pm {moe:.{decimal}f}")
            st.latex(fr"({lower:.{decimal}f}, {upper:.{decimal}f})")

            interpretation_box(
                f"We are <b>{conf*100:.3f}% confident</b> that μ lies between "
                f"<b>{lower:.{decimal}f}</b> and <b>{upper:.{decimal}f}</b>."
            )


    # ==========================================================
    # 4) CI for Mean (s given, t)
    # ==========================================================
    elif choice == categories[3]:

        st.latex(r"""
            \bar{X} \pm t_{\alpha/2,\,n-1}\left(\frac{s}{\sqrt{n}}\right)
        """)

        mean = st.number_input("Sample mean (x̄)")
        s = st.number_input("Sample SD (s)")
        n = st.number_input("Sample size (n)", min_value=2)
        conf = st.number_input("Confidence level", value=0.950, step=0.001, format="%.3f")

        if st.button("👨‍💻 Calculate"):

            df = int(n - 1)
            alpha = 1 - conf
            tcrit = stats.t.ppf((1 + conf) / 2, df)
            se = s / np.sqrt(n)
            moe = tcrit * se
            lower, upper = mean - moe, mean + moe

            st.subheader("Step-by-Step Solution")

            st.markdown("**Step 1: Identify the confidence interval formula**")
            st.latex(r"\bar{X} \pm t_{\alpha/2,\,n-1}\left(\frac{s}{\sqrt{n}}\right)")

            st.markdown("**Step 2: List the known values**")
            st.latex(fr"\bar{{x}} = {mean:.{decimal}f}, \quad s = {s:.{decimal}f}, \quad n = {n}")

            st.markdown("**Step 3: Compute the degrees of freedom**")
            st.latex(fr"df = n - 1 = {n} - 1 = {df}")

            st.markdown("**Step 4: Compute α and the critical value**")
            st.latex(fr"\alpha = 1 - {conf:.3f} = {alpha:.{decimal}f}")
            st.latex(fr"\alpha/2 = {alpha/2:.{decimal}f}")
            st.latex(fr"t_{{\alpha/2,{df}}} = {tcrit:.{decimal}f}")

            st.markdown("**Step 5: Compute the standard error**")
            st.latex(r"SE = \frac{s}{\sqrt{n}}")
            st.latex(fr"SE = \frac{{{s:.{decimal}f}}}{{\sqrt{{{n}}}}} = {se:.{decimal}f}")

            st.markdown("**Step 6: Compute the margin of error**")
            st.latex(r"E = t_{\alpha/2,\,n-1}\left(\frac{s}{\sqrt{n}}\right)")
            st.latex(fr"E = ({tcrit:.{decimal}f})({se:.{decimal}f}) = {moe:.{decimal}f}")

            st.markdown("**Step 7: Construct the confidence interval**")
            st.latex(fr"\bar{{x}} \pm E = {mean:.{decimal}f} \pm {moe:.{decimal}f}")
            st.latex(fr"({lower:.{decimal}f}, {upper:.{decimal}f})")

            interpretation_box(
                f"We are <b>{conf*100:.3f}% confident</b> that μ lies between "
                f"<b>{lower:.{decimal}f}</b> and <b>{upper:.{decimal}f}</b>."
            )

    # ==========================================================
    # 5) CI for Mean (with data)
    # ==========================================================
    elif choice == categories[4]:

        st.latex(r"""
            \bar{X} \pm t_{\alpha/2,\,n-1}\left(\frac{s}{\sqrt{n}}\right)
        """)

        data = load_uploaded_data()
        raw = st.text_area("Or enter comma-separated values:")

        if data is None and raw:
            try:
                data = np.array([float(x) for x in raw.split(",")])
            except:
                st.error("❌ Invalid input.")
                return

        conf = st.number_input("Confidence level", value=0.950, step=0.001, format="%.3f")

        if st.button("👨‍💻 Calculate"):

            if data is None or len(data) < 2:
                st.warning("⚠️ Need at least 2 numbers.")
                return

            n = len(data)
            df = n - 1
            mean = np.mean(data)
            s = np.std(data, ddof=1)
            alpha = 1 - conf
            tcrit = stats.t.ppf((1 + conf) / 2, df)
            se = s / np.sqrt(n)
            moe = tcrit * se
            lower, upper = mean - moe, mean + moe

            st.subheader("Step-by-Step Solution")

            st.markdown("**Step 1: Identify the confidence interval formula**")
            st.latex(r"\bar{X} \pm t_{\alpha/2,\,n-1}\left(\frac{s}{\sqrt{n}}\right)")

            st.markdown("**Step 2: Compute the sample statistics**")
            st.latex(fr"n = {n}")
            st.latex(fr"\bar{{x}} = {mean:.{decimal}f}")
            st.latex(fr"s = {s:.{decimal}f}")

            st.markdown("**Step 3: Compute the degrees of freedom**")
            st.latex(fr"df = n - 1 = {n} - 1 = {df}")

            st.markdown("**Step 4: Compute α and find the critical value**")
            st.latex(fr"\alpha = 1 - {conf:.3f} = {alpha:.{decimal}f}")
            st.latex(fr"\alpha/2 = {alpha/2:.{decimal}f}")
            st.latex(fr"t_{{\alpha/2,{df}}} = {tcrit:.{decimal}f}")

            st.markdown("**Step 5: Compute the standard error**")
            st.latex(r"SE = \frac{s}{\sqrt{n}}")
            st.latex(fr"SE = \frac{{{s:.{decimal}f}}}{{\sqrt{{{n}}}}} = {se:.{decimal}f}")

            st.markdown("**Step 6: Compute the margin of error**")
            st.latex(r"E = t_{\alpha/2,\,n-1}\left(\frac{s}{\sqrt{n}}\right)")
            st.latex(fr"E = ({tcrit:.{decimal}f})({se:.{decimal}f}) = {moe:.{decimal}f}")

            st.markdown("**Step 7: Construct the confidence interval**")
            st.latex(fr"\bar{{x}} \pm E = {mean:.{decimal}f} \pm {moe:.{decimal}f}")
            st.latex(fr"({lower:.{decimal}f}, {upper:.{decimal}f})")

            interpretation_box(
                f"We are <b>{conf*100:.3f}% confident</b> that the population mean μ "
                f"lies between <b>{lower:.{decimal}f}</b> and <b>{upper:.{decimal}f}</b>."
            )

    # ==========================================================
    # 6) Sample Size for Mean (σ known)
    # ==========================================================
    elif choice == categories[5]:

        st.latex(r"""
            n = \left(\frac{z_{\alpha/2}\sigma}{E}\right)^2
        """)

        conf = st.number_input("Confidence level", value=0.950, step=0.001, format="%.3f")
        sigma = st.number_input("Population SD (σ)", min_value=0.0)
        E = st.number_input("Margin of error (E)", value=0.050, step=0.001, format="%.3f")

        if st.button("👨‍💻 Calculate"):

            alpha = 1 - conf
            z = stats.norm.ppf((1 + conf) / 2)
            n_req = (z * sigma / E) ** 2
            n_round = int(np.ceil(n_req))

            st.subheader("Step-by-Step Solution")

            st.markdown("**Step 1: Identify the sample size formula**")
            st.latex(r"n = \left(\frac{z_{\alpha/2}\sigma}{E}\right)^2")

            st.markdown("**Step 2: Compute α and the critical value**")
            st.latex(fr"\alpha = 1 - {conf:.3f} = {alpha:.{decimal}f}")
            st.latex(fr"\alpha/2 = {alpha/2:.{decimal}f}")
            st.latex(fr"z_{{\alpha/2}} = {z:.{decimal}f}")

            st.markdown("**Step 3: Substitute the known values**")
            st.latex(fr"n = \left(\frac{{({z:.{decimal}f})({sigma:.{decimal}f})}}{{{E:.{decimal}f}}}\right)^2")

            st.markdown("**Step 4: Compute the required sample size**")
            st.latex(fr"n = {n_req:.{decimal}f}")

            st.markdown("**Step 5: Round up to the next whole number**")
            st.latex(fr"n = {n_round}")

            interpretation_box(
                f"At <b>{conf*100:.3f}% confidence</b>, you need at least "
                f"<b>{n_round}</b> samples to estimate μ with margin of error <b>{E:.3f}</b>."
            )

    # ==========================================================
    # 7) CI for Variance & SD (χ²)
    # ==========================================================
    elif choice == categories[6]:

        st.latex(r"""
            \left(
            \frac{(n-1)s^2}{\chi^2_{upper}},
            \frac{(n-1)s^2}{\chi^2_{lower}}
            \right)
        """)

        n = st.number_input("Sample size (n)", min_value=2)
        method = st.radio(
            "Provide input:",
            ["Enter SD (s)", "Enter Variance (s²)"],
            horizontal=True
        )

        if method == "Enter SD (s)":
            s = st.number_input("Sample SD (s)", min_value=0.0)
            s2 = s ** 2
        else:
            s2 = st.number_input("Sample Variance (s²)", min_value=0.0)
            s = np.sqrt(s2)

        conf = st.number_input("Confidence level", value=0.950, step=0.001, format="%.3f")

        if st.button("👨‍💻 Calculate"):

            df = int(n - 1)
            alpha = 1 - conf
            chi_lower = stats.chi2.ppf((1 - conf) / 2, df)
            chi_upper = stats.chi2.ppf(1 - (1 - conf) / 2, df)

            var_lower = df * s2 / chi_upper
            var_upper = df * s2 / chi_lower

            sd_lower = np.sqrt(var_lower)
            sd_upper = np.sqrt(var_upper)

            st.subheader("Step-by-Step Solution")

            st.markdown("**Step 1: Identify the variance confidence interval formula**")
            st.latex(r"\left(\frac{(n-1)s^2}{\chi^2_{upper}},\; \frac{(n-1)s^2}{\chi^2_{lower}}\right)")

            st.markdown("**Step 2: List the known values**")
            st.latex(fr"n = {n}, \quad df = n-1 = {df}, \quad s^2 = {s2:.{decimal}f}")

            st.markdown("**Step 3: Compute α and the chi-square critical values**")
            st.latex(fr"\alpha = 1 - {conf:.3f} = {alpha:.{decimal}f}")
            st.latex(fr"\alpha/2 = {alpha/2:.{decimal}f}")
            st.latex(fr"\chi^2_{{lower}} = {chi_lower:.{decimal}f}")
            st.latex(fr"\chi^2_{{upper}} = {chi_upper:.{decimal}f}")

            st.markdown("**Step 4: Compute the confidence interval for variance**")
            st.latex(fr"\text{{Variance CI}} = \left({var_lower:.{decimal}f},\; {var_upper:.{decimal}f}\right)")

            st.markdown("**Step 5: Compute the confidence interval for standard deviation**")
            st.latex(fr"\text{{SD CI}} = \left({sd_lower:.{decimal}f},\; {sd_upper:.{decimal}f}\right)")

            interpretation_box(
                f"We are <b>{conf*100:.3f}% confident</b> that the population variance "
                f"lies between <b>{var_lower:.{decimal}f}</b> and <b>{var_upper:.{decimal}f}</b>, "
                f"and that the population standard deviation lies between "
                f"<b>{sd_lower:.{decimal}f}</b> and <b>{sd_upper:.{decimal}f}</b>."
            )

    # ==========================================================
    # 8) CI for Variance & SD with Data
    # ==========================================================
    else:

        st.latex(r"""
            \text{CI using } \chi^2 \text{ and sample variance}
        """)

        data = load_uploaded_data()
        raw = st.text_area("Or enter comma-separated values:")

        if data is None and raw:
            try:
                data = np.array([float(x) for x in raw.split(",")])
            except:
                st.error("❌ Invalid input.")
                return

        conf = st.number_input("Confidence level", value=0.950, step=0.001, format="%.3f")

        if st.button("👨‍💻 Calculate"):

            if data is None or len(data) < 2:
                st.warning("⚠️ Need at least two numbers.")
                return

            n = len(data)
            df = n - 1
            s2 = np.var(data, ddof=1)
            s = np.sqrt(s2)
            alpha = 1 - conf

            chi_lower = stats.chi2.ppf((1 - conf) / 2, df)
            chi_upper = stats.chi2.ppf(1 - (1 - conf) / 2, df)

            var_lower = df * s2 / chi_upper
            var_upper = df * s2 / chi_lower
            sd_lower = np.sqrt(var_lower)
            sd_upper = np.sqrt(var_upper)

            st.subheader("Step-by-Step Solution")

            st.markdown("**Step 1: Compute the sample statistics**")
            st.latex(fr"n = {n}")
            st.latex(fr"df = n - 1 = {df}")
            st.latex(fr"s^2 = {s2:.{decimal}f}")
            st.latex(fr"s = {s:.{decimal}f}")

            st.markdown("**Step 2: Compute α and the chi-square critical values**")
            st.latex(fr"\alpha = 1 - {conf:.3f} = {alpha:.{decimal}f}")
            st.latex(fr"\alpha/2 = {alpha/2:.{decimal}f}")
            st.latex(fr"\chi^2_{{lower}} = {chi_lower:.{decimal}f}")
            st.latex(fr"\chi^2_{{upper}} = {chi_upper:.{decimal}f}")

            st.markdown("**Step 3: Compute the confidence interval for variance**")
            st.latex(fr"\text{{Variance CI}} = \left({var_lower:.{decimal}f},\; {var_upper:.{decimal}f}\right)")

            st.markdown("**Step 4: Compute the confidence interval for standard deviation**")
            st.latex(fr"\text{{SD CI}} = \left({sd_lower:.{decimal}f},\; {sd_upper:.{decimal}f}\right)")

            interpretation_box(
                f"We are <b>{conf*100:.3f}% confident</b> that the population "
                f"variance lies between <b>{var_lower:.{decimal}f}</b> and "
                f"<b>{var_upper:.{decimal}f}</b>, and that the population standard deviation "
                f"lies between <b>{sd_lower:.{decimal}f}</b> and <b>{sd_upper:.{decimal}f}</b>."
            )


# ==========================================================
# Run app directly
# ==========================================================
if __name__ == "__main__":
    run()
