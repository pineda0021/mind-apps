# ==========================================================
# confidence_intervals_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# MIND: Statistics Visualizer Suite
# Updated with Dark/Light Mode Safe Interpretation Boxes
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
        n = st.number_input("Sample size (n)", min_value=max(1, int(x)), step=1)
        conf = st.number_input("Confidence level (0–1)", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            
            p_hat = x / n
            z = stats.norm.ppf((1 + conf) / 2)
            se = np.sqrt(p_hat * (1 - p_hat) / n)
            moe = z * se
            lower, upper = p_hat - moe, p_hat + moe

            st.subheader("Step-by-Step Solution")
            st.markdown("**Step 1:** Compute sample proportion")
            st.latex(fr"\hat{{p}} = {p_hat:.{decimal}f}")

            st.markdown("**Step 2:** Find critical value")
            st.latex(fr"z_{{\alpha/2}} = {z:.{decimal}f}")

            st.markdown("**Step 3:** Standard error")
            st.latex(fr"SE = {se:.{decimal}f}")

            st.markdown("**Step 4:** Margin of error")
            st.latex(fr"E = {moe:.{decimal}f}")

            st.markdown("**Step 5:** Final CI")
            st.latex(fr"({lower:.{decimal}f},\; {upper:.{decimal}f})")

            interpretation_box(
                f"We are <b>{conf*100:.1f}% confident</b> that the true population "
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

        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        p_est = st.number_input("Estimated proportion (p̂)", value=0.5)
        E = st.number_input("Margin of error (E)", value=0.05)

        if st.button("👨‍💻 Calculate"):

            z = stats.norm.ppf((1 + conf) / 2)
            n_req = p_est * (1 - p_est) * (z / E) ** 2
            n_round = int(np.ceil(n_req))

            st.subheader("Step-by-Step Solution")
            st.markdown("**Step 1:** Critical value")
            st.latex(fr"z = {z:.{decimal}f}")

            st.markdown("**Step 2:** Compute required n")
            st.latex(fr"n = {n_req:.{decimal}f}")

            st.markdown("**Step 3:** Round up")
            st.latex(fr"n = {n_round}")

            interpretation_box(
                f"A minimum of <b>{n_round}</b> participants is needed to achieve "
                f"<b>{conf*100:.1f}% confidence</b> with margin of error <b>{E}</b>."
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
        conf = st.number_input("Confidence level", value=0.95)

        if st.button("👨‍💻 Calculate"):

            z = stats.norm.ppf((1 + conf) / 2)
            se = sigma / np.sqrt(n)
            moe = z * se
            lower, upper = mean - moe, mean + moe

            st.subheader("Step-by-Step")
            st.markdown("**Step 1:** Critical z-value")
            st.latex(fr"z = {z:.{decimal}f}")

            st.markdown("**Step 2:** Standard error")
            st.latex(fr"SE = {se:.{decimal}f}")

            st.markdown("**Step 3:** Margin of error")
            st.latex(fr"E = {moe:.{decimal}f}")

            st.markdown("**Step 4:** Final CI")
            st.latex(fr"({lower:.{decimal}f}, {upper:.{decimal}f})")

            interpretation_box(
                f"We are <b>{conf*100:.1f}% confident</b> that μ lies between "
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
        conf = st.number_input("Confidence level", value=0.95)

        if st.button("👨‍💻 Calculate"):

            df = int(n - 1)
            tcrit = stats.t.ppf((1 + conf) / 2, df)
            se = s / np.sqrt(n)
            moe = tcrit * se
            lower, upper = mean - moe, mean + moe

            st.subheader("Step-by-Step")
            st.markdown("**Step 1:** t critical value")
            st.latex(fr"t_{{df}} = {tcrit:.{decimal}f}")

            st.markdown("**Step 2:** Compute SE")
            st.latex(fr"SE = {se:.{decimal}f}")

            st.markdown("**Step 3:** MOE")
            st.latex(fr"E = {moe:.{decimal}f}")

            st.markdown("**Step 4:** CI")
            st.latex(fr"({lower:.{decimal}f}, {upper:.{decimal}f})")

            interpretation_box(
                f"We are <b>{conf*100:.1f}% confident</b> that μ lies between "
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

        conf = st.number_input("Confidence level", value=0.95)

        if st.button("👨‍💻 Calculate"):

            if data is None or len(data) < 2:
                st.warning("⚠️ Need at least 2 numbers.")
                return

            n = len(data)
            df = n - 1
            mean = np.mean(data)
            s = np.std(data, ddof=1)
            tcrit = stats.t.ppf((1 + conf) / 2, df)
            se = s / np.sqrt(n)
            moe = tcrit * se
            lower, upper = mean - moe, mean + moe

            st.subheader("Step-by-Step")
            st.markdown("**Step 1:** Summary stats")
            st.write(f"n={n}, x̄={mean:.{decimal}f}, s={s:.{decimal}f}")

            st.markdown("**Step 2:** Critical value")
            st.latex(fr"t = {tcrit:.{decimal}f}")

            st.markdown("**Step 3:** Compute CI")
            st.latex(fr"({lower:.{decimal}f}, {upper:.{decimal}f})")

            interpretation_box(
                f"We are <b>{conf*100:.1f}% confident</b> that μ lies between "
                f"<b>{lower:.{decimal}f}</b> and <b>{upper:.{decimal}f}</b>."
            )

    # ==========================================================
    # 6) Sample Size for Mean (σ known)
    # ==========================================================
    elif choice == categories[5]:

        st.latex(r"""
            n = \left(\frac{z_{\alpha/2}\sigma}{E}\right)^2
        """)

        conf = st.number_input("Confidence level", value=0.95)
        sigma = st.number_input("Population SD (σ)", min_value=0.0)
        E = st.number_input("Margin of error (E)", value=0.05)

        if st.button("👨‍💻 Calculate"):

            z = stats.norm.ppf((1 + conf) / 2)
            n_req = (z * sigma / E) ** 2
            n_round = int(np.ceil(n_req))

            st.subheader("Step-by-Step")
            st.latex(fr"z = {z:.{decimal}f}")
            st.latex(fr"n = {n_req:.{decimal}f}")
            st.latex(fr"n = {n_round}")

            interpretation_box(
                f"At <b>{conf*100:.1f}% confidence</b>, you need at least "
                f"<b>{n_round}</b> samples to estimate μ with margin of error <b>{E}</b>."
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

        conf = st.number_input("Confidence level", value=0.95)

        if st.button("👨‍💻 Calculate"):

            df = int(n - 1)
            chi_lower = stats.chi2.ppf((1 - conf) / 2, df)
            chi_upper = stats.chi2.ppf(1 - (1 - conf) / 2, df)

            var_lower = df * s2 / chi_upper
            var_upper = df * s2 / chi_lower

            sd_lower = np.sqrt(var_lower)
            sd_upper = np.sqrt(var_upper)

            st.subheader("Step-by-Step")
            st.latex(fr"χ^2_{{lower}} = {chi_lower:.{decimal}f}")
            st.latex(fr"χ^2_{{upper}} = {chi_upper:.{decimal}f}")
            st.latex(fr"\text{{Var CI}} = ({var_lower:.{decimal}f}, {var_upper:.{decimal}f})")
            st.latex(fr"\text{{SD CI}} = ({sd_lower:.{decimal}f}, {sd_upper:.{decimal}f})")

            interpretation_box(
                f"Variance is between <b>{var_lower:.{decimal}f}</b> and "
                f"<b>{var_upper:.{decimal}f}</b>. "
                f"Standard deviation is between <b>{sd_lower:.{decimal}f}</b> and "
                f"<b>{sd_upper:.{decimal}f}</b>."
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

        conf = st.number_input("Confidence level", value=0.95)

        if st.button("👨‍💻 Calculate"):

            if data is None or len(data) < 2:
                st.warning("⚠️ Need at least two numbers.")
                return

            n = len(data)
            df = n - 1
            s2 = np.var(data, ddof=1)
            s = np.sqrt(s2)

            chi_lower = stats.chi2.ppf((1 - conf) / 2, df)
            chi_upper = stats.chi2.ppf(1 - (1 - conf) / 2, df)

            var_lower = df * s2 / chi_upper
            var_upper = df * s2 / chi_lower
            sd_lower = np.sqrt(var_lower)
            sd_upper = np.sqrt(var_upper)

            st.subheader("Step-by-Step")
            st.write(f"n={n}, s²={s2:.{decimal}f}, s={s:.{decimal}f}")

            st.latex(fr"χ^2_{{lower}} = {chi_lower:.{decimal}f}")
            st.latex(fr"χ^2_{{upper}} = {chi_upper:.{decimal}f}")
            st.latex(fr"\text{{Variance CI}} = ({var_lower:.{decimal}f}, {var_upper:.{decimal}f})")
            st.latex(fr"\text{{SD CI}} = ({sd_lower:.{decimal}f}, {sd_upper:.{decimal}f})")

            interpretation_box(
                f"We are <b>{conf*100:.1f}% confident</b> that the population "
                f"variance lies between <b>{var_lower:.{decimal}f}</b> and "
                f"<b>{var_upper:.{decimal}f}</b>, and deviation between "
                f"<b>{sd_lower:.{decimal}f}</b> and <b>{sd_upper:.{decimal}f}</b>."
            )


# ==========================================================
# Run app directly
# ==========================================================
if __name__ == "__main__":
    run()

