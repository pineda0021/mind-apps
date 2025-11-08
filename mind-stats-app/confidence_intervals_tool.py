# ==========================================================
# confidence_intervals_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# MIND: Statistics Visualizer Suite
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
        x = st.number_input("Number of successes (x)", min_value=0, step=1)
        n = st.number_input("Sample size (n)", min_value=max(1, int(x)), step=1)
        conf = st.number_input("Confidence level (0–1)", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            p_hat = x / n
            z = stats.norm.ppf((1 + conf) / 2)
            se = np.sqrt(p_hat * (1 - p_hat) / n)
            moe = z * se
            lower, upper = p_hat - moe, p_hat + moe

            st.latex(r"\hat{p} \pm z_{\alpha/2}\sqrt{\dfrac{\hat{p}(1-\hat{p})}{n}}")

            st.subheader("Step-by-Step Solution")
            st.markdown("**Step 1:** Compute the sample proportion")
            st.latex(fr"\hat{{p}} = \dfrac{{x}}{{n}} = {x}/{n} = {p_hat:.{decimal}f}")

            st.markdown("**Step 2:** Find the critical z-value")
            st.latex(fr"z_{{\alpha/2}} = {z:.{decimal}f} \quad \text{{for confidence}} = {conf:.3f}")

            st.markdown("**Step 3:** Compute the standard error")
            st.latex(fr"SE = \sqrt{{\hat{{p}}(1-\hat{{p}})/n}} = {se:.{decimal}f}")

            st.markdown("**Step 4:** Margin of error")
            st.latex(fr"E = z \times SE = {moe:.{decimal}f}")

            st.markdown("**Step 5:** Confidence Interval")
            st.latex(fr"(\hat{{p}} - E,\; \hat{{p}} + E) = ({lower:.{decimal}f},\; {upper:.{decimal}f})")

            st.markdown(
                f"""
<div style="background-color:#e6f3ff; padding:10px; border-radius:10px;">
<b>Interpretation:</b><br>
We are <b>{conf*100:.1f}% confident</b> that the true population proportion lies between 
<b>{lower:.{decimal}f}</b> and <b>{upper:.{decimal}f}</b>.
</div>
""", unsafe_allow_html=True)

    # ==========================================================
    # 2) Sample Size for Proportion (p, z, E)
    # ==========================================================
    elif choice == categories[1]:
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        p_est = st.number_input("Estimated proportion (p̂)", value=0.5, min_value=0.0, max_value=1.0)
        E = st.number_input("Margin of error (E)", value=0.05, min_value=0.000001)

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + conf) / 2)
            n_req = p_est * (1 - p_est) * (z / E) ** 2
            n_ceiled = int(np.ceil(n_req))

            st.latex(r"n = \hat{p}(1-\hat{p})\!\left(\dfrac{Z_{\alpha/2}}{E}\right)^{2}")

            st.subheader("Step-by-Step Solution")
            st.markdown("**Step 1:** Compute z-value")
            st.latex(fr"z_{{\alpha/2}} = {z:.{decimal}f}")

            st.markdown("**Step 2:** Substitute values into the formula")
            st.latex(fr"n = {p_est:.{decimal}f}(1-{p_est:.{decimal}f})({z:.{decimal}f}/{E})^2 = {n_req:.{decimal}f}")

            st.markdown("**Step 3:** Round up to the next whole number")
            st.latex(fr"n = {n_ceiled}")

            st.markdown(
                f"""
<div style="background-color:#e6f3ff; padding:10px; border-radius:10px;">
<b>Interpretation:</b><br>
A minimum of <b>{n_ceiled}</b> observations is needed to achieve {conf*100:.1f}% confidence 
with a margin of error of <b>{E}</b>.
</div>
""", unsafe_allow_html=True)

    # ==========================================================
    # 3) Confidence Interval for Mean (σ known, z)
    # ==========================================================
    elif choice == categories[2]:
        mean = st.number_input("Sample mean (x̄)")
        sigma = st.number_input("Population SD (σ)", min_value=0.0)
        n = st.number_input("Sample size (n)", min_value=1, step=1)
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + conf) / 2)
            se = sigma / np.sqrt(n)
            moe = z * se
            lower, upper = mean - moe, mean + moe

            st.latex(r"\bar{X} \pm z_{\alpha/2}\!\left(\dfrac{\sigma}{\sqrt{n}}\right)")

            st.subheader("Step-by-Step Solution")
            st.markdown("**Step 1:** Inputs")
            st.write(f"x̄ = {mean}, σ = {sigma}, n = {int(n)}")

            st.markdown("**Step 2:** Critical value")
            st.latex(fr"z_{{\alpha/2}} = {z:.{decimal}f}")

            st.markdown("**Step 3:** Compute standard error")
            st.latex(fr"SE = \dfrac{{\sigma}}{{\sqrt{{n}}}} = {se:.{decimal}f}")

            st.markdown("**Step 4:** Margin of error")
            st.latex(fr"E = z \times SE = {moe:.{decimal}f}")

            st.markdown("**Step 5:** Confidence Interval")
            st.latex(fr"(\bar{{X}} - E,\; \bar{{X}} + E) = ({lower:.{decimal}f},\; {upper:.{decimal}f})")

            st.markdown(
                f"""
<div style="background-color:#e6f3ff; padding:10px; border-radius:10px;">
<b>Interpretation:</b><br>
We are <b>{conf*100:.1f}% confident</b> that the population mean μ lies between 
<b>{lower:.{decimal}f}</b> and <b>{upper:.{decimal}f}</b>.
</div>
""", unsafe_allow_html=True)

    # ==========================================================
    # 4) Confidence Interval for Mean (s given, t)
    # ==========================================================
    elif choice == categories[3]:
        mean = st.number_input("Sample mean (x̄)")
        s = st.number_input("Sample SD (s)", min_value=0.0)
        n = st.number_input("Sample size (n)", min_value=2, step=1)
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            df = int(n - 1)
            t_crit = stats.t.ppf((1 + conf) / 2, df)
            se = s / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe

            st.latex(r"\bar{X} \pm t_{\alpha/2,\,n-1}\!\left(\dfrac{s}{\sqrt{n}}\right)")
            st.subheader("Step-by-Step Solution")
            st.markdown("**Step 1:** Inputs")
            st.write(f"x̄ = {mean}, s = {s}, n = {n}, df = {df}")

            st.markdown("**Step 2:** Find t critical value")
            st.latex(fr"t_{{\alpha/2,df}} = {t_crit:.{decimal}f}")

            st.markdown("**Step 3:** Compute standard error")
            st.latex(fr"SE = \dfrac{{s}}{{\sqrt{{n}}}} = {se:.{decimal}f}")

            st.markdown("**Step 4:** Margin of error")
            st.latex(fr"E = t \times SE = {moe:.{decimal}f}")

            st.markdown("**Step 5:** Confidence Interval")
            st.latex(fr"(\bar{{X}} - E,\; \bar{{X}} + E) = ({lower:.{decimal}f},\; {upper:.{decimal}f})")

            st.markdown(
                f"""
<div style="background-color:#e6f3ff; padding:10px; border-radius:10px;">
<b>Interpretation:</b><br>
We are <b>{conf*100:.1f}% confident</b> that μ lies between 
<b>{lower:.{decimal}f}</b> and <b>{upper:.{decimal}f}</b>.
</div>
""", unsafe_allow_html=True)

    # ==========================================================
    # 5) Confidence Interval for Mean (with data, t)
    # ==========================================================
    elif choice == categories[4]:
        data = load_uploaded_data()
        raw_input = st.text_area("Or enter comma-separated values:")
        if data is None and raw_input:
            try:
                data = np.array([float(x.strip()) for x in raw_input.split(",") if x.strip()])
            except:
                st.error("❌ Invalid input.")
                return

        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            if data is None or len(data) < 2:
                st.warning("⚠️ Provide at least two data points.")
                return

            n, mean, s = len(data), np.mean(data), np.std(data, ddof=1)
            df = n - 1
            t_crit = stats.t.ppf((1 + conf) / 2, df)
            se = s / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe

            st.latex(r"\bar{X} \pm t_{\alpha/2,\,n-1}\!\left(\dfrac{s}{\sqrt{n}}\right)")
            st.subheader("Step-by-Step Solution")
            st.markdown("**Step 1:** From data")
            st.write(f"n={n}, x̄={mean:.{decimal}f}, s={s:.{decimal}f}")

            st.markdown("**Step 2:** Compute critical value")
            st.latex(fr"t_{{\alpha/2,df}} = {t_crit:.{decimal}f}")

            st.markdown("**Step 3:** Compute SE and MOE")
            st.latex(fr"SE = \dfrac{{s}}{{\sqrt{{n}}}} = {se:.{decimal}f}, \quad E = t \times SE = {moe:.{decimal}f}")

            st.markdown("**Step 4:** Confidence Interval")
            st.latex(fr"({lower:.{decimal}f}, {upper:.{decimal}f})")

            st.markdown(
                f"""
<div style="background-color:#e6f3ff; padding:10px; border-radius:10px;">
<b>Interpretation:</b><br>
We are <b>{conf*100:.1f}% confident</b> that μ lies between 
<b>{lower:.{decimal}f}</b> and <b>{upper:.{decimal}f}</b>.
</div>
""", unsafe_allow_html=True)

    # ==========================================================
    # 6) Sample Size for Mean (σ known, z, E)
    # ==========================================================
    elif choice == categories[5]:
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        sigma = st.number_input("Population SD (σ)", min_value=0.0)
        E = st.number_input("Margin of error (E)", value=0.05, min_value=0.000001)

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + conf) / 2)
            n_req = (z * sigma / E) ** 2
            n_ceiled = int(np.ceil(n_req))

            st.latex(r"n = \left(\dfrac{z_{\alpha/2}\sigma}{E}\right)^2")

            st.subheader("Step-by-Step Solution")
            st.markdown("**Step 1:** Critical z-value")
            st.latex(fr"z_{{\alpha/2}} = {z:.{decimal}f}")

            st.markdown("**Step 2:** Substitute into formula")
            st.latex(fr"n = ({z:.{decimal}f} \times {sigma:.{decimal}f} / {E})^2 = {n_req:.{decimal}f}")

            st.markdown("**Step 3:** Round up")
            st.latex(fr"n = {n_ceiled}")

            st.markdown(
                f"""
<div style="background-color:#e6f3ff; padding:10px; border-radius:10px;">
<b>Interpretation:</b><br>
At {conf*100:.1f}% confidence, at least <b>{n_ceiled}</b> samples are required 
to estimate μ with a margin of error of <b>{E}</b>.
</div>
""", unsafe_allow_html=True)

    # ==========================================================
    # 7) Confidence Interval for Variance & SD (χ²)
    # ==========================================================
    elif choice == categories[6]:
        n = st.number_input("Sample size (n)", min_value=2, step=1)
        input_type = st.radio(
            "Provide summary input:",
            ["Enter sample standard deviation (s)", "Enter sample variance (s²)"],
            horizontal=True
        )

        if input_type == "Enter sample standard deviation (s)":
            s = st.number_input("Sample SD (s)", min_value=0.0)
            s2 = s ** 2
        else:
            s2 = st.number_input("Sample variance (s²)", min_value=0.0)
            s = np.sqrt(s2)

        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            df = int(n - 1)
            chi2_lower = stats.chi2.ppf((1 - conf)/2, df)
            chi2_upper = stats.chi2.ppf(1 - (1 - conf)/2, df)
            var_lower, var_upper = df * s2 / chi2_upper, df * s2 / chi2_lower
            sd_lower, sd_upper = np.sqrt(var_lower), np.sqrt(var_upper)

            st.latex(r"\text{Var CI: } \left(\dfrac{(n-1)s^2}{\chi^2_{(1-\alpha/2)}}, \dfrac{(n-1)s^2}{\chi^2_{(\alpha/2)}}\right)")
            st.subheader("Step-by-Step Solution")
            st.markdown("**Step 1:** Inputs")
            st.write(f"n={int(n)}, df={df}, s²={s2:.{decimal}f}, s={s:.{decimal}f}")

            st.markdown("**Step 2:** χ² critical values")
            st.latex(fr"χ²_{{lower}} = {chi2_lower:.{decimal}f}, \quad χ²_{{upper}} = {chi2_upper:.{decimal}f}")

            st.markdown("**Step 3:** Variance Interval")
            st.latex(fr"\text{{Variance CI}} = \left(\dfrac{{(n-1)s^2}}{{χ²_{{upper}}}}, \dfrac{{(n-1)s^2}}{{χ²_{{lower}}}}\right) = ({var_lower:.{decimal}f}, {var_upper:.{decimal}f})")

            st.markdown("**Step 4:** Standard Deviation Interval")
            st.latex(fr"\text{{SD CI}} = (\sqrt{{{var_lower:.{decimal}f}}}, \sqrt{{{var_upper:.{decimal}f}}}) = ({sd_lower:.{decimal}f}, {sd_upper:.{decimal}f})")

            st.markdown(
                f"""
<div style="background-color:#e6f3ff; padding:10px; border-radius:10px;">
<b>Interpretation:</b><br>
We are <b>{conf*100:.1f}% confident</b> that the population variance lies between 
<b>{var_lower:.{decimal}f}</b> and <b>{var_upper:.{decimal}f}</b>, and the population standard deviation lies between 
<b>{sd_lower:.{decimal}f}</b> and <b>{sd_upper:.{decimal}f}</b>.
</div>
""", unsafe_allow_html=True)

    # ==========================================================
    # 8) Confidence Interval for Variance & SD (with data, χ²)
    # ==========================================================
    else:
        st.subheader("📊 Confidence Interval for Variance & SD (with data, χ²)")
        data = load_uploaded_data()
        raw_input = st.text_area("Or enter comma-separated values:")
        if data is None and raw_input:
            try:
                data = np.array([float(x.strip()) for x in raw_input.split(",") if x.strip()])
            except:
                st.error("❌ Invalid input. Use numeric comma-separated values only.")
                return

        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            if data is None or len(data) < 2:
                st.warning("⚠️ Provide at least two data points.")
                return

            n = len(data)
            df = n - 1
            s2 = np.var(data, ddof=1)
            s = np.sqrt(s2)
            chi2_lower = stats.chi2.ppf((1 - conf)/2, df)
            chi2_upper = stats.chi2.ppf(1 - (1 - conf)/2, df)
            var_lower, var_upper = df * s2 / chi2_upper, df * s2 / chi2_lower
            sd_lower, sd_upper = np.sqrt(var_lower), np.sqrt(var_upper)

            st.latex(r"\text{CI for Variance and SD using χ² distribution}")

            st.subheader("Step-by-Step Solution")
            st.markdown("**Step 1:** Compute summary statistics from data")
            st.write(f"n={n}, df={df}, s²={s2:.{decimal}f}, s={s:.{decimal}f}")

            st.markdown("**Step 2:** χ² critical values")
            st.latex(fr"χ²_{{lower}} = {chi2_lower:.{decimal}f}, \quad χ²_{{upper}} = {chi2_upper:.{decimal}f}")

            st.markdown("**Step 3:** Variance Interval")
            st.latex(fr"\text{{Variance CI}} = ({var_lower:.{decimal}f}, {var_upper:.{decimal}f})")

            st.markdown("**Step 4:** Standard Deviation Interval")
            st.latex(fr"\text{{SD CI}} = ({sd_lower:.{decimal}f}, {sd_upper:.{decimal}f})")

            st.markdown(
                f"""
<div style="background-color:#e6f3ff; padding:10px; border-radius:10px;">
<b>Interpretation:</b><br>
We are <b>{conf*100:.1f}% confident</b> that the population variance lies between 
<b>{var_lower:.{decimal}f}</b> and <b>{var_upper:.{decimal}f}</b>, and the population standard deviation lies between 
<b>{sd_lower:.{decimal}f}</b> and <b>{sd_upper:.{decimal}f}</b>.
</div>
""", unsafe_allow_html=True)


# ==========================================================
# Run app directly
# ==========================================================
if __name__ == "__main__":
    run()


