# ==========================================================
# confidence_intervals_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
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
        "📂 Upload CSV or Excel file with a single column of numeric data",
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
    st.header("🔮 MIND: Confidence Interval Calculator")
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
    # 1. CI for Proportion (p, z)
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

            st.latex(r"\hat{p} \;\pm\; z_{\alpha/2}\sqrt{\dfrac{\hat{p}(1-\hat{p})}{n}}")
            st.text(f"""
=====================
Confidence Interval for Proportion (p, z)
=====================
1) Inputs: x={int(x)}, n={int(n)}, p̂={p_hat:.{decimal}f}, confidence={conf:.3f}
2) z_(α/2)={z:.{decimal}f}
3) SE=sqrt[p̂(1−p̂)/n]={se:.{decimal}f}
4) E=z*SE={moe:.{decimal}f}
5) CI=({lower:.{decimal}f}, {upper:.{decimal}f})

Interpretation:
  We are {conf*100:.1f}% confident the population proportion lies between {lower:.{decimal}f} and {upper:.{decimal}f}.
""")

    # ==========================================================
    # 2. Sample Size for Proportion (p, z, E)
    # ==========================================================
    elif choice == categories[1]:
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        p_est = st.number_input("Estimated proportion (p̂)", value=0.5, min_value=0.0, max_value=1.0, step=0.001)
        E = st.number_input("Margin of error (E)", value=0.05, min_value=0.000001, step=0.001)

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + conf) / 2)
            n_req = p_est * (1 - p_est) * (z / E)**2
            n_ceiled = int(np.ceil(n_req))
            st.latex(r"n \;=\; \hat{p}(1-\hat{p})\!\left(\dfrac{Z_{\alpha/2}}{E}\right)^{2}")
            st.text(f"""
=====================
Sample Size for Proportion (p, z, E)
=====================
1) Inputs: Z_(α/2)={z:.{decimal}f}, p̂={p_est:.{decimal}f}, E={E}
2) Compute: n=p̂(1-p̂)(Z/E)^2={n_req:.{decimal}f}
3) Round up: n={n_ceiled}

Interpretation:
  A sample of at least {n_ceiled} is required at {conf*100:.1f}% confidence.
""")

    # ==========================================================
    # 3. CI for Mean (σ known, z)
    # ==========================================================
    elif choice == categories[2]:
        mean = st.number_input("Sample mean (x̄)")
        sigma = st.number_input("Population SD (σ)", min_value=0.0, format="%.4f")
        n = st.number_input("Sample size (n)", min_value=1, step=1)
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + conf) / 2)
            se = sigma / np.sqrt(n)
            moe = z * se
            lower, upper = mean - moe, mean + moe

            st.latex(r"\bar{X} \;\pm\; z_{\alpha/2}\!\left(\dfrac{\sigma}{\sqrt{n}}\right)")
            st.text(f"""
=====================
Confidence Interval for Mean (σ known, z)
=====================
1) Inputs: x̄={mean:.{decimal}f}, σ={sigma:.{decimal}f}, n={int(n)}
2) z_(α/2)={z:.{decimal}f}
3) SE=σ/√n={se:.{decimal}f}
4) E=z*SE={moe:.{decimal}f}
5) CI=({lower:.{decimal}f}, {upper:.{decimal}f})

Interpretation:
  We are {conf*100:.1f}% confident μ lies between {lower:.{decimal}f} and {upper:.{decimal}f}.
""")

    # ==========================================================
    # 4. CI for Mean (s given, t)
    # ==========================================================
    elif choice == categories[3]:
        mean = st.number_input("Sample mean (x̄)")
        s = st.number_input("Sample SD (s)", min_value=0.0, format="%.4f")
        n = st.number_input("Sample size (n)", min_value=2, step=1)
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            df = int(n - 1)
            t_crit = stats.t.ppf((1 + conf) / 2, df)
            se = s / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe
            st.latex(r"\bar{X} \;\pm\; t_{\alpha/2,\,n-1}\!\left(\dfrac{s}{\sqrt{n}}\right)")
            st.text(f"""
=====================
Confidence Interval for Mean (s given, t)
=====================
1) Inputs: x̄={mean:.{decimal}f}, s={s:.{decimal}f}, n={int(n)}, df={df}
2) t_(α/2,df)={t_crit:.{decimal}f}
3) SE=s/√n={se:.{decimal}f}
4) E=t*SE={moe:.{decimal}f}
5) CI=({lower:.{decimal}f}, {upper:.{decimal}f})

Interpretation:
  We are {conf*100:.1f}% confident μ lies between {lower:.{decimal}f} and {upper:.{decimal}f}.
""")

    # ==========================================================
    # 5. CI for Mean (with data, t)
    # ==========================================================
    elif choice == categories[4]:
        st.subheader("📊 Confidence Interval for Mean (with data, t)")
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
            n, mean, s = len(data), np.mean(data), np.std(data, ddof=1)
            df = n - 1
            t_crit = stats.t.ppf((1 + conf) / 2, df)
            se = s / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe
            st.latex(r"\bar{X} \;\pm\; t_{\alpha/2,\,n-1}\!\left(\dfrac{s}{\sqrt{n}}\right)")
            st.text(f"""
=====================
Confidence Interval for Mean (with data, t)
=====================
1) n={n}, x̄={mean:.{decimal}f}, s={s:.{decimal}f}, df={df}
2) t_(α/2,df)={t_crit:.{decimal}f}
3) SE={se:.{decimal}f}
4) E={moe:.{decimal}f}
5) CI=({lower:.{decimal}f}, {upper:.{decimal}f})

Interpretation:
  We are {conf*100:.1f}% confident μ lies between {lower:.{decimal}f} and {upper:.{decimal}f}.
=====================
""")

    # ==========================================================
    # 6. Sample Size for Mean (σ known, z, E)
    # ==========================================================
    elif choice == categories[5]:
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        sigma = st.number_input("Population SD (σ)", min_value=0.0, format="%.4f")
        E = st.number_input("Margin of error (E)", value=0.05, min_value=0.000001, step=0.001, format="%.6f")
        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + conf)/2)
            n_req = (z * sigma / E)**2
            n_ceiled = int(np.ceil(n_req))
            st.latex(r"n \;=\; \left(\dfrac{z_{\alpha/2}\,\sigma}{E}\right)^{2}")
            st.text(f"""
=====================
Sample Size for Mean (σ known, z, E)
=====================
1) Inputs: z_(α/2)={z:.{decimal}f}, σ={sigma}, E={E}
2) Compute: n=(zσ/E)^2={n_req:.{decimal}f}
3) Round up: n={n_ceiled}

Interpretation:
  Need at least {n_ceiled} observations for {conf*100:.1f}% confidence.
""")

    # ==========================================================
    # 7–8. CI for Variance & SD (χ²)
    # ==========================================================
    else:
        with_data = "with data" in choice
        if with_data:
            data = load_uploaded_data()
            if data is None:
                st.warning("⚠️ Upload numeric data.")
                return
            n = len(data)
            s2 = np.var(data, ddof=1)
        else:
            n = st.number_input("Sample size (n)", min_value=2, step=1)
            s2 = st.number_input("Sample variance (s²)", min_value=0.0, format="%.6f")
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        df = n - 1
        chi2_lower = stats.chi2.ppf((1 - conf)/2, df)
        chi2_upper = stats.chi2.ppf(1 - (1 - conf)/2, df)
        st.latex(r"\text{Var CI: } \left(\dfrac{(n-1)s^2}{\chi^2_{(1-\alpha/2),\,df}},\; \dfrac{(n-1)s^2}{\chi^2_{(\alpha/2),\,df}}\right)\quad \text{SD CI: } \left(\sqrt{\dfrac{(n-1)s^2}{\chi^2_{(1-\alpha/2),\,df}}},\; \sqrt{\dfrac{(n-1)s^2}{\chi^2_{(\alpha/2),\,df}}}\right)")
        if st.button("👨‍💻 Calculate"):
            numer = df * s2
            var_lower, var_upper = numer / chi2_upper, numer / chi2_lower
            sd_lower, sd_upper = np.sqrt(var_lower), np.sqrt(var_upper)
            st.text(f"""
=====================
Confidence Interval for Variance & SD (χ²)
=====================
1) Inputs: n={int(n)}, df={df}, s²={s2:.{decimal}f}
2) χ² upper={chi2_upper:.{decimal}f}, χ² lower={chi2_lower:.{decimal}f}
3) Var CI=({var_lower:.{decimal}f}, {var_upper:.{decimal}f})
4) SD CI=({sd_lower:.{decimal}f}, {sd_upper:.{decimal}f})

Interpretation:
  We are {conf*100:.1f}% confident that the population variance lies between {var_lower:.{decimal}f} and {var_upper:.{decimal}f},
  and the population SD lies between {sd_lower:.{decimal}f} and {sd_upper:.{decimal}f}.
""")

# ==========================================================
# Run app directly
# ==========================================================
if __name__ == "__main__":
    run()
