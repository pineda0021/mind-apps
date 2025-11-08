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
    """Round numeric values consistently"""
    return round(float(value), decimals)


def load_uploaded_data():
    """Load numeric data from CSV or Excel file"""
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
        "Confidence Interval for Proportion",
        "Sample Size for Proportion",
        "Confidence Interval for Mean (Known SD)",
        "Confidence Interval for Mean (Given Sample SD)",
        "Confidence Interval for Mean (With Data)",
        "Sample Size for Mean",
        "Confidence Interval for Variance (Without Data)",
        "Confidence Interval for Variance (With Data)",
        "Confidence Interval for Standard Deviation (Without Data)",
        "Confidence Interval for Standard Deviation (With Data)"
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
    # 1. Confidence Interval for Proportion (reordered inputs)
    # ==========================================================
    if choice == categories[0]:
        x = st.number_input("Number of successes (x)", min_value=0, step=1)
        n = st.number_input("Sample size (n)", min_value=max(1, int(x)), step=1)
        if x > n:
            st.warning("⚠️ Adjusted: successes x cannot exceed sample size n. Setting x = n.")
            x = n
        conf = st.number_input("Confidence level (0–1)", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            if n <= 0:
                st.error("❌ Sample size n must be > 0.")
                st.stop()
            if x < 0 or x > n:
                st.error("❌ x must be between 0 and n.")
                st.stop()

            p_hat = x / n
            z = stats.norm.ppf((1 + conf) / 2)
            se = np.sqrt(p_hat * (1 - p_hat) / n)
            moe = z * se
            lower, upper = p_hat - moe, p_hat + moe

            st.latex(r"\hat{p} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}")
            st.text(f"""
=====================
Confidence Interval for Proportion
=====================
1) Inputs:
   x = {int(x)}, n = {int(n)}, p̂ = x/n = {p_hat:.{decimal}f}, confidence = {conf:.3f}
2) Critical value:
   z_(α/2) = {z:.{decimal}f}
3) Standard error:
   SE = sqrt( p̂(1-p̂) / n ) = sqrt( {p_hat:.{decimal}f}(1-{p_hat:.{decimal}f}) / {int(n)} ) = {se:.{decimal}f}
4) Margin of error:
   E = z * SE = {z:.{decimal}f} * {se:.{decimal}f} = {moe:.{decimal}f}
5) Interval:
   p̂ ± E = {p_hat:.{decimal}f} ± {moe:.{decimal}f} → ({lower:.{decimal}f}, {upper:.{decimal}f})

Interpretation:
  We are {conf*100:.1f}% confident that the true population proportion lies between {lower:.{decimal}f} and {upper:.{decimal}f}.
""")

    # ==========================================================
    # 2. Sample Size for Proportion
    # ==========================================================
    elif choice == categories[1]:
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        p_est = st.number_input("Estimated proportion (p̂)", value=0.5, min_value=0.0, max_value=1.0, step=0.001)
        E = st.number_input("Margin of error (E)", value=0.05, min_value=0.000001, step=0.001)

        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1 + conf) / 2)
            n_req = (p_est * (1 - p_est) *z**2) / (E**2)
            n_ceiled = int(np.ceil(n_req))

            st.latex(r"\hat{p} \;\pm\; z_{\alpha/2} \sqrt{\dfrac{\hat{p}(1-\hat{p})}{n}}")
            st.text(f"""
=====================
Sample Size for Proportion
=====================
1) Inputs:
   confidence = {conf:.3f}, z_(α/2) = {z:.{decimal}f}, p̂ = {p_est:.{decimal}f}, E = {E}
2) Compute:
   n = ( z^2 * p̂(1-p̂) ) / E^2
     = ( {z:.{decimal}f}^2 * {p_est:.{decimal}f}(1-{p_est:.{decimal}f}) ) / {E}^2
     = {n_req:.{decimal}f}
3) Round up:
   n (required) = {n_ceiled}

Interpretation:
  A sample of at least {n_ceiled} is required to estimate the proportion with margin of error {E} at {conf*100:.1f}% confidence.
""")

    # ==========================================================
    # 3. Confidence Interval for Mean (Known SD)
    # ==========================================================
    elif choice == categories[2]:
        mean = st.number_input("Sample mean (x̄)")
        sigma = st.number_input("Population SD (σ)", min_value=0.0, format="%.4f")
        n = st.number_input("Sample size (n)", min_value=1, step=1)
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            if n <= 0:
                st.error("❌ Sample size n must be > 0.")
                st.stop()
            if sigma < 0:
                st.error("❌ σ must be ≥ 0.")
                st.stop()

            z = stats.norm.ppf((1 + conf) / 2)
            se = sigma / np.sqrt(n)
            moe = z * se
            lower, upper = mean - moe, mean + moe

            st.latex(r"\bar{X} \pm z_{\alpha/2}\left(\frac{\sigma}{\sqrt{n}}\right)")
            st.text(f"""
=====================
Confidence Interval for Mean (Known SD)
=====================
1) Inputs:
   x̄ = {mean:.{decimal}f}, σ = {sigma:.{decimal}f}, n = {int(n)}, confidence = {conf:.3f}
2) Critical value:
   z_(α/2) = {z:.{decimal}f}
3) Standard error:
   SE = σ/√n = {sigma:.{decimal}f}/√{int(n)} = {se:.{decimal}f}
4) Margin of error:
   E = z * SE = {z:.{decimal}f} * {se:.{decimal}f} = {moe:.{decimal}f}
5) Interval:
   x̄ ± E = {mean:.{decimal}f} ± {moe:.{decimal}f} → ({lower:.{decimal}f}, {upper:.{decimal}f})

Interpretation:
  We are {conf*100:.1f}% confident that the true population mean lies between {lower:.{decimal}f} and {upper:.{decimal}f}.
""")

    # ==========================================================
    # 4. Confidence Interval for Mean (Given Sample SD)
    # ==========================================================
    elif choice == categories[3]:
        mean = st.number_input("Sample mean (x̄)")
        s = st.number_input("Sample SD (s)", min_value=0.0, format="%.4f")
        n = st.number_input("Sample size (n)", min_value=2, step=1)
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            if n < 2:
                st.error("❌ n must be at least 2 for a t-interval.")
                st.stop()

            df = int(n - 1)
            t_crit = stats.t.ppf((1 + conf) / 2, df)
            se = s / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe

            st.latex(r"\bar{X} \pm t_{\alpha/2,\,n-1}\left(\frac{s}{\sqrt{n}}\right)")
            st.text(f"""
=====================
Confidence Interval for Mean (Given Sample SD)
=====================
1) Inputs:
   x̄ = {mean:.{decimal}f}, s = {s:.{decimal}f}, n = {int(n)}, df = {df}, confidence = {conf:.3f}
2) Critical value:
   t_(α/2, df) = {t_crit:.{decimal}f}
3) Standard error:
   SE = s/√n = {s:.{decimal}f}/√{int(n)} = {se:.{decimal}f}
4) Margin of error:
   E = t * SE = {t_crit:.{decimal}f} * {se:.{decimal}f} = {moe:.{decimal}f}
5) Interval:
   x̄ ± E = {mean:.{decimal}f} ± {moe:.{decimal}f} → ({lower:.{decimal}f}, {upper:.{decimal}f})

Interpretation:
  We are {conf*100:.1f}% confident that the true population mean lies between {lower:.{decimal}f} and {upper:.{decimal}f}.
""")

    # ==========================================================
    # 5. Confidence Interval for Mean (With Raw Data)
    # ==========================================================
    elif choice == categories[4]:
        st.subheader("📊 Confidence Interval for Mean (Using Raw Data)")
        data = load_uploaded_data()
        raw_input = st.text_area("Or enter comma-separated values:")

        if data is None and raw_input:
            try:
                data = np.array([float(x.strip()) for x in raw_input.split(",") if x.strip() != ""])
            except:
                st.error("❌ Invalid input. Please use only numeric comma-separated values.")
                data = None

        conf = st.number_input("Confidence level (e.g., 0.95)", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate"):
            if data is None or len(data) < 2:
                st.warning("⚠️ Please provide at least two data points.")
                st.stop()

            n = len(data)
            mean = float(np.mean(data))
            s = float(np.std(data, ddof=1))
            df = int(n - 1)
            t_crit = float(stats.t.ppf((1 + conf) / 2, df))
            se = s / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe

            st.latex(r"\bar{X} \pm t_{\alpha/2,\,n-1}\left(\frac{s}{\sqrt{n}}\right)")
            st.text(f"""
=====================
Confidence Interval for Mean (With Data)
=====================
1) Inputs (from data):
   n = {int(n)}, df = {df}, x̄ = {mean:.{decimal}f}, s = {s:.{decimal}f}, confidence = {conf:.3f}
2) Critical value:
   t_(α/2, df) = {t_crit:.{decimal}f}
3) Standard error:
   SE = s/√n = {s:.{decimal}f}/√{int(n)} = {se:.{decimal}f}
4) Margin of error:
   E = t * SE = {t_crit:.{decimal}f} * {se:.{decimal}f} = {moe:.{decimal}f}
5) Interval:
   x̄ ± E = {mean:.{decimal}f} ± {moe:.{decimal}f} → ({lower:.{decimal}f}, {upper:.{decimal}f})

Interpretation:
  We are {conf*100:.1f}% confident that the true population mean lies between {lower:.{decimal}f} and {upper:.{decimal}f}.
=====================
""")

            # Descriptive Table
            summary = pd.DataFrame({
                "Statistic": [
                    "Count (n)", "Mean", "Standard Deviation", "Standard Error",
                    "t Critical", "Margin of Error", "CI Lower", "CI Upper"
                ],
                "Value": [
                    int(n), round_value(mean, decimal), round_value(s, decimal),
                    round_value(se, decimal), round_value(t_crit, decimal),
                    round_value(moe, decimal), round_value(lower, decimal), round_value(upper, decimal)
                ]
            })
            st.dataframe(summary, use_container_width=True)

    # ==========================================================
    # 6. Sample Size for Mean
    # ==========================================================
    elif choice == categories[5]:
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        sigma = st.number_input("Population SD (σ)", min_value=0.0, format="%.4f")
        E = st.number_input("Margin of error (E)", min_value=0.000001, value=0.05, step=0.001, format="%.6f")

        if st.button("👨‍💻 Calculate"):
            if sigma < 0:
                st.error("❌ σ must be ≥ 0.")
                st.stop()

            z = stats.norm.ppf((1 + conf)/2)
            n_req = (z * sigma / E)**2
            n_ceiled = int(np.ceil(n_req))

            st.latex(r"n \;=\; \left(\frac{z_{\alpha/2}\,\sigma}{E}\right)^2")
            st.text(f"""
=====================
Sample Size for Mean
=====================
1) Inputs:
   confidence = {conf:.3f}, z_(α/2) = {z:.{decimal}f}, σ = {sigma}, E = {E}
2) Compute:
   n = ( z*σ/E )^2 = ( {z:.{decimal}f} * {sigma} / {E} )^2 = {n_req:.{decimal}f}
3) Round up:
   n (required) = {n_ceiled}

Interpretation:
  A sample of at least {n_ceiled} is required to estimate the mean with margin of error {E} at {conf*100:.1f}% confidence.
""")

    # ==========================================================
    # 7–10. Variance and SD Confidence Intervals (step-by-step)
    # ==========================================================
    else:
        st.subheader(f"📈 {choice}")
        data = None
        if "With Data" in choice:
            data = load_uploaded_data()
            if data is not None:
                n = len(data)
                s2 = float(np.var(data, ddof=1))
                s = float(np.sqrt(s2))
            else:
                st.warning("⚠️ Please upload data to proceed.")
                return
        else:
            n = st.number_input("Sample size (n)", min_value=2, step=1)
            if "Variance" in choice:
                s2 = st.number_input("Sample variance (s²)", min_value=0.0, format="%.6f")
                s = float(np.sqrt(s2))
            else:
                s = st.number_input("Sample SD (s)", min_value=0.0, format="%.6f")
                s2 = float(s**2)

        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        df = int(n - 1)
        chi2_lower = float(stats.chi2.ppf((1 - conf)/2, df))
        chi2_upper = float(stats.chi2.ppf(1 - (1 - conf)/2, df))

        st.latex(r"""
\text{Variance CI: } 
\left(\frac{(n-1)s^2}{\chi^2_{(1-\alpha/2),\,df}},\;
      \frac{(n-1)s^2}{\chi^2_{(\alpha/2),\,df}}\right)
\quad\quad
\text{SD CI: }
\left(\sqrt{\frac{(n-1)s^2}{\chi^2_{(1-\alpha/2),\,df}}},\;
      \sqrt{\frac{(n-1)s^2}{\chi^2_{(\alpha/2),\,df}}}\right)
""")

        if st.button("👨‍💻 Calculate"):
            numer = df * s2
            var_lower = numer / chi2_upper
            var_upper = numer / chi2_lower
            sd_lower, sd_upper = float(np.sqrt(var_lower)), float(np.sqrt(var_upper))

            st.text(f"""
=====================
{choice}
=====================
1) Inputs:
   n = {int(n)}, df = {df}, s² = {s2:.{decimal}f}, s = {s:.{decimal}f}, confidence = {conf:.3f}
2) Chi-square critical values:
   χ²_(1-α/2, df) = {chi2_upper:.{decimal}f}
   χ²_(α/2, df)   = {chi2_lower:.{decimal}f}
3) Compute (n-1)s²:
   df * s² = {df} * {s2:.{decimal}f} = {numer:.{decimal}f}
4) Variance bounds:
   Lower = {numer:.{decimal}f} / {chi2_upper:.{decimal}f} = {var_lower:.{decimal}f}
   Upper = {numer:.{decimal}f} / {chi2_lower:.{decimal}f} = {var_upper:.{decimal}f}
5) SD bounds:
   Lower = sqrt({var_lower:.{decimal}f}) = {sd_lower:.{decimal}f}
   Upper = sqrt({var_upper:.{decimal}f}) = {sd_upper:.{decimal}f}

Results:
  {conf*100:.1f}% CI for Variance = ({var_lower:.{decimal}f}, {var_upper:.{decimal}f})
  {conf*100:.1f}% CI for SD       = ({sd_lower:.{decimal}f}, {sd_upper:.{decimal}f})

Interpretation:
  We are {conf*100:.1f}% confident that the true population variance lies between {var_lower:.{decimal}f} and {var_upper:.{decimal}f},
  and the true population standard deviation lies between {sd_lower:.{decimal}f} and {sd_upper:.{decimal}f}.
""")

            st.dataframe(pd.DataFrame({
                "Statistic": [
                    "Degrees of Freedom", "(n-1)s²",
                    "χ² Lower (α/2)", "χ² Upper (1-α/2)",
                    "Variance CI (Lower)", "Variance CI (Upper)",
                    "SD CI (Lower)", "SD CI (Upper)"
                ],
                "Value": [
                    df, round_value(numer, decimal),
                    round_value(chi2_lower, decimal), round_value(chi2_upper, decimal),
                    round_value(var_lower, decimal), round_value(var_upper, decimal),
                    round_value(sd_lower, decimal), round_value(sd_upper, decimal)
                ]
            }), use_container_width=True)


# ==========================================================
# Run app directly
# ==========================================================
if __name__ == "__main__":
    run()


       
