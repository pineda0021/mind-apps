# ==========================================================
# confidence_intervals_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

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

# ---------- Plotting helpers ----------
def _style_axes(ax, title, xlabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)

def plot_ci_normal(center, se, lower, upper, title="Normal-based CI", xlabel="Value"):
    # Plot Normal(center, se)
    x = np.linspace(center - 4*se, center + 4*se, 800)
    y = stats.norm.pdf(x, loc=center, scale=se)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(x, y, lw=2)
    ax.fill_between(x, 0, y, where=(x >= lower) & (x <= upper), alpha=0.35)
    ax.axvline(center, color="red", lw=2, linestyle="--", label="Center")
    ax.axvline(lower, color="black", lw=1.5)
    ax.axvline(upper, color="black", lw=1.5)
    _style_axes(ax, title, xlabel)
    ax.legend(loc="upper right")
    st.pyplot(fig)

def plot_ci_t(center, se, df, lower, upper, title="t-based CI", xlabel="Value"):
    # Plot t with scale se, centered at center
    x = np.linspace(center - 4*se, center + 4*se, 800)
    y = stats.t.pdf((x - center)/se, df) / se  # scaled t density
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(x, y, lw=2)
    ax.fill_between(x, 0, y, where=(x >= lower) & (x <= upper), alpha=0.35)
    ax.axvline(center, color="red", lw=2, linestyle="--", label="Center")
    ax.axvline(lower, color="black", lw=1.5)
    ax.axvline(upper, color="black", lw=1.5)
    _style_axes(ax, title, xlabel)
    ax.legend(loc="upper right")
    st.pyplot(fig)

def plot_ci_chi2(df, chi2_lower, chi2_upper, title="χ²-based CI (quantiles)", xlabel="χ² value"):
    # Plot chi-square(df) with shaded region between quantiles
    x_max = stats.chi2.ppf(0.999, df)
    x = np.linspace(0, x_max, 800)
    y = stats.chi2.pdf(x, df)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(x, y, lw=2)
    ax.fill_between(x, 0, y, where=(x >= chi2_lower) & (x <= chi2_upper), alpha=0.35)
    ax.axvline(chi2_lower, color="black", lw=1.5, label="χ² lower/upper")
    ax.axvline(chi2_upper, color="black", lw=1.5)
    _style_axes(ax, title, xlabel)
    ax.legend(loc="upper right")
    st.pyplot(fig)

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
    # 1) Proportion CI (p, z)
    # ==========================================================
    if choice == categories[0]:
        x = st.number_input("Number of successes (x)", min_value=0, step=1)
        n = st.number_input("Sample size (n)", min_value=max(1, int(x)), step=1)
        conf = st.number_input("Confidence level (0–1)", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate", key="btn_prop_ci"):
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

            if st.checkbox("📊 Show graph of confidence interval", key="plot_prop_ci"):
                plot_ci_normal(
                    center=p_hat, se=se, lower=lower, upper=upper,
                    title="Proportion CI (Normal Approximation)", xlabel="p"
                )

    # ==========================================================
    # 2) Sample Size for Proportion (p, z, E)
    # ==========================================================
    elif choice == categories[1]:
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        p_est = st.number_input("Estimated proportion (p̂)", value=0.5, min_value=0.0, max_value=1.0, step=0.001)
        E = st.number_input("Margin of error (E)", value=0.05, min_value=0.000001, step=0.001)

        if st.button("👨‍💻 Calculate", key="btn_n_prop"):
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
    # 3) Mean CI (σ known, z)
    # ==========================================================
    elif choice == categories[2]]:
        mean = st.number_input("Sample mean (x̄)")
        sigma = st.number_input("Population SD (σ)", min_value=0.0, format="%.4f")
        n = st.number_input("Sample size (n)", min_value=1, step=1)
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate", key="btn_mean_z"):
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

            if st.checkbox("📊 Show graph of confidence interval", key="plot_mean_z"):
                plot_ci_normal(
                    center=mean, se=se, lower=lower, upper=upper,
                    title="Mean CI (σ known, z)", xlabel="μ"
                )

    # ==========================================================
    # 4) Mean CI (s given, t)
    # ==========================================================
    elif choice == categories[3]:
        mean = st.number_input("Sample mean (x̄)")
        s = st.number_input("Sample SD (s)", min_value=0.0, format="%.4f")
        n = st.number_input("Sample size (n)", min_value=2, step=1)
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")

        if st.button("👨‍💻 Calculate", key="btn_mean_t"):
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

            if st.checkbox("📊 Show graph of confidence interval", key="plot_mean_t"):
                plot_ci_t(
                    center=mean, se=se, df=df, lower=lower, upper=upper,
                    title="Mean CI (s given, t)", xlabel="μ"
                )

    # ==========================================================
    # 5) Mean CI (with data, t)
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

        if st.button("👨‍💻 Calculate", key="btn_mean_data"):
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

            # Optional summary table
            summary = pd.DataFrame({
                "Statistic": ["n", "Mean (x̄)", "SD (s)", "SE", "t critical", "MOE", "CI lower", "CI upper"],
                "Value": [n, round_value(mean, decimal), round_value(s, decimal), round_value(se, decimal),
                          round_value(t_crit, decimal), round_value(moe, decimal),
                          round_value(lower, decimal), round_value(upper, decimal)]
            })
            st.dataframe(summary, use_container_width=True)

            if st.checkbox("📊 Show graph of confidence interval", key="plot_mean_data"):
                plot_ci_t(
                    center=mean, se=se, df=df, lower=lower, upper=upper,
                    title="Mean CI (with data, t)", xlabel="μ"
                )

    # ==========================================================
    # 6) Sample Size for Mean (σ known, z, E)
    # ==========================================================
    elif choice == categories[5]:
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        sigma = st.number_input("Population SD (σ)", min_value=0.0, format="%.4f")
        E = st.number_input("Margin of error (E)", value=0.05, min_value=0.000001, step=0.001, format="%.6f")

        if st.button("👨‍💻 Calculate", key="btn_n_mean"):
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
    # 7) Variance & SD CI (χ²) — WITHOUT DATA (flexible input)
    # ==========================================================
    elif choice == categories[6]:
        n = st.number_input("Sample size (n)", min_value=2, step=1)
        input_type = st.radio(
            "Provide summary input:",
            ["Enter sample variance (s²)", "Enter sample standard deviation (s)"],
            horizontal=True
        )
        if input_type == "Enter sample variance (s²)":
            s2 = st.number_input("Sample variance (s²)", min_value=0.0, format="%.6f")
            s = np.sqrt(s2)
        else:
            s = st.number_input("Sample standard deviation (s)", min_value=0.0, format="%.6f")
            s2 = s**2

        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        df = int(n - 1)
        chi2_lower = stats.chi2.ppf((1 - conf)/2, df)
        chi2_upper = stats.chi2.ppf(1 - (1 - conf)/2, df)

        if st.button("👨‍💻 Calculate", key="btn_var_sd_no_data"):
            numer = df * s2
            var_lower, var_upper = numer / chi2_upper, numer / chi2_lower
            sd_lower, sd_upper = np.sqrt(var_lower), np.sqrt(var_upper)

            st.latex(r"\text{Var CI: } \left(\dfrac{(n-1)s^2}{\chi^2_{(1-\alpha/2),\,df}},\; \dfrac{(n-1)s^2}{\chi^2_{(\alpha/2),\,df}}\right)\quad \text{SD CI: } \left(\sqrt{\dfrac{(n-1)s^2}{\chi^2_{(1-\alpha/2),\,df}}},\; \sqrt{\dfrac{(n-1)s^2}{\chi^2_{(\alpha/2),\,df}}}\right)")
            st.text(f"""
=====================
Confidence Interval for Variance & SD (χ²) — Without Data
=====================
1) Inputs: n={int(n)}, df={df}, s²={s2:.{decimal}f}, s={s:.{decimal}f}, confidence={conf:.3f}
2) χ² upper={chi2_upper:.{decimal}f}, χ² lower={chi2_lower:.{decimal}f}
3) Var CI=({var_lower:.{decimal}f}, {var_upper:.{decimal}f})
4) SD CI=({sd_lower:.{decimal}f}, {sd_upper:.{decimal}f})

Interpretation:
  We are {conf*100:.1f}% confident that the population variance lies between {var_lower:.{decimal}f} and {var_upper:.{decimal}f},
  and the population SD lies between {sd_lower:.{decimal}f} and {sd_upper:.{decimal}f}.
""")

            if st.checkbox("📊 Show graph of χ² interval (quantiles)", key="plot_chi2_no_data"):
                plot_ci_chi2(
                    df=df, chi2_lower=chi2_lower, chi2_upper=chi2_upper,
                    title="χ² Quantiles Shaded for CI", xlabel="χ²"
                )

    # ==========================================================
    # 8) Variance & SD CI (χ²) — WITH DATA
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

        if st.button("👨‍💻 Calculate", key="btn_var_sd_data"):
            if data is None or len(data) < 2:
                st.warning("⚠️ Provide at least two data points.")
                return

            n = len(data)
            s2 = np.var(data, ddof=1)
            s = np.sqrt(s2)
            df = int(n - 1)
            chi2_lower = stats.chi2.ppf((1 - conf)/2, df)
            chi2_upper = stats.chi2.ppf(1 - (1 - conf)/2, df)
            numer = df * s2
            var_lower, var_upper = numer / chi2_upper, numer / chi2_lower
            sd_lower, sd_upper = np.sqrt(var_lower), np.sqrt(var_upper)

            st.latex(r"\text{Var CI: } \left(\dfrac{(n-1)s^2}{\chi^2_{(1-\alpha/2),\,df}},\; \dfrac{(n-1)s^2}{\chi^2_{(\alpha/2),\,df}}\right)\quad \text{SD CI: } \left(\sqrt{\dfrac{(n-1)s^2}{\chi^2_{(1-\alpha/2),\,df}}},\; \sqrt{\dfrac{(n-1)s^2}{\chi^2_{(\alpha/2),\,df}}}\right)")
            st.text(f"""
=====================
Confidence Interval for Variance & SD (with data, χ²)
=====================
1) Inputs (from data): n={n}, df={df}, s²={s2:.{decimal}f}, s={s:.{decimal}f}, confidence={conf:.3f}
2) χ² upper={chi2_upper:.{decimal}f}, χ² lower={chi2_lower:.{decimal}f}
3) Var CI=({var_lower:.{decimal}f}, {var_upper:.{decimal}f})
4) SD CI=({sd_lower:.{decimal}f}, {sd_upper:.{decimal}f})

Interpretation:
  We are {conf*100:.1f}% confident that the population variance lies between {var_lower:.{decimal}f} and {var_upper:.{decimal}f},
  and the population SD lies between {sd_lower:.{decimal}f} and {sd_upper:.{decimal}f}.
=====================
""")

            # Optional summary table
            summary = pd.DataFrame({
                "Statistic": ["n", "Variance (s²)", "SD (s)", "df",
                              "χ² lower (α/2)", "χ² upper (1−α/2)",
                              "Var CI lower", "Var CI upper", "SD CI lower", "SD CI upper"],
                "Value": [n, round_value(s2, decimal), round_value(s, decimal), df,
                          round_value(chi2_lower, decimal), round_value(chi2_upper, decimal),
                          round_value(var_lower, decimal), round_value(var_upper, decimal),
                          round_value(sd_lower, decimal), round_value(sd_upper, decimal)]
            })
            st.dataframe(summary, use_container_width=True)

            if st.checkbox("📊 Show graph of χ² interval (quantiles)", key="plot_chi2_data"):
                plot_ci_chi2(
                    df=df, chi2_lower=chi2_lower, chi2_upper=chi2_upper,
                    title="χ² Quantiles Shaded for CI", xlabel="χ²"
                )

# ==========================================================
# Run app directly
# ==========================================================
if __name__ == "__main__":
    run()
