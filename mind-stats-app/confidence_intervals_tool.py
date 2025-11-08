# ==========================================================
# confidence_intervals_tool.py
# Professor Edition v3
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

def plot_ci_normal(center, se, lower, upper, conf, title="Normal-based CI", xlabel="Value"):
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
    st.caption(
        f"🧠 The shaded region represents the middle {(conf*100):.1f}% "
        "of the normal distribution corresponding to the confidence interval."
    )

def plot_ci_t(center, se, df, lower, upper, conf, title="t-based CI", xlabel="Value"):
    x = np.linspace(center - 4*se, center + 4*se, 800)
    y = stats.t.pdf((x - center)/se, df) / se
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(x, y, lw=2)
    ax.fill_between(x, 0, y, where=(x >= lower) & (x <= upper), alpha=0.35)
    ax.axvline(center, color="red", lw=2, linestyle="--", label="Center")
    ax.axvline(lower, color="black", lw=1.5)
    ax.axvline(upper, color="black", lw=1.5)
    _style_axes(ax, title, xlabel)
    ax.legend(loc="upper right")
    st.pyplot(fig)
    st.caption(
        f"🧠 The shaded region represents the middle {(conf*100):.1f}% "
        f"of the t-distribution with df = {df}, corresponding to the confidence interval."
    )

def plot_ci_chi2(df, chi2_lower, chi2_upper, conf, title="χ²-based CI (quantiles)", xlabel="χ² value"):
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
    st.caption(
        f"🧠 The shaded region shows the middle {(conf*100):.1f}% "
        f"of the χ²-distribution (df = {df}), bounded by χ²₍α∕2₎ and χ²₍1−α∕2₎."
    )

# ==========================================================
# Main App
# ==========================================================
def run():
    st.header("🔮 MIND: Confidence Interval Calculator (Professor Edition v3)")
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

    # --- Proportion (p, z)
    if choice == categories[0]:
        x = st.number_input("Number of successes (x)", min_value=0, step=1)
        n = st.number_input("Sample size (n)", min_value=max(1, int(x)), step=1)
        conf = st.number_input("Confidence level (0–1)", value=0.95, format="%.3f")
        if st.button("👨‍💻 Calculate"):
            p_hat = x/n
            z = stats.norm.ppf((1+conf)/2)
            se = np.sqrt(p_hat*(1-p_hat)/n)
            moe = z*se
            lower, upper = p_hat-moe, p_hat+moe
            st.latex(r"\hat{p} \;\pm\; z_{\alpha/2}\sqrt{\dfrac{\hat{p}(1-\hat{p})}{n}}")
            st.text(f"""
=====================
Confidence Interval for Proportion (p, z)
=====================
1) Inputs: x={int(x)}, n={int(n)}, p̂={p_hat:.{decimal}f}
2) z_(α/2)={z:.{decimal}f}
3) SE=sqrt[p̂(1−p̂)/n]={se:.{decimal}f}
4) E=z*SE={moe:.{decimal}f}
5) CI=({lower:.{decimal}f}, {upper:.{decimal}f})

Interpretation:
  We are {conf*100:.1f}% confident that the true population proportion lies between {lower:.{decimal}f} and {upper:.{decimal}f}.
""")
            if st.checkbox("📊 Show graph of confidence interval"):
                plot_ci_normal(p_hat, se, lower, upper, conf, "Proportion CI (Normal)", "p")

    # --- Sample Size for Proportion
    elif choice == categories[1]:
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        p_est = st.number_input("Estimated proportion (p̂)", value=0.5, min_value=0.0, max_value=1.0)
        E = st.number_input("Margin of error (E)", value=0.05, min_value=0.000001, step=0.001)
        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1+conf)/2)
            n_req = p_est*(1-p_est)*(z/E)**2
            n_ceiled = int(np.ceil(n_req))
            st.latex(r"n \;=\; \hat{p}(1-\hat{p})\!\left(\dfrac{Z_{\alpha/2}}{E}\right)^{2}")
            st.text(f"Required n ≈ {n_req:.{decimal}f} → {n_ceiled}")

    # --- Mean (σ known, z)
    elif choice == categories[2]:
        mean = st.number_input("Sample mean (x̄)")
        sigma = st.number_input("Population SD (σ)", min_value=0.0)
        n = st.number_input("Sample size (n)", min_value=1, step=1)
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1+conf)/2)
            se = sigma/np.sqrt(n)
            moe = z*se
            lower, upper = mean-moe, mean+moe
            st.latex(r"\bar{X} \;\pm\; z_{\alpha/2}\!\left(\dfrac{\sigma}{\sqrt{n}}\right)")
            st.text(f"CI = ({lower:.{decimal}f}, {upper:.{decimal}f})")
            if st.checkbox("📊 Show graph of confidence interval"):
                plot_ci_normal(mean, se, lower, upper, conf, "Mean CI (σ known, z)", "μ")

    # --- Mean (s given, t)
    elif choice == categories[3]:
        mean = st.number_input("Sample mean (x̄)")
        s = st.number_input("Sample SD (s)", min_value=0.0)
        n = st.number_input("Sample size (n)", min_value=2, step=1)
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        if st.button("👨‍💻 Calculate"):
            df = int(n-1)
            t_crit = stats.t.ppf((1+conf)/2, df)
            se = s/np.sqrt(n)
            moe = t_crit*se
            lower, upper = mean-moe, mean+moe
            st.latex(r"\bar{X} \;\pm\; t_{\alpha/2,\,n-1}\!\left(\dfrac{s}{\sqrt{n}}\right)")
            st.text(f"CI = ({lower:.{decimal}f}, {upper:.{decimal}f})")
            if st.checkbox("📊 Show graph of confidence interval"):
                plot_ci_t(mean, se, df, lower, upper, conf, "Mean CI (s given, t)", "μ")

    # --- Mean (with data, t)
    elif choice == categories[4]:
        st.subheader("📊 Confidence Interval for Mean (with data, t)")
        data = load_uploaded_data()
        raw_input = st.text_area("Or enter comma-separated values:")
        if data is None and raw_input:
            try:
                data = np.array([float(x.strip()) for x in raw_input.split(",") if x.strip()])
            except:
                st.error("❌ Invalid numeric input.")
                return
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        if st.button("👨‍💻 Calculate"):
            if data is None or len(data)<2:
                st.warning("⚠️ Provide at least two data points."); return
            n=len(data); mean=np.mean(data); s=np.std(data,ddof=1); df=n-1
            t_crit=stats.t.ppf((1+conf)/2,df)
            se=s/np.sqrt(n); moe=t_crit*se
            lower,upper=mean-moe,mean+moe
            st.latex(r"\bar{X} \;\pm\; t_{\alpha/2,\,n-1}\!\left(\dfrac{s}{\sqrt{n}}\right)")
            st.text(f"CI = ({lower:.{decimal}f}, {upper:.{decimal}f})")
            summary=pd.DataFrame({
                "Statistic":["n","Mean","SD","SE","t-crit","MOE","CI low","CI up"],
                "Value":[n,round_value(mean,decimal),round_value(s,decimal),
                         round_value(se,decimal),round_value(t_crit,decimal),
                         round_value(moe,decimal),round_value(lower,decimal),round_value(upper,decimal)]
            })
            st.dataframe(summary,use_container_width=True)
            if st.checkbox("📊 Show graph of confidence interval"):
                plot_ci_t(mean,se,df,lower,upper,conf,"Mean CI (with data, t)","μ")

    # --- Sample Size for Mean
    elif choice == categories[5]:
        conf = st.number_input("Confidence level", value=0.95, format="%.3f")
        sigma = st.number_input("Population SD (σ)", min_value=0.0)
        E = st.number_input("Margin of error (E)", value=0.05)
        if st.button("👨‍💻 Calculate"):
            z = stats.norm.ppf((1+conf)/2)
            n_req=(z*sigma/E)**2; n_ceiled=int(np.ceil(n_req))
            st.latex(r"n \;=\; \left(\dfrac{z_{\alpha/2}\,\sigma}{E}\right)^{2}")
            st.text(f"Required n ≈ {n_req:.{decimal}f} → {n_ceiled}")

    # --- Variance & SD (χ²) without data
    elif choice == categories[6]:
        n=st.number_input("Sample size (n)",min_value=2,step=1)
        input_type=st.radio("Provide summary input:",
            ["Enter sample variance (s²)","Enter sample standard deviation (s)"],horizontal=True)
        if input_type=="Enter sample variance (s²)":
            s2=st.number_input("Sample variance (s²)",min_value=0.0)
            s=np.sqrt(s2)
        else:
            s=st.number_input("Sample standard deviation (s)",min_value=0.0)
            s2=s**2
        conf=st.number_input("Confidence level",value=0.95,format="%.3f")
        df=int(n-1)
        chi2_lower=stats.chi2.ppf((1-conf)/2,df)
        chi2_upper=stats.chi2.ppf(1-(1-conf)/2,df)
        if st.button("👨‍💻 Calculate"):
            numer=df*s2
            var_lower, var_upper = numer/chi2_upper, numer/chi2_lower
            sd_lower, sd_upper = np.sqrt(var_lower), np.sqrt(var_upper)
            st.latex(r"\text{Var CI and SD CI formulas using χ²}")
            st.text(f"Var CI=({var_lower:.{decimal}f}, {var_upper:.{decimal}f})\nSD CI=({sd_lower:.{decimal}f}, {sd_upper:.{decimal}f})")
            if st.checkbox("📊 Show graph of χ² interval (quantiles)"):
                plot_ci_chi2(df,chi2_lower,chi2_upper,conf,"χ² Quantiles for CI","χ²")

    # --- Variance & SD (χ²) with data
    elif choice == categories[7]:
        st.subheader("📊 Confidence Interval for Variance & SD (with data, χ²)")
        data=load_uploaded_data()
        raw_input=st.text_area("Or enter comma-separated values:")
        if data is None and raw_input:
            try: data=np.array([float(x.strip()) for x in raw_input.split(",") if x.strip()])
            except: st.error("❌ Invalid numeric input."); return
        conf=st.number_input("Confidence level",value=0.95,format="%.3f")
        if st.button("👨‍💻 Calculate"):
            if data is None or len(data)<2:
                st.warning("⚠️ Provide at least two data points."); return
            n=len(data); s2=np.var(data,ddof=1); s=np.sqrt(s2); df=n-1
            chi2_lower=stats.chi2.ppf((1-conf)/2,df); chi2_upper=stats.chi2.ppf(1-(1-conf)/2,df)
            numer=df*s2
            var_lower, var_upper=numer/chi2_upper, numer/chi2_lower
            sd_lower, sd_upper=np.sqrt(var_lower),np.sqrt(var_upper)
            st.text(f"Var CI=({var_lower:.{decimal}f}, {var_upper:.{decimal}f})\nSD CI=({sd_lower:.{decimal}f}, {sd_upper:.{decimal}f})")
            if st.checkbox("📊 Show graph of χ² interval (quantiles)"):
                plot_ci_chi2(df,chi2_lower,chi2_upper,conf,"χ² Quantiles for CI","χ²")

# ==========================================================
# Run app directly
# ==========================================================
if __name__ == "__main__":
    run()

      
