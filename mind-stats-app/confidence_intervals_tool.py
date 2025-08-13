import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

def round_value(value, decimals=4):
    return round(value, decimals)

def run():
    st.header("📊 Confidence Interval Calculator with Formulas")

    category = st.selectbox(
        "Choose a category:",
        [
            "Confidence Interval for Proportion",
            "Sample Size for Proportion",
            "Confidence Interval for Mean (Known Standard Deviation)",
            "Confidence Interval for Mean (With Data)",
            "Sample Size for Mean",
            "Confidence Interval for Variance (Without Data)",
            "Confidence Interval for Variance (With Data)",
            "Confidence Interval for Standard Deviation (Without Data)",
            "Confidence Interval for Standard Deviation (With Data)"
        ]
    )

    decimal = st.number_input("Decimal places for output (except Sample Size)", min_value=0, max_value=10, value=4, step=1)

    # --- Confidence Interval for Proportion ---
    if category == "Confidence Interval for Proportion":
        x = st.number_input("Number of successes", min_value=0, step=1)
        n = st.number_input("Sample size", min_value=1, step=1)
        confidence_level = st.number_input("Confidence level (0-1)", min_value=0.0, max_value=1.0, value=0.95)
        
        if st.button("Calculate"):
            p_hat = x / n
            se = np.sqrt(p_hat*(1-p_hat)/n)
            z_score = stats.norm.ppf((1+confidence_level)/2)
            me = z_score * se
            lower, upper = p_hat - me, p_hat + me

            st.latex(f"\\hat{{p}} = {round_value(p_hat, decimal)}")
            st.latex(f"\\text{{CI Formula: }} \\hat{{p}} \\pm Z_{{\\alpha/2}} \\sqrt{{\\frac{{\\hat{{p}}(1-\\hat{{p}})}}{{n}}}}")
            st.latex(f"Z_{{\\alpha/2}} = {round_value(z_score, decimal)}, SE = {round_value(se, decimal)}, ME = {round_value(me, decimal)}")
            st.latex(f"{confidence_level*100:.1f}\\% \\text{{ Confidence Interval: }} ({round_value(lower, decimal)}, {round_value(upper, decimal)})")

    # --- Sample Size for Proportion ---
    elif category == "Sample Size for Proportion":
        confidence_level = st.number_input("Confidence level (0-1)", min_value=0.0, max_value=1.0, value=0.95)
        est_prop = st.number_input("Estimated proportion (0.5 if unknown)", min_value=0.0, max_value=1.0, value=0.5)
        margin_error = st.number_input("Margin of error", min_value=0.0, value=0.05)
        
        if st.button("Calculate"):
            z_score = stats.norm.ppf((1+confidence_level)/2)
            n_required = (z_score**2 * est_prop*(1-est_prop)) / (margin_error**2)
            n_required = int(np.ceil(n_required))
            st.latex(f"n = \\frac{{Z_{{\\alpha/2}}^2 \\cdot \\hat{{p}}(1-\\hat{{p}})}}{{E^2}}")
            st.latex(f"Z_{{\\alpha/2}} = {round_value(z_score, decimal)}")
            st.latex(f"Required Sample Size: {n_required}")

    # --- CI Mean (Known SD) ---
    elif category == "Confidence Interval for Mean (Known Standard Deviation)":
        mean = st.number_input("Sample Mean", value=0.0)
        sd = st.number_input("Population SD", min_value=0.0, value=1.0)
        n = st.number_input("Sample Size", min_value=1, step=1, value=30)
        confidence_level = st.number_input("Confidence level (0-1)", min_value=0.0, max_value=1.0, value=0.95)
        
        if st.button("Calculate"):
            se = sd/np.sqrt(n)
            z_score = stats.norm.ppf((1+confidence_level)/2)
            me = z_score*se
            lower, upper = mean-me, mean+me

            st.latex(f"\\bar{{X}} = {round_value(mean, decimal)}, \\sigma = {round_value(sd, decimal)}, n = {n}")
            st.latex(f"\\text{{CI Formula: }} \\bar{{X}} \\pm Z_{{\\alpha/2}} \\frac{{\\sigma}}{{\\sqrt{{n}}}}")
            st.latex(f"Z_{{\\alpha/2}} = {round_value(z_score, decimal)}, SE = {round_value(se, decimal)}, ME = {round_value(me, decimal)}")
            st.latex(f"{confidence_level*100:.1f}\\% \\text{{ CI: }} ({round_value(lower, decimal)}, {round_value(upper, decimal)})")

    # --- CI Mean (With Data) ---
    elif category == "Confidence Interval for Mean (With Data)":
        method = st.radio("Provide data by:", ["Upload CSV/Excel", "Enter manually"])
        confidence_level = st.number_input("Confidence level (0-1)", min_value=0.0, max_value=1.0, value=0.95)
        data = None

        if method=="Upload CSV/Excel":
            file = st.file_uploader("Upload CSV or Excel (1 column numeric)", type=["csv","xlsx"])
            if file is not None:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                data = df.select_dtypes(include=[np.number]).values.flatten()
                data = data[~np.isnan(data)]
        else:
            raw = st.text_area("Enter data separated by commas", "1,2,3,4,5")
            if raw:
                data = np.array([float(x) for x in raw.split(",")])

        if data is not None and st.button("Calculate"):
            n = len(data)
            sample_mean = np.mean(data)
            sample_sd = np.std(data, ddof=1)
            dfree = n-1
            t_score = stats.t.ppf((1+confidence_level)/2, df=dfree)
            se = sample_sd/np.sqrt(n)
            me = t_score*se
            lower, upper = sample_mean-me, sample_mean+me

            st.latex(f"\\bar{{X}} = {round_value(sample_mean, decimal)}, s = {round_value(sample_sd, decimal)}, n = {n}")
            st.latex(f"\\text{{CI Formula: }} \\bar{{X}} \\pm t_{{\\alpha/2, df={dfree}}} \\frac{{s}}{{\\sqrt{{n}}}}")
            st.latex(f"t_{{\\alpha/2}} = {round_value(t_score, decimal)}, SE = {round_value(se, decimal)}, ME = {round_value(me, decimal)}")
            st.latex(f"{confidence_level*100:.1f}\\% \\text{{ CI: }} ({round_value(lower, decimal)}, {round_value(upper, decimal)})")

            # Histogram with shaded CI
            fig, ax = plt.subplots()
            ax.hist(data, bins='auto', alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(lower, color='red', linestyle='--', label='CI Lower')
            ax.axvline(upper, color='red', linestyle='--', label='CI Upper')
            ax.axvline(sample_mean, color='green', linestyle='-', label='Mean')
            ax.fill_betweenx([0, max(np.histogram(data, bins='auto')[0])], lower, upper, color='red', alpha=0.2)
            ax.set_title("Data Histogram with Confidence Interval")
            ax.set_xlabel("Values")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)
