import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# Rounding helper
def round_value(value, decimals=4):
    return round(value, decimals)

def run():
    st.header("📊 Confidence Interval Calculator")
    
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

    # CI for Proportion
    if category == "Confidence Interval for Proportion":
        x = st.number_input("Enter the number of successes", min_value=0, step=1)
        n = st.number_input("Enter the sample size", min_value=1, step=1)
        confidence_level = st.number_input("Confidence level (e.g., 0.95)", min_value=0.0, max_value=1.0, value=0.95)
        
        if st.button("Calculate"):
            p_hat = x / n
            se = np.sqrt(p_hat * (1 - p_hat) / n)
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            me = z_score * se
            lower = p_hat - me
            upper = p_hat + me
            
            st.latex(f"\\text{{Sample Proportion: }} {round_value(p_hat, decimal)}")
            st.latex(f"\\text{{Critical Value (Z-Score): }} {round_value(z_score, decimal)}")
            st.latex(f"\\text{{{confidence_level*100}% Confidence Interval: }} ({round_value(lower, decimal)}, {round_value(upper, decimal)})")
    
    # Sample Size for Proportion
    elif category == "Sample Size for Proportion":
        confidence_level = st.number_input("Confidence level (e.g., 0.95)", min_value=0.0, max_value=1.0, value=0.95)
        est_prop = st.number_input("Estimated proportion (0.5 if unknown)", min_value=0.0, max_value=1.0, value=0.5)
        margin_error = st.number_input("Margin of error", min_value=0.0, value=0.05)
        
        if st.button("Calculate"):
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            n_required = (z_score**2 * est_prop * (1 - est_prop)) / (margin_error**2)
            n_required = int(np.ceil(n_required))
            st.latex(f"\\text{{Critical Value (Z-Score): }} {round_value(z_score,4)}")
            st.latex(f"\\text{{Required Sample Size: }} {n_required}")

    # CI for Mean (Known SD)
    elif category == "Confidence Interval for Mean (Known Standard Deviation)":
        mean = st.number_input("Sample Mean", value=0.0)
        sd = st.number_input("Population SD", min_value=0.0, value=1.0)
        n = st.number_input("Sample Size", min_value=1, step=1, value=30)
        confidence_level = st.number_input("Confidence level (e.g., 0.95)", min_value=0.0, max_value=1.0, value=0.95)
        
        if st.button("Calculate"):
            se = sd / np.sqrt(n)
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            me = z_score * se
            lower = mean - me
            upper = mean + me
            
            st.latex(f"\\text{{Critical Value (Z-Score): }} {round_value(z_score, decimal)}")
            st.latex(f"\\text{{{confidence_level*100}% Confidence Interval: }} ({round_value(lower, decimal)}, {round_value(upper, decimal)})")

    # CI Mean with Data
    elif category == "Confidence Interval for Mean (With Data)":
        file = st.file_uploader("Upload CSV or Excel (1 column of numeric data)", type=["csv", "xlsx"])
        confidence_level = st.number_input("Confidence level (e.g., 0.95)", min_value=0.0, max_value=1.0, value=0.95)

        if file is not None:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            data = df.select_dtypes(include=[np.number]).values.flatten()
            data = data[~np.isnan(data)]
            
            sample_mean = np.mean(data)
            sample_sd = np.std(data, ddof=1)
            n = len(data)
            dfree = n - 1
            t_crit = stats.t.ppf((1 + confidence_level) / 2, df=dfree)
            se = sample_sd / np.sqrt(n)
            me = t_crit * se
            lower = sample_mean - me
            upper = sample_mean + me
            
            st.latex(f"\\text{{Sample Mean: }} {round_value(sample_mean, decimal)}")
            st.latex(f"\\text{{Sample SD: }} {round_value(sample_sd, decimal)}")
            st.latex(f"\\text{{Critical Value (t-Score): }} {round_value(t_crit, decimal)}")
            st.latex(f"\\text{{{confidence_level*100}% Confidence Interval: }} ({round_value(lower, decimal)}, {round_value(upper, decimal)})")

            # Histogram
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

    # Sample Size for Mean
    elif category == "Sample Size for Mean":
        confidence_level = st.number_input("Confidence level (e.g., 0.95)", min_value=0.0, max_value=1.0, value=0.95)
        pop_sd = st.number_input("Population Standard Deviation", min_value=0.0, value=1.0)
        margin_error = st.number_input("Margin of Error", min_value=0.0, value=0.05)
        
        if st.button("Calculate"):
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            n_required = (z_score * pop_sd / margin_error)**2
            n_required = int(np.ceil(n_required))
            st.latex(f"\\text{{Critical Value (Z-Score): }} {round_value(z_score,4)}")
            st.latex(f"\\text{{Required Sample Size: }} {n_required}")

    # CI Variance and SD
    elif category in ["Confidence Interval for Variance (Without Data)", 
                      "Confidence Interval for Variance (With Data)",
                      "Confidence Interval for Standard Deviation (Without Data)",
                      "Confidence Interval for Standard Deviation (With Data)"]:
        use_data = "With Data" in category
        if use_data:
            file = st.file_uploader("Upload CSV or Excel (1 column of numeric data)", type=["csv", "xlsx"])
        else:
            n = st.number_input("Sample size", min_value=1, step=1)
            if "Variance" in category:
                sample_var = st.number_input("Sample Variance", min_value=0.0)
            else:
                sample_sd = st.number_input("Sample SD", min_value=0.0)

        confidence_level = st.number_input("Confidence level (e.g., 0.95)", min_value=0.0, max_value=1.0, value=0.95)

        if st.button("Calculate"):
            if use_data:
                if file is not None:
                    if file.name.endswith(".csv"):
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_excel(file)
                    data = df.select_dtypes(include=[np.number]).values.flatten()
                    data = data[~np.isnan(data)]
                    n = len(data)
                    if "Variance" in category:
                        sample_var = np.var(data, ddof=1)
                        sample_sd = np.sqrt(sample_var)
                    else:
                        sample_sd = np.std(data, ddof=1)
                        sample_var = sample_sd**2
            dfree = n - 1
            chi_lower = stats.chi2.ppf((1 - confidence_level) / 2, df=dfree)
            chi_upper = stats.chi2.ppf(1 - (1 - confidence_level) / 2, df=dfree)
            if "Variance" in category:
                lower = dfree * sample_var / chi_upper
                upper = dfree * sample_var / chi_lower
            else:
                lower = np.sqrt(dfree * sample_sd**2 / chi_upper)
                upper = np.sqrt(dfree * sample_sd**2 / chi_lower)
            st.latex(f"\\text{{Critical Values (Chi-Square): Lower = {round_value(chi_lower, decimal)}, Upper = {round_value(chi_upper, decimal)}}}")
            if "Variance" in category:
                st.latex(f"\\text{{Sample Variance: }} {round_value(sample_var, decimal)}")
                st.latex(f"\\text{{{confidence_level*100}% Confidence Interval: }} ({round_value(lower, decimal)}, {round_value(upper, decimal)})")
            else:
                st.latex(f"\\text{{Sample SD: }} {round_value(sample_sd, decimal)}")
                st.latex(f"\\text{{{confidence_level*100}% Confidence Interval: }} ({round_value(lower, decimal)}, {round_value(upper, decimal)})")
            
            # Histogram shading for with data
            if use_data:
                fig, ax = plt.subplots()
                ax.hist(data, bins='auto', alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(lower, color='red', linestyle='--', label='CI Lower')
                ax.axvline(upper, color='red', linestyle='--', label='CI Upper')
                if "Variance" in category:
                    mean_val = np.mean(data)
                    ax.axvline(mean_val, color='green', linestyle='-', label='Mean')
                else:
                    mean_val = np.mean(data)
                    ax.axvline(mean_val, color='green', linestyle='-', label='Mean')
                ax.fill_betweenx([0, max(np.histogram(data, bins='auto')[0])], lower, upper, color='red', alpha=0.2)
                ax.set_title("Data Histogram with Confidence Interval")
                ax.set_xlabel("Values")
                ax.set_ylabel("Frequency")
                ax.legend()
                st.pyplot(fig)


