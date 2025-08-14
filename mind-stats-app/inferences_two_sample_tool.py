import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

st.set_page_config(page_title="MIND: Two-Sample Inference", layout="wide")
st.title("🧠 MIND: Two-Sample Inference Tool")

def run_two_sample_tool():
    # Sidebar header
    st.sidebar.header("📊 Two-Sample Inference")
    
    # Dropdown for test selection
    test_choice = st.sidebar.selectbox(
        "Select a Test",
        [
            "Two-Proportion Z-Test",
            "Confidence Interval for Proportion Difference",
            "Paired t-Test using Data",
            "Paired Confidence Interval using Data",
            "Paired t-Test using Summary Statistics",
            "Paired Confidence Interval using Summary Statistics",
            "Independent t-Test using Data",
            "Independent Confidence Interval using Data",
            "Independent t-Test using Summary Statistics",
            "Independent Confidence Interval using Summary Statistics",
            "F-Test for Standard Deviation using Data",
            "F-Test for Standard Deviation using Summary Statistics"
        ]
    )

    # ------------------- TWO-PROPORTION TESTS -------------------
    if test_choice in ["Two-Proportion Z-Test", "Confidence Interval for Proportion Difference"]:
        st.sidebar.subheader("Enter Sample Data")
        x1 = st.sidebar.number_input("Number of successes in Sample 1", min_value=0, step=1)
        n1 = st.sidebar.number_input("Sample size 1", min_value=1, step=1)
        x2 = st.sidebar.number_input("Number of successes in Sample 2", min_value=0, step=1)
        n2 = st.sidebar.number_input("Sample size 2", min_value=1, step=1)
        alpha = st.sidebar.number_input("Significance level α", value=0.05, min_value=0.001, max_value=0.5, step=0.01)
        
        if st.sidebar.button("Calculate"):
            # Calculations
            p1 = x1 / n1
            p2 = x2 / n2
            p_pool = (x1 + x2) / (n1 + n2)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            z_stat = (p1 - p2)/se
            p_val = 2*(1 - stats.norm.cdf(abs(z_stat)))
            z_crit = stats.norm.ppf(1 - alpha/2)

            # Output
            st.subheader("Step-by-Step Calculation")
            st.latex(r"\hat{p}_1 = \frac{x_1}{n_1} = " + f"{p1:.3f}")
            st.latex(r"\hat{p}_2 = \frac{x_2}{n_2} = " + f"{p2:.3f}")
            st.latex(r"\hat{p} = \frac{x_1 + x_2}{n_1 + n_2} = " + f"{p_pool:.3f}")
            st.latex(r"SE = \sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)} = " + f"{se:.3f}")
            st.latex(r"Z = \frac{\hat{p}_1 - \hat{p}_2}{SE} = " + f"{z_stat:.3f}")
            st.latex(r"Z_{\alpha/2} = " + f"{z_crit:.3f}")
            st.write(f"P-Value = {p_val:.4f}")
            st.write("Decision:", "Reject H₀" if abs(z_stat) > z_crit else "Fail to reject H₀")

            if test_choice == "Confidence Interval for Proportion Difference":
                ci_lower = (p1 - p2) - z_crit * np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
                ci_upper = (p1 - p2) + z_crit * np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
                st.latex(r"CI: (\hat{p}_1 - \hat{p}_2) \pm Z_{\alpha/2} \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}")
                st.write(f"Confidence Interval = ({ci_lower:.3f}, {ci_upper:.3f})")

    # ------------------- PAIRED T-TESTS -------------------
    elif test_choice in ["Paired t-Test using Data", "Paired Confidence Interval using Data"]:
        st.sidebar.subheader("Upload Paired Data (CSV)")
        uploaded_file = st.sidebar.file_uploader("Upload CSV with two columns: Sample1, Sample2", type="csv")
        alpha = st.sidebar.number_input("Significance level α", value=0.05, min_value=0.001, max_value=0.5, step=0.01)

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if "Sample1" not in df.columns or "Sample2" not in df.columns:
                st.error("CSV must have columns 'Sample1' and 'Sample2'")
            else:
                diff = df["Sample1"] - df["Sample2"]
                mean_diff = np.mean(diff)
                sd_diff = np.std(diff, ddof=1)
                n = len(diff)
                se = sd_diff / np.sqrt(n)
                t_stat = mean_diff / se
                t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
                p_val = 2*(1 - stats.t.cdf(abs(t_stat), df=n-1))

                st.subheader("Step-by-Step Calculation")
                st.latex(r"\bar{d} = " + f"{mean_diff:.3f}")
                st.latex(r"s_d = " + f"{sd_diff:.3f}")
                st.latex(r"SE = s_d / \sqrt{n} = {se:.3f}")
                st.latex(r"t = \bar{d} / SE = {t_stat:.3f}")
                st.latex(r"t_{{\alpha/2, n-1}} = {t_crit:.3f}")
                st.write(f"P-Value = {p_val:.4f}")
                st.write("Decision:", "Reject H₀" if abs(t_stat) > t_crit else "Fail to reject H₀")

                if test_choice == "Paired Confidence Interval using Data":
                    ci_lower = mean_diff - t_crit*se
                    ci_upper = mean_diff + t_crit*se
                    st.latex(r"CI = \bar{d} \pm t_{\alpha/2,n-1} SE")
                    st.write(f"Confidence Interval = ({ci_lower:.3f}, {ci_upper:.3f})")

    # ------------------- PAIRED SUMMARY STATISTICS -------------------
    elif test_choice in ["Paired t-Test using Summary Statistics", "Paired Confidence Interval using Summary Statistics"]:
        st.sidebar.subheader("Enter Summary Statistics")
        mean_diff = st.sidebar.number_input("Mean of differences", value=0.0)
        sd_diff = st.sidebar.number_input("Standard deviation of differences", value=1.0)
        n = st.sidebar.number_input("Sample size", min_value=2, step=1)
        alpha = st.sidebar.number_input("Significance level α", value=0.05, min_value=0.001, max_value=0.5, step=0.01)

        if st.sidebar.button("Calculate"):
            se = sd_diff / np.sqrt(n)
            t_stat = mean_diff / se
            t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
            p_val = 2*(1 - stats.t.cdf(abs(t_stat), df=n-1))

            st.subheader("Step-by-Step Calculation")
            st.latex(r"SE = " + f"{se:.3f}")
            st.latex(r"t = " + f"{t_stat:.3f}")
            st.latex(r"t_{{\alpha/2, n-1}} = {t_crit:.3f}")
            st.write(f"P-Value = {p_val:.4f}")
            st.write("Decision:", "Reject H₀" if abs(t_stat) > t_crit else "Fail to reject H₀")

            if test_choice == "Paired Confidence Interval using Summary Statistics":
                ci_lower = mean_diff - t_crit*se
                ci_upper = mean_diff + t_crit*se
                st.latex(r"CI = \bar{d} \pm t_{\alpha/2,n-1} SE")
                st.write(f"Confidence Interval = ({ci_lower:.3f}, {ci_upper:.3f})")

    # ------------------- INDEPENDENT T-TESTS -------------------
    # ... The same pattern applies for Independent t-Test using Data and Summary Statistics
    # ------------------- F-TESTS -------------------
    # ... The same pattern applies for F-Test using Data and Summary Statistics

if __name__ == "__main__":
    run_two_sample_tool()
