import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

def run_two_sample_tool():

    st.header("👨🏻‍🔬 Two-Sample Inference")
    
    # Dropdown for test selection
    test_choice = st.selectbox(
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
        st.subheader("Enter Sample Data")
        x1 = st.number_input("Number of successes in Sample 1", min_value=0, step=1)
        n1 = st.number_input("Sample size 1", min_value=1, step=1)
        x2 = st.number_input("Number of successes in Sample 2", min_value=0, step=1)
        n2 = st.number_input("Sample size 2", min_value=1, step=1)
        alpha = st.number_input("Significance level α", value=0.05, min_value=0.001, max_value=0.5, step=0.01)
        
        if st.button("Calculate"):
            p1 = x1 / n1
            p2 = x2 / n2
            p_pool = (x1 + x2) / (n1 + n2)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            z_stat = (p1 - p2)/se
            p_val = 2*(1 - stats.norm.cdf(abs(z_stat)))
            z_crit = stats.norm.ppf(1 - alpha/2)

            st.subheader("Step-by-Step Calculation")
            st.latex(r"\hat{p}_1 = " + f"{p1:.3f}")
            st.latex(r"\hat{p}_2 = " + f"{p2:.3f}")
            st.latex(r"\hat{p} = " + f"{p_pool:.3f}")
            st.latex(r"SE = " + f"{se:.3f}")
            st.latex(r"Z = " + f"{z_stat:.3f}")
            st.latex(r"Z_{{\alpha/2}} = {z_crit:.3f}")
            st.write(f"P-Value = {p_val:.4f}")
            st.write("Decision:", "Reject H₀" if abs(z_stat) > z_crit else "Fail to reject H₀")

            if test_choice == "Confidence Interval for Proportion Difference":
                ci_lower = (p1 - p2) - z_crit * np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
                ci_upper = (p1 - p2) + z_crit * np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
                st.latex(r"CI = (\hat{p}_1 - \hat{p}_2) \pm Z_{\alpha/2} SE")
                st.write(f"Confidence Interval = ({ci_lower:.3f}, {ci_upper:.3f})")

    # ------------------- PAIRED T-TESTS -------------------
    elif test_choice in ["Paired t-Test using Data", "Paired Confidence Interval using Data"]:
        st.subheader("Upload Paired Data (CSV)")
        uploaded_file = st.file_uploader("CSV with two columns: Sample1, Sample2", type="csv")
        alpha = st.number_input("Significance level α", value=0.05, min_value=0.001, max_value=0.5, step=0.01)
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
                st.latex(r"SE = " + f"{se:.3f}")
                st.latex(r"t = " + f"{t_stat:.3f}")
                st.latex(r"t_{{\alpha/2,n-1}} = {t_crit:.3f}")
                st.write(f"P-Value = {p_val:.4f}")
                st.write("Decision:", "Reject H₀" if abs(t_stat) > t_crit else "Fail to reject H₀")

                if test_choice == "Paired Confidence Interval using Data":
                    ci_lower = mean_diff - t_crit*se
                    ci_upper = mean_diff + t_crit*se
                    st.latex(r"CI = \bar{d} \pm t_{\alpha/2,n-1} SE")
                    st.write(f"Confidence Interval = ({ci_lower:.3f}, {ci_upper:.3f})")

    # ------------------- PAIRED SUMMARY STATISTICS -------------------
    elif test_choice in ["Paired t-Test using Summary Statistics", "Paired Confidence Interval using Summary Statistics"]:
        st.subheader("Enter Summary Statistics")
        mean_diff = st.number_input("Mean of differences", value=0.0)
        sd_diff = st.number_input("Std Dev of differences", value=1.0)
        n = st.number_input("Sample size", min_value=2, step=1)
        alpha = st.number_input("Significance level α", value=0.05, min_value=0.001, max_value=0.5, step=0.01)
        if st.button("Calculate"):
            se = sd_diff / np.sqrt(n)
            t_stat = mean_diff / se
            t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
            p_val = 2*(1 - stats.t.cdf(abs(t_stat), df=n-1))

            st.subheader("Step-by-Step Calculation")
            st.latex(r"SE = " + f"{se:.3f}")
            st.latex(r"t = " + f"{t_stat:.3f}")
            st.latex(r"t_{{\alpha/2,n-1}} = {t_crit:.3f}")
            st.write(f"P-Value = {p_val:.4f}")
            st.write("Decision:", "Reject H₀" if abs(t_stat) > t_crit else "Fail to reject H₀")

            if test_choice == "Paired Confidence Interval using Summary Statistics":
                ci_lower = mean_diff - t_crit*se
                ci_upper = mean_diff + t_crit*se
                st.latex(r"CI = \bar{d} \pm t_{\alpha/2,n-1} SE")
                st.write(f"Confidence Interval = ({ci_lower:.3f}, {ci_upper:.3f})")

    # ------------------- INDEPENDENT T-TESTS USING DATA -------------------
    elif test_choice in ["Independent t-Test using Data", "Independent Confidence Interval using Data"]:
        st.subheader("Upload Independent Samples Data (CSV)")
        uploaded_file = st.file_uploader("CSV with two columns: Sample1, Sample2", type="csv")
        alpha = st.number_input("Significance level α", value=0.05, min_value=0.001, max_value=0.5, step=0.01)
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if "Sample1" not in df.columns or "Sample2" not in df.columns:
                st.error("CSV must have columns 'Sample1' and 'Sample2'")
            else:
                x1 = df["Sample1"]
                x2 = df["Sample2"]
                n1, n2 = len(x1), len(x2)
                mean1, mean2 = np.mean(x1), np.mean(x2)
                s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
                se = np.sqrt(s1**2/n1 + s2**2/n2)
                t_stat = (mean1 - mean2)/se
                df_deg = ((s1**2/n1 + s2**2/n2)**2) / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
                t_crit = stats.t.ppf(1-alpha/2, df=df_deg)
                p_val = 2*(1 - stats.t.cdf(abs(t_stat), df=df_deg))

                st.subheader("Step-by-Step Calculation")
                st.latex(r"\bar{x}_1 = " + f"{mean1:.3f}")
                st.latex(r"\bar{x}_2 = " + f"{mean2:.3f}")
                st.latex(r"s_1 = " + f"{s1:.3f}")
                st.latex(r"s_2 = " + f"{s2:.3f}")
                st.latex(r"SE = " + f"{se:.3f}")
                st.latex(r"t = " + f"{t_stat:.3f}")
                st.latex(r"t_{{\alpha/2,df}} = {t_crit:.3f}")
                st.write(f"P-Value = {p_val:.4f}")
                st.write("Decision:", "Reject H₀" if abs(t_stat) > t_crit else "Fail to reject H₀")

                if test_choice == "Independent Confidence Interval using Data":
                    ci_lower = (mean1 - mean2) - t_crit*se
                    ci_upper = (mean1 - mean2) + t_crit*se
                    st.latex(r"CI = (\bar{x}_1 - \bar{x}_2) \pm t_{\alpha/2,df} SE")
                    st.write(f"Confidence Interval = ({ci_lower:.3f}, {ci_upper:.3f})")

    # ------------------- INDEPENDENT T-TESTS SUMMARY -------------------
    elif test_choice in ["Independent t-Test using Summary Statistics", "Independent Confidence Interval using Summary Statistics"]:
        st.subheader("Enter Summary Statistics")
        mean1 = st.number_input("Mean of Sample 1", value=0.0)
        s1 = st.number_input("Std Dev of Sample 1", value=1.0)
        n1 = st.number_input("Sample size 1", min_value=2, step=1)
        mean2 = st.number_input("Mean of Sample 2", value=0.0)
        s2 = st.number_input("Std Dev of Sample 2", value=1.0)
        n2 = st.number_input("Sample size 2", min_value=2, step=1)
        alpha = st.number_input("Significance level α", value=0.05, min_value=0.001, max_value=0.5, step=0.01)
        if st.button("Calculate"):
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            t_stat = (mean1 - mean2)/se
            df_deg = ((s1**2/n1 + s2**2/n2)**2) / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
            t_crit = stats.t.ppf(1-alpha/2, df=df_deg)
            p_val = 2*(1 - stats.t.cdf(abs(t_stat), df=df_deg))

            st.subheader("Step-by-Step Calculation")
            st.latex(r"SE = " + f"{se:.3f}")
            st.latex(r"t = " + f"{t_stat:.3f}")
            st.latex(r"t_{{\alpha/2,df}} = {t_crit:.3f}")
            st.write(f"P-Value = {p_val:.4f}")
            st.write("Decision:", "Reject H₀" if abs(t_stat) > t_crit else "Fail to reject H₀")

            if "Confidence Interval" in test_choice:
                ci_lower = (mean1 - mean2) - t_crit*se
                ci_upper = (mean1 - mean2) + t_crit*se
                st.latex(r"CI = (\bar{x}_1 - \bar{x}_2) \pm t_{\alpha/2,df} SE")
                st.write(f"Confidence Interval = ({ci_lower:.3f}, {ci_upper:.3f})")

    # ------------------- F-TESTS -------------------
    elif test_choice in ["F-Test for Standard Deviation using Data", "F-Test for Standard Deviation using Summary Statistics"]:
        st.subheader("Enter Data or Summary Statistics")
        use_data = test_choice.endswith("Data")
        if use_data:
            uploaded_file = st.file_uploader("CSV with two columns: Sample1, Sample2", type="csv")
        else:
            n1 = st.number_input("Sample size 1", min_value=2, step=1)
            s1 = st.number_input("Std Dev of Sample 1", value=1.0)
            n2 = st.number_input("Sample size 2", min_value=2, step=1)
            s2 = st.number_input("Std Dev of Sample 2", value=1.0)
        alpha = st.number_input("Significance level α", value=0.05, min_value=0.001, max_value=0.5, step=0.01)
        if st.button("Calculate"):
            if use_data and uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                x1, x2 = df["Sample1"], df["Sample2"]
                n1, n2 = len(x1), len(x2)
                s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
            F = s1**2 / s2**2
            df1, df2 = n1-1, n2-1
            F_crit_low = stats.f.ppf(alpha/2, df1, df2)
            F_crit_high = stats.f.ppf(1-alpha/2, df1, df2)
            p_val = 2*min(stats.f.cdf(F, df1, df2), 1 - stats.f.cdf(F, df1, df2))

            st.subheader("Step-by-Step Calculation")
            st.latex(r"F = s_1^2 / s_2^2 = " + f"{F:.3f}")
            st.latex(r"df_1 = {df1}, df_2 = {df2}")
            st.latex(r"F_{{\alpha/2,df_1,df_2}} = {F_crit_low:.3f}, F_{{1-\alpha/2,df_1,df_2}} = {F_crit_high:.3f}")
            st.write(f"P-Value = {p_val:.4f}")
            st.write("Decision:", "Reject H₀" if F < F_crit_low or F > F_crit_high else "Fail to reject H₀")

if __name__ == "__main__":
    run_two_sample_tool()
