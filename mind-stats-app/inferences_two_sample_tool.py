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

    alpha = st.number_input("Significance level α", value=0.05, min_value=0.001, max_value=0.5, step=0.01)

    # ------------------- TWO-PROPORTION TESTS -------------------
    if test_choice in ["Two-Proportion Z-Test", "Confidence Interval for Proportion Difference"]:
        st.subheader("Enter Sample Data")
        x1 = st.number_input("Number of successes in Sample 1", min_value=0, step=1)
        n1 = st.number_input("Sample size 1", min_value=1, step=1)
        x2 = st.number_input("Number of successes in Sample 2", min_value=0, step=1)
        n2 = st.number_input("Sample size 2", min_value=1, step=1)

        if st.button("👨‍💻 Calculate"):
            p1 = x1 / n1
            p2 = x2 / n2
            p_pool = (x1 + x2) / (n1 + n2)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            z_stat = (p1 - p2)/se
            p_val = 2*(1 - stats.norm.cdf(abs(z_stat)))
            z_crit = stats.norm.ppf(1 - alpha/2)
            decision = "Reject the null hypothesis." if abs(z_stat) > z_crit else "Fail to reject the null hypothesis."

            report = f"""
=====================
{test_choice}
=====================
phat_1 = {p1:.4f}
phat_2 = {p2:.4f}
phat (pooled) = {p_pool:.4f}
Z = {z_stat:.4f}
Critical Value = ±{z_crit:.2f}
P-value = {p_val:.4f}
{decision}
"""
            st.text(report)

            if test_choice == "Confidence Interval for Proportion Difference":
                ci_lower = (p1 - p2) - z_crit * np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
                ci_upper = (p1 - p2) + z_crit * np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
                st.text(f"Confidence Interval = ({ci_lower:.4f}, {ci_upper:.4f})")

    # ------------------- PAIRED T-TESTS USING DATA -------------------
    elif test_choice in ["Paired t-Test using Data", "Paired Confidence Interval using Data"]:
        st.subheader("Enter Paired Data")
        st.write("Option 1: Upload CSV with two columns: Sample1, Sample2")
        uploaded_file = st.file_uploader("CSV", type="csv")

        st.write("Option 2: Enter data manually (comma-separated)")
        sample1_input = st.text_area("Sample1", placeholder="1.2, 2.3, 3.1, 4.5")
        sample2_input = st.text_area("Sample2", placeholder="0.9, 2.1, 3.0, 4.2")

        if st.button("👨‍💻 Calculate"):
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                if "Sample1" not in df.columns or "Sample2" not in df.columns:
                    st.error("CSV must have columns 'Sample1' and 'Sample2'")
                    return
                x1 = df["Sample1"].values
                x2 = df["Sample2"].values
            else:
                try:
                    x1 = np.array([float(i.strip()) for i in sample1_input.split(",")])
                    x2 = np.array([float(i.strip()) for i in sample2_input.split(",")])
                except:
                    st.error("Invalid manual data. Make sure values are numeric and comma-separated.")
                    return

            diff = x1 - x2
            mean_diff = np.mean(diff)
            sd_diff = np.std(diff, ddof=1)
            n = len(diff)
            se = sd_diff / np.sqrt(n)
            t_stat = mean_diff / se
            t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
            p_val = 2*(1 - stats.t.cdf(abs(t_stat), df=n-1))
            decision = "Reject the null hypothesis." if abs(t_stat) > t_crit else "Fail to reject the null hypothesis."

            report = f"""
=====================
{test_choice}
=====================
Mean difference = {mean_diff:.4f}
Std Dev of differences = {sd_diff:.4f}
SE = {se:.4f}
t = {t_stat:.4f}
Critical Value = ±{t_crit:.4f}
P-value = {p_val:.4f}
{decision}
"""
            st.text(report)

            if test_choice == "Paired Confidence Interval using Data":
                ci_lower = mean_diff - t_crit*se
                ci_upper = mean_diff + t_crit*se
                st.text(f"Confidence Interval = ({ci_lower:.4f}, {ci_upper:.4f})")

    # ------------------- PAIRED SUMMARY STATISTICS -------------------
    elif test_choice in ["Paired t-Test using Summary Statistics", "Paired Confidence Interval using Summary Statistics"]:
        st.subheader("Enter Summary Statistics")
        mean_diff = st.number_input("Mean of differences", value=0.0)
        sd_diff = st.number_input("Std Dev of differences", value=1.0)
        n = st.number_input("Sample size", min_value=2, step=1)

        if st.button("👨‍💻 Calculate"):
            se = sd_diff / np.sqrt(n)
            t_stat = mean_diff / se
            t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
            p_val = 2*(1 - stats.t.cdf(abs(t_stat), df=n-1))
            decision = "Reject the null hypothesis." if abs(t_stat) > t_crit else "Fail to reject the null hypothesis."

            report = f"""
=====================
{test_choice}
=====================
Mean difference = {mean_diff:.4f}
Std Dev of differences = {sd_diff:.4f}
SE = {se:.4f}
t = {t_stat:.4f}
Critical Value = ±{t_crit:.4f}
P-value = {p_val:.4f}
{decision}
"""
            st.text(report)

            if test_choice == "Paired Confidence Interval using Summary Statistics":
                ci_lower = mean_diff - t_crit*se
                ci_upper = mean_diff + t_crit*se
                st.text(f"Confidence Interval = ({ci_lower:.4f}, {ci_upper:.4f})")

    # ------------------- INDEPENDENT T-TESTS USING DATA -------------------
    elif test_choice in ["Independent t-Test using Data", "Independent Confidence Interval using Data"]:
        st.subheader("Enter Independent Samples Data")
        st.write("Option 1: Upload CSV with two columns: Sample1, Sample2")
        uploaded_file = st.file_uploader("CSV", type="csv", key="indep_csv")

        st.write("Option 2: Enter data manually (comma-separated)")
        sample1_input = st.text_area("Sample1", placeholder="1.2, 2.3, 3.1, 4.5", key="indep1")
        sample2_input = st.text_area("Sample2", placeholder="0.9, 2.1, 3.0, 4.2", key="indep2")

        if st.button("👨‍💻 Calculate", key="indep_calc"):
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                if "Sample1" not in df.columns or "Sample2" not in df.columns:
                    st.error("CSV must have columns 'Sample1' and 'Sample2'")
                    return
                x1 = df["Sample1"].values
                x2 = df["Sample2"].values
            else:
                try:
                    x1 = np.array([float(i.strip()) for i in sample1_input.split(",")])
                    x2 = np.array([float(i.strip()) for i in sample2_input.split(",")])
                except:
                    st.error("Invalid manual data. Make sure values are numeric and comma-separated.")
                    return

            n1, n2 = len(x1), len(x2)
            mean1, mean2 = np.mean(x1), np.mean(x2)
            s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            t_stat = (mean1 - mean2)/se
            df_deg = ((s1**2/n1 + s2**2/n2)**2) / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
            t_crit = stats.t.ppf(1-alpha/2, df=df_deg)
            p_val = 2*(1 - stats.t.cdf(abs(t_stat), df=df_deg))
            decision = "Reject the null hypothesis." if abs(t_stat) > t_crit else "Fail to reject the null hypothesis."

            report = f"""
=====================
{test_choice}
=====================
Mean Sample1 = {mean1:.4f}
Mean Sample2 = {mean2:.4f}
Std Dev Sample1 = {s1:.4f}
Std Dev Sample2 = {s2:.4f}
SE = {se:.4f}
t = {t_stat:.4f}
Critical Value = ±{t_crit:.4f}
P-value = {p_val:.4f}
{decision}
"""
            st.text(report)

            if "Confidence Interval" in test_choice:
                ci_lower = (mean1 - mean2) - t_crit*se
                ci_upper = (mean1 - mean2) + t_crit*se
                st.text(f"Confidence Interval = ({ci_lower:.4f}, {ci_upper:.4f})")

    # ------------------- INDEPENDENT SUMMARY STATISTICS -------------------
    elif test_choice in ["Independent t-Test using Summary Statistics", "Independent Confidence Interval using Summary Statistics"]:
        st.subheader("Enter Summary Statistics")
        mean1 = st.number_input("Mean of Sample 1", value=0.0)
        s1 = st.number_input("Std Dev of Sample 1", value=1.0)
        n1 = st.number_input("Sample size 1", min_value=2, step=1)
        mean2 = st.number_input("Mean of Sample 2", value=0.0)
        s2 = st.number_input("Std Dev of Sample 2", value=1.0)
        n2 = st.number_input("Sample size 2", min_value=2, step=1)

        if st.button("👨‍💻 Calculate", key="indep_sum"):
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            t_stat = (mean1 - mean2)/se
            df_deg = ((s1**2/n1 + s2**2/n2)**2) / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
            t_crit = stats.t.ppf(1-alpha/2, df=df_deg)
            p_val = 2*(1 - stats.t.cdf(abs(t_stat), df=df_deg))
            decision = "Reject the null hypothesis." if abs(t_stat) > t_crit else "Fail to reject the null hypothesis."

            report = f"""
=====================
{test_choice}
=====================
Mean Sample1 = {mean1:.4f}
Mean Sample2 = {mean2:.4f}
Std Dev Sample1 = {s1:.4f}
Std Dev Sample2 = {s2:.4f}
SE = {se:.4f}
t = {t_stat:.4f}
Critical Value = ±{t_crit:.4f}
P-value = {p_val:.4f}
{decision}
"""
            st.text(report)

            if "Confidence Interval" in test_choice:
                ci_lower = (mean1 - mean2) - t_crit*se
                ci_upper = (mean1 - mean2) + t_crit*se
                st.text(f"Confidence Interval = ({ci_lower:.4f}, {ci_upper:.4f})")

    # ------------------- F-TESTS -------------------
    elif test_choice in ["F-Test for Standard Deviation using Data", "F-Test for Standard Deviation using Summary Statistics"]:
        st.subheader("F-Test Input")
        use_data = test_choice.endswith("Data")
        if use_data:
            st.write("Option 1: Upload CSV with two columns: Sample1, Sample2")
            uploaded_file = st.file_uploader("CSV", type="csv", key="f_csv")

            st.write("Option 2: Enter data manually (comma-separated)")
            sample1_input = st.text_area("Sample1", placeholder="1.2, 2.3, 3.1, 4.5", key="f1")
            sample2_input = st.text_area("Sample2", placeholder="0.9, 2.1, 3.0, 4.2", key="f2")

        else:
            n1 = st.number_input("Sample size 1", min_value=2, step=1)
            s1 = st.number_input("Std Dev of Sample 1", value=1.0)
            n2 = st.number_input("Sample size 2", min_value=2, step=1)
            s2 = st.number_input("Std Dev of Sample 2", value=1.0)

        if st.button("👨‍💻 Calculate", key="f_calc"):
            if use_data:
                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    x1 = df["Sample1"].values
                    x2 = df["Sample2"].values
                else:
                    try:
                        x1 = np.array([float(i.strip()) for i in sample1_input.split(",")])
                        x2 = np.array([float(i.strip()) for i in sample2_input.split(",")])
                    except:
                        st.error("Invalid manual data. Make sure values are numeric and comma-separated.")
                        return
                n1, n2 = len(x1), len(x2)
                s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)

            F = s1**2 / s2**2
            df1, df2 = n1-1, n2-1
            F_crit_low = stats.f.ppf(alpha/2, df1, df2)
            F_crit_high = stats.f.ppf(1-alpha/2, df1, df2)
            p_val = 2*min(stats.f.cdf(F, df1, df2), 1 - stats.f.cdf(F, df1, df2))
            decision = "Reject the null hypothesis." if F < F_crit_low or F > F_crit_high else "Fail to reject the null hypothesis."

            report = f"""
=====================
{test_choice}
=====================
F = {F:.4f}
df1 = {df1}, df2 = {df2}
Critical Values = ({F_crit_low:.4f}, {F_crit_high:.4f})
P-value = {p_val:.4f}
{decision}
"""
            st.text(report)


if __name__ == "__main__":
    run_two_sample_tool()
