import streamlit as st
import numpy as np
from scipy.stats import f

def print_group_means(groups):
    means = [np.mean(group) for group in groups]
    for i, mean in enumerate(means, 1):
        st.write(f"Group {i} Mean: {mean:.4f}")
    return means

def one_way_anova_test(groups, alpha, decimal_places):
    group_means = [np.mean(group) for group in groups]
    overall_mean = np.mean(np.concatenate(groups))

    ssb = sum(len(group) * (group_mean - overall_mean) ** 2
              for group, group_mean in zip(groups, group_means))
    ssw = sum(np.sum((group - np.mean(group)) ** 2) for group in groups)

    df_between = len(groups) - 1
    df_within = sum(len(group) for group in groups) - len(groups)

    msb = ssb / df_between
    msw = ssw / df_within

    f_statistic = msb / msw
    p_value = 1 - f.cdf(f_statistic, df_between, df_within)
    critical_value = f.ppf(1 - alpha, df_between, df_within)

    st.markdown("### One-Way ANOVA Test")
    st.markdown("---")
    st.write(f"F-statistic: {round(f_statistic, decimal_places)}")
    st.write(f"P-value: {round(p_value, decimal_places)}")
    st.write(f"Critical Value: {round(critical_value, decimal_places)}")
    st.write(f"Degrees of Freedom (Between Groups): {df_between}")
    st.write(f"Degrees of Freedom (Within Groups): {df_within}")
    st.markdown("**Group Means:**")
    print_group_means(groups)

    if p_value <= alpha:
        st.success("✅ Reject Null Hypothesis: There is a significant difference between the groups.")
    else:
        st.info("❌ Do Not Reject Null Hypothesis: There is no significant difference between the groups.")

def parse_groups(input_text):
    try:
        groups = [list(map(float, group.strip().split(',')))
                  for group in input_text.strip().split(';')]
        return groups
    except Exception as e:
        st.error(f"Error parsing groups: {e}")
        return None

def run_anova_tool():
    st.header("📊 One-Way ANOVA")

    input_method = st.radio(
        "Choose data input method:",
        [
            "Enter all group data in one line (semicolon-separated)",
            "Enter number of groups and input each group separately"
        ]
    )

    groups = []
    if input_method == "Enter all group data in one line (semicolon-separated)":
        input_text = st.text_area("Example: 12,15,18; 25,28,29; 10,12,14")
        if input_text:
            groups = parse_groups(input_text)

    elif input_method == "Enter number of groups and input each group separately":
        num_groups = st.number_input("Number of groups", min_value=2, step=1)
        for i in range(num_groups):
            group_text = st.text_input(f"Group {i+1} data (comma-separated)")
            if group_text:
                groups.append(list(map(float, group_text.strip().split(','))))

    alpha = st.number_input("Significance level α", min_value=0.001, max_value=0.5, value=0.05)
    decimal_places = st.number_input("Decimal places for rounding", min_value=1, max_value=10, value=4, step=1)

    if st.button("👨‍💻 Calculate ANOVA") and groups:
        one_way_anova_test(groups, alpha, decimal_places)

if __name__ == "__main__":
    run_anova_tool()

