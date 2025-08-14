import streamlit as st
import numpy as np
from scipy.stats import chi2

def print_report(title, chi2_stat, p_value, critical_value, df, expected_matrix, decimal_places):
    st.markdown(f"### {title}")
    st.markdown("---")
    st.write(f"**Chi-Squared Statistic:** {round(chi2_stat, decimal_places)}")
    st.write(f"**P-value:** {round(p_value, decimal_places)}")
    st.write(f"**Critical Value:** {round(critical_value, decimal_places)}")
    st.write(f"**Degrees of Freedom:** {df}")
    st.write("**Expected Frequencies:**")
    st.write(np.round(expected_matrix, decimal_places))
    reject = p_value <= st.session_state.alpha
    st.write(f"**Conclusion:** {'Reject' if reject else 'Do Not Reject'} the null hypothesis")

def chi_squared_gof(observed, expected_perc, alpha, decimal_places):
    total = sum(observed)
    expected = np.array(expected_perc) * total
    chi2_stat = np.sum((np.array(observed) - expected)**2 / expected)
    df = len(observed) - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)
    critical_value = chi2.ppf(1 - alpha, df)
    print_report("Chi-Squared Goodness-of-Fit Test (Non-Uniform)", chi2_stat, p_value, critical_value, df, expected, decimal_places)

def chi_squared_uniform(observed, alpha, decimal_places):
    k = len(observed)
    total = sum(observed)
    expected = np.full(k, total / k)
    chi2_stat = np.sum((np.array(observed) - expected)**2 / expected)
    df = k - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)
    critical_value = chi2.ppf(1 - alpha, df)
    print_report("Chi-Squared Goodness-of-Fit Test (Uniform)", chi2_stat, p_value, critical_value, df, expected, decimal_places)

def chi_squared_independence(matrix, alpha, decimal_places):
    observed = np.array(matrix)
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    total = observed.sum()
    expected = np.outer(row_totals, col_totals) / total
    chi2_stat = np.sum((observed - expected)**2 / expected)
    df = (observed.shape[0]-1)*(observed.shape[1]-1)
    p_value = 1 - chi2.cdf(chi2_stat, df)
    critical_value = chi2.ppf(1 - alpha, df)
    print_report("Chi-Squared Test of Independence / Homogeneity", chi2_stat, p_value, critical_value, df, expected, decimal_places)

def parse_matrix(input_text):
    # Split by lines
    lines = input_text.strip().split("\n")
    matrix = []
    for line in lines:
        # Split by spaces or commas
        row = [float(x) for x in line.replace(",", " ").split()]
        matrix.append(row)
    return matrix

def run_chi_square_tool():
    st.header("👨‍💻 Chi-Squared Tests")

    test_choice = st.selectbox(
        "Select a Chi-Squared Test:",
        [
            "Goodness-of-Fit Test (with expected percentages)",
            "Goodness-of-Fit Test (uniform distribution)",
            "Chi-Square Test of Independence / Homogeneity"
        ]
    )

    st.session_state.alpha = st.number_input("Significance level α", min_value=0.001, max_value=0.5, value=0.05)
    decimal_places = st.number_input("Decimal places for rounding", min_value=1, max_value=10, value=4, step=1)

    if test_choice == "Goodness-of-Fit Test (with expected percentages)":
        observed_input = st.text_area("Enter observed frequencies (space-separated, e.g., 50 30 20)")
        expected_input = st.text_area("Enter expected percentages (sum=1, e.g., 0.5 0.3 0.2)")

        if st.button("👨‍💻 Calculate GOF (Non-Uniform)"):
            try:
                observed = list(map(float, observed_input.replace(",", " ").split()))
                expected_perc = list(map(float, expected_input.replace(",", " ").split()))
                chi_squared_gof(observed, expected_perc, st.session_state.alpha, decimal_places)
            except Exception as e:
                st.error(f"Error: {e}")

    elif test_choice == "Goodness-of-Fit Test (uniform distribution)":
        observed_input = st.text_area("Enter observed frequencies (space-separated, e.g., 10 15 20 5)")

        if st.button("👨‍💻 Calculate GOF (Uniform)"):
            try:
                observed = list(map(float, observed_input.replace(",", " ").split()))
                chi_squared_uniform(observed, st.session_state.alpha, decimal_places)
            except Exception as e:
                st.error(f"Error: {e}")

    elif test_choice == "Chi-Square Test of Independence / Homogeneity":
        st.markdown("Enter observed frequency matrix (rows separated by newline, values by space or comma):")
        matrix_input = st.text_area("Example:\n10 20 30\n15 25 35")

        if st.button("👨‍💻 Calculate Independence Test"):
            try:
                matrix = parse_matrix(matrix_input)
                chi_squared_independence(matrix, st.session_state.alpha, decimal_places)
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    run_chi_square_tool()

