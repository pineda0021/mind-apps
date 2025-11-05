import streamlit as st
import numpy as np
from scipy.stats import chi2

# ---------- Report Formatter ----------
def print_report(title, chi2_stat, p_value, critical_value, df, expected_matrix, decimal_places):
    st.markdown(f"### {title}")
    st.markdown("---")
    st.write(f"**Chi-Squared Statistic (χ²):** {round(chi2_stat, decimal_places)}")
    st.write(f"**P-value:** {round(p_value, decimal_places)}")
    st.write(f"**Critical Value:** {round(critical_value, decimal_places)}")
    st.write(f"**Degrees of Freedom (df):** {df}")
    st.write("**Expected Frequencies:**")
    st.write(np.round(expected_matrix, decimal_places))

    reject = p_value <= st.session_state.alpha
    decision = "✅ Reject the null hypothesis" if reject else "❌ Do not reject the null hypothesis"
    st.markdown(f"**Conclusion:** {decision}")

# ---------- Chi-Square Tests ----------
def chi_squared_gof(observed, expected_perc, alpha, decimal_places):
    total = sum(observed)
    expected = np.array(expected_perc) * total
    chi2_stat = np.sum((np.array(observed) - expected) ** 2 / expected)
    df = len(observed) - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)
    critical_value = chi2.ppf(1 - alpha, df)
    print_report("📊 Chi-Squared Goodness-of-Fit Test (Non-Uniform)", chi2_stat, p_value, critical_value, df, expected, decimal_places)

def chi_squared_uniform(observed, alpha, decimal_places):
    k = len(observed)
    total = sum(observed)
    expected = np.full(k, total / k)
    chi2_stat = np.sum((np.array(observed) - expected) ** 2 / expected)
    df = k - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)
    critical_value = chi2.ppf(1 - alpha, df)
    print_report("📈 Chi-Squared Goodness-of-Fit Test (Uniform)", chi2_stat, p_value, critical_value, df, expected, decimal_places)

def chi_squared_independence(matrix, alpha, decimal_places):
    observed = np.array(matrix)
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    total = observed.sum()
    expected = np.outer(row_totals, col_totals) / total
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    p_value = 1 - chi2.cdf(chi2_stat, df)
    critical_value = chi2.ppf(1 - alpha, df)
    print_report("🔢 Chi-Squared Test of Independence / Homogeneity", chi2_stat, p_value, critical_value, df, expected, decimal_places)

# ---------- Parse Matrix ----------
def parse_matrix(input_text):
    lines = input_text.strip().split("\n")
    matrix = []
    for line in lines:
        row = [float(x) for x in line.replace(",", " ").split()]
        matrix.append(row)
    return matrix

# ---------- Main App ----------
def run_chi_square_tool():
    st.header("🧮 Chi-Squared Test Suite")

    test_choice = st.selectbox(
        "Choose a Chi-Squared Test:",
        [
            "Goodness-of-Fit Test (with expected percentages)",
            "Goodness-of-Fit Test (uniform distribution)",
            "Chi-Square Test of Independence / Homogeneity"
        ],
        index=None,
        placeholder="Select a Chi-Squared Test to begin..."
    )

    if not test_choice:
        st.info("👆 Please select a Chi-Squared test to begin.")
        return

    st.session_state.alpha = st.number_input("Significance level (α)", min_value=0.001, max_value=0.5, value=0.05)
    decimal_places = st.number_input("Decimal places for rounding", min_value=1, max_value=10, value=4, step=1)

    # ------------------- GOODNESS OF FIT: NON-UNIFORM -------------------
    if test_choice == "Goodness-of-Fit Test (with expected percentages)":
        st.subheader("📊 Input Data for Non-Uniform Distribution")
        st.markdown("Enter the observed frequencies and the expected **percentages** (which should sum to 1).")

        observed_input = st.text_area("Observed frequencies", placeholder="50 30 20")
        expected_input = st.text_area("Expected percentages", placeholder="0.5 0.3 0.2")

        if st.button("👨‍💻 Run GOF (Non-Uniform)"):
            try:
                observed = list(map(float, observed_input.replace(",", " ").split()))
                expected_perc = list(map(float, expected_input.replace(",", " ").split()))
                if not np.isclose(sum(expected_perc), 1.0):
                    st.error("Expected percentages must sum to 1.0.")
                    return
                chi_squared_gof(observed, expected_perc, st.session_state.alpha, decimal_places)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # ------------------- GOODNESS OF FIT: UNIFORM -------------------
    elif test_choice == "Goodness-of-Fit Test (uniform distribution)":
        st.subheader("📈 Input Data for Uniform Distribution")
        st.markdown("Enter the observed frequencies separated by spaces or commas:")

        observed_input = st.text_area("Observed frequencies", placeholder="10 15 20 5")

        if st.button("👨‍💻 Run GOF (Uniform)"):
            try:
                observed = list(map(float, observed_input.replace(",", " ").split()))
                chi_squared_uniform(observed, st.session_state.alpha, decimal_places)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # ------------------- TEST OF INDEPENDENCE -------------------
    elif test_choice == "Chi-Square Test of Independence / Homogeneity":
        st.subheader("🔢 Input Contingency Table Data")
        st.markdown("Enter your **observed frequency matrix**, where each row represents a group or category.\n"
                    "Separate numbers with spaces or commas, and separate rows by newlines.")

        matrix_input = st.text_area("Example:\n10 20 30\n15 25 35")

        if st.button("👨‍💻 Run Test of Independence"):
            try:
                matrix = parse_matrix(matrix_input)
                chi_squared_independence(matrix, st.session_state.alpha, decimal_places)
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ---------- Run ----------
if __name__ == "__main__":
    run_chi_square_tool()
