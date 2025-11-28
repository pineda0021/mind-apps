# ==========================================================
# chi_square_tests_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# Updated for Dark/Light Mode + Example Preload
# ==========================================================

import streamlit as st
import numpy as np
from scipy.stats import chi2
import pandas as pd

# ==========================================================
# Helper Functions
# ==========================================================
def round_value(value, decimals=4):
    try:
        return round(float(value), decimals)
    except:
        return value


def parse_matrix(input_text):
    """Parse text input into numeric matrix."""
    try:
        lines = input_text.strip().split("\n")
        matrix = [[float(x) for x in line.replace(",", " ").split()] for line in lines]
        row_lengths = [len(row) for row in matrix]
    except:
        raise ValueError("Matrix parsing error. Check commas/spaces/newlines.")

    if len(set(row_lengths)) != 1:
        raise ValueError("Each row must have the same number of columns.")

    return np.array(matrix)


# ==========================================================
# Universal Step Box (Dark/Light Safe)
# ==========================================================
def step_box(text):
    st.markdown(
        f"""
        <div style="
            background-color:rgba(255,255,255,0.08);
            padding:12px;
            border-radius:10px;
            border-left:5px solid #4aa3ff;
            margin-bottom:12px;
            color:inherit;
        ">
            <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# Universal Decision Box (Matches One-Sample Style)
# ==========================================================
def decision_box(reject: bool):
    if reject:
        # GREEN ✅ Reject H0
        st.markdown(
            """
            <div style='display:flex; align-items:center; gap:8px;
                padding:10px; border-radius:8px;
                background-color:#c8f7c5; margin:10px 0;'>
                <span style='font-size:22px; color:#2ecc71;'>✔️</span>
                <span style='font-size:18px; color:black;'><b>Decision: Reject H₀</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # RED ❌ Do not reject H0
        st.markdown(
            """
            <div style='display:flex; align-items:center; gap:8px;
                padding:10px; border-radius:8px;
                background-color:#f7c5c5; margin:10px 0;'>
                <span style='font-size:22px; color:#e74c3c;'>❌</span>
                <span style='font-size:18px; color:black;'><b>Decision: Do not reject H₀</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================================
# Report Generator
# ==========================================================
def print_report(title, chi2_stat, p_value, crit_val, df, expected_matrix, alpha, decimals, observed=None):

    st.markdown(f"## {title}")
    st.markdown("---")

    # Hypotheses
    st.markdown("### 🧩 Hypotheses")
    if "Goodness-of-Fit" in title:
        st.latex(r"H_0: \text{Observed frequencies follow the expected distribution}")
        st.latex(r"H_a: \text{Observed frequencies differ from the expected distribution}")
    else:
        st.latex(r"H_0: \text{The variables are independent}")
        st.latex(r"H_a: \text{The variables are dependent}")

    # Step 1
    step_box("**Step 1:** Compute the Chi-Squared Test Statistic")
    st.latex(r"\chi^2 = \sum \frac{(O - E)^2}{E}")
    st.write(f"χ² = **{round_value(chi2_stat, decimals)}**")

    # Step 2
    step_box("**Step 2:** Degrees of Freedom")
    st.write(f"df = **{df}**")

    # Step 3
    step_box("**Step 3:** Critical Value & P-Value")
    st.write(f"Critical value: **{round_value(crit_val, decimals)}**")
    st.write(f"P-value: **{round_value(p_value, decimals)}**")

    # Decision
    step_box("**Step 4:** Decision Rule")
    st.markdown(f"If **p ≤ α = {alpha}**, reject H₀.")
    decision_box(reject=(p_value <= alpha))

    # Step 5
    step_box("**Step 5:** Expected Frequencies")
    st.dataframe(np.round(expected_matrix, decimals))

    # Step 6
    if observed is not None:
        step_box("**Step 6:** Observed vs Expected Comparison")
        comp = pd.DataFrame({
            "Observed (O)": observed.flatten(),
            "Expected (E)": expected_matrix.flatten(),
            "O−E": np.round(observed.flatten() - expected_matrix.flatten(), decimals),
            "(O−E)²/E": np.round(((observed - expected_matrix)**2 / expected_matrix).flatten(), decimals)
        })
        st.dataframe(comp)

    # Step 7
    step_box("**Step 7:** Interpretation")
    if p_value <= alpha:
        msg = "Evidence suggests a **significant difference** / **association**."
    else:
        msg = "There is **not enough evidence** to claim a difference or association."
    st.success(msg)


# ==========================================================
# Core Test Functions
# ==========================================================
def chi_squared_gof(observed, expected_pct, alpha, decimals):
    observed = np.array(observed)
    expected = np.array(expected_pct) * np.sum(observed)
    chi2_stat = np.sum((observed - expected)**2 / expected)
    df = len(observed) - 1
    p_val = 1 - chi2.cdf(chi2_stat, df)
    crit_val = chi2.ppf(1 - alpha, df)
    print_report("📊 Chi-Squared Goodness-of-Fit Test (Non-Uniform)",
                 chi2_stat, p_val, crit_val, df, expected, alpha, decimals, observed)


def chi_squared_uniform(observed, alpha, decimals):
    observed = np.array(observed)
    k = len(observed)
    expected = np.full(k, np.sum(observed) / k)
    chi2_stat = np.sum((observed - expected)**2 / expected)
    df = k - 1
    p_val = 1 - chi2.cdf(chi2_stat, df)
    crit_val = chi2.ppf(1 - alpha, df)
    print_report("📈 Chi-Squared Goodness-of-Fit Test (Uniform)",
                 chi2_stat, p_val, crit_val, df, expected, alpha, decimals, observed)


def chi_squared_independence(matrix, alpha, decimals):
    observed = np.array(matrix)
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    total = observed.sum()
    expected = np.outer(row_totals, col_totals) / total
    chi2_stat = np.sum((observed - expected)**2 / expected)
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    p_val = 1 - chi2.cdf(chi2_stat, df)
    crit_val = chi2.ppf(1 - alpha, df)
    print_report("🔢 Chi-Squared Test of Independence / Homogeneity",
                 chi2_stat, p_val, crit_val, df, expected, alpha, decimals, observed)


# ==========================================================
# Main App
# ==========================================================
def run():
    st.header("🧮 Chi-Squared Test Suite")

    test_choice = st.selectbox(
        "Choose a test:",
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

    alpha = st.number_input("Significance level (α)", 0.001, 0.5, 0.05)
    decimals = st.number_input("Decimal places", 1, 10, 4)

    # ------------------------------------------------------
    # GOF — NON-UNIFORM
    # ------------------------------------------------------
    if test_choice == "Goodness-of-Fit Test (with expected percentages)":
        obs = st.text_area("Observed frequencies", value="50, 30, 20")
        exp = st.text_area("Expected percentages (sum to 1.0)", value="0.5, 0.3, 0.2")

        if st.button("▶️ Run Test"):
            try:
                observed = list(map(float, obs.replace(",", " ").split()))
                expected = list(map(float, exp.replace(",", " ").split()))
                if not np.isclose(sum(expected), 1.0):
                    st.error("Expected percentages must sum to 1.0.")
                else:
                    chi_squared_gof(observed, expected, alpha, decimals)
            except Exception as e:
                st.error(str(e))

    # ------------------------------------------------------
    # GOF — UNIFORM
    # ------------------------------------------------------
    elif test_choice == "Goodness-of-Fit Test (uniform distribution)":
        obs = st.text_area("Observed frequencies", value="10, 15, 20, 15, 10")

        if st.button("▶️ Run Uniform GOF"):
            try:
                observed = list(map(float, obs.replace(",", " ").split()))
                chi_squared_uniform(observed, alpha, decimals)
            except Exception as e:
                st.error(str(e))

    # ------------------------------------------------------
    # CHI-SQUARE — INDEPENDENCE
    # ------------------------------------------------------
    elif test_choice == "Chi-Square Test of Independence / Homogeneity":
        mat = st.text_area("Enter contingency table:", 
                           value="10, 20, 30\n15, 25, 35")

        if st.button("▶️ Run Test of Independence"):
            try:
                matrix = parse_matrix(mat)
                chi_squared_independence(matrix, alpha, decimals)
            except Exception as e:
                st.error(str(e))


# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":
    run()

run_chi_square_tool = run

