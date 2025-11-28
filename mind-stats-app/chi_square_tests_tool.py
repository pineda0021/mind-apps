# ==========================================================
# chi_square_tests_tool.py
# Rebuilt Version (Dark/Light Mode Safe + Presets + Clean UI)
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import chi2

# ==========================================================
# Helper Functions
# ==========================================================
def round_value(v, d=4):
    try:
        return round(float(v), d)
    except:
        return v


def parse_matrix(text):
    """Parse a multiline matrix safely."""
    try:
        lines = text.strip().split("\n")
        m = [[float(x) for x in line.replace(",", " ").split()] for line in lines]
        lens = [len(r) for r in m]
        if len(set(lens)) != 1:
            raise ValueError("All rows must have the same number of columns.")
        return np.array(m)
    except:
        raise ValueError("Matrix error: use commas/spaces, new lines for rows.")


# ==========================================================
# STYLING BLOCKS (Dark/Light Mode Safe)
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
            color:inherit;">
            <b>{text}</b>
        </div>
        """, unsafe_allow_html=True
    )


def decision_box(reject: bool):
    color = "#2ecc71" if reject else "#e74c3c"
    icon = "✔️" if reject else "✖️"
    text = "Reject H₀" if reject else "Fail to reject H₀"
    st.markdown(
        f"""
        <div style="
            padding:12px;
            border-radius:10px;
            background-color:rgba(255,255,255,0.05);
            border-left:5px solid {color};
            margin-top:10px;
            color:inherit;">
            <b style="color:{color};">{icon} {text}</b>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# REPORT GENERATOR
# ==========================================================
def generate_report(
        title, chi2_stat, p_value, crit_val, df,
        expected, alpha, decimals, observed=None):

    st.markdown(f"## {title}")
    st.markdown("---")

    # Hypotheses
    st.markdown("### 🧩 Hypotheses")
    if "Goodness-of-Fit" in title:
        st.latex(r"H_0: \text{Observed frequencies follow the expected distribution}")
        st.latex(r"H_a: \text{Observed frequencies differ from the expected distribution}")
    else:
        st.latex(r"H_0: \text{Variables are independent}")
        st.latex(r"H_a: \text{Variables are dependent}")

    # Step 1
    step_box("**Step 1:** Compute the Chi-Squared Test Statistic")
    st.latex(r"\chi^2 = \sum \frac{(O - E)^2}{E}")
    st.write(f"χ² = **{round_value(chi2_stat, decimals)}**")

    # Step 2
    step_box("**Step 2:** Degrees of Freedom")
    st.write(f"df = **{df}**")

    # Step 3
    step_box("**Step 3:** Critical Value and P-Value")
    st.write(f"Critical Value = **{round_value(crit_val, decimals)}**")
    st.write(f"P-Value = **{round_value(p_value, decimals)}**")

    # Step 4
    step_box("**Step 4:** Decision Rule")
    st.write(f"If **p ≤ α = {alpha}**, reject H₀.")

    reject = p_value <= alpha
    decision_box(reject)

    # Step 5
    step_box("**Step 5:** Expected Frequencies")
    st.dataframe(np.round(expected, decimals))

    # Step 6
    if observed is not None:
        step_box("**Step 6:** Observed vs Expected Comparison")
        comparison_df = pd.DataFrame({
            "Observed (O)": observed.flatten(),
            "Expected (E)": expected.flatten(),
            "O−E": np.round((observed - expected).flatten(), decimals),
            "(O−E)²/E": np.round(((observed - expected) ** 2 / expected).flatten(), decimals)
        })
        st.dataframe(comparison_df)

    # Step 7
    step_box("**Step 7:** Interpretation")
    if reject:
        msg = "There **is evidence** of a significant difference."
    else:
        msg = "There is **not enough evidence** to conclude a difference."

    st.success(msg)


# ==========================================================
# CORE CHI-SQUARE TESTS
# ==========================================================
def chi2_gof_nonuniform(observed, expected_pct, alpha, decimals):
    observed = np.array(observed)
    expected = np.array(expected_pct) * np.sum(observed)
    chi_stat = np.sum((observed - expected)**2 / expected)
    df = len(observed) - 1
    p = 1 - chi2.cdf(chi_stat, df)
    crit = chi2.ppf(1 - alpha, df)

    generate_report(
        "📊 Chi-Squared Goodness-of-Fit Test (Non-Uniform)",
        chi_stat, p, crit, df, expected, alpha, decimals, observed
    )


def chi2_gof_uniform(observed, alpha, decimals):
    observed = np.array(observed)
    k = len(observed)
    expected = np.full(k, np.sum(observed)/k)
    chi_stat = np.sum((observed - expected)**2 / expected)
    df = k - 1
    p = 1 - chi2.cdf(chi_stat, df)
    crit = chi2.ppf(1 - alpha, df)

    generate_report(
        "📈 Chi-Squared Goodness-of-Fit Test (Uniform)",
        chi_stat, p, crit, df, expected, alpha, decimals, observed
    )


def chi2_independence(matrix, alpha, decimals):
    observed = np.array(matrix)
    row_tot = observed.sum(axis=1)
    col_tot = observed.sum(axis=0)
    total = observed.sum()

    expected = np.outer(row_tot, col_tot) / total
    chi_stat = np.sum((observed - expected)**2 / expected)
    df = (observed.shape[0]-1)*(observed.shape[1]-1)
    p = 1 - chi2.cdf(chi_stat, df)
    crit = chi2.ppf(1 - alpha, df)

    generate_report(
        "🔢 Chi-Squared Test of Independence / Homogeneity",
        chi_stat, p, crit, df, expected, alpha, decimals, observed
    )


# ==========================================================
# MAIN APP
# ==========================================================
def run():
    st.header("🧮 Chi-Squared Test Suite")

    choice = st.selectbox(
        "Choose a Chi-Squared Test:",
        [
            "Goodness-of-Fit Test (expected percentages)",
            "Goodness-of-Fit Test (uniform distribution)",
            "Chi-Square Test of Independence / Homogeneity"
        ],
        index=None,
        placeholder="Select a test..."
    )

    if not choice:
        return

    alpha = st.number_input("Significance level (α)", 0.001, 0.5, 0.05)
    decimals = st.number_input("Decimal places for output", 1, 10, 4)

    # ------------------------------------------------------
    # GOODNESS-OF-FIT (NON-UNIFORM)
    # ------------------------------------------------------
    if choice == "Goodness-of-Fit Test (expected percentages)":
        obs = st.text_area("Observed frequencies:", "50, 30, 20")
        exp = st.text_area("Expected percentages (must sum to 1):", "0.5, 0.3, 0.2")

        if st.button("▶️ Run Test"):
            try:
                observed = list(map(float, obs.replace(",", " ").split()))
                expected = list(map(float, exp.replace(",", " ").split()))
                if not np.isclose(sum(expected), 1.0):
                    st.error("Expected percentages must sum to 1.0")
                else:
                    chi2_gof_nonuniform(observed, expected, alpha, decimals)
            except Exception as e:
                st.error(e)

    # ------------------------------------------------------
    # GOODNESS-OF-FIT (UNIFORM)
    # ------------------------------------------------------
    elif choice == "Goodness-of-Fit Test (uniform distribution)":
        obs = st.text_area("Observed frequencies:", "12, 18, 25, 15")

        if st.button("▶️ Run Uniform GOF"):
            try:
                observed = list(map(float, obs.replace(",", " ").split()))
                chi2_gof_uniform(observed, alpha, decimals)
            except Exception as e:
                st.error(e)

    # ------------------------------------------------------
    # CHI-SQUARE TEST OF INDEPENDENCE
    # ------------------------------------------------------
    elif choice == "Chi-Square Test of Independence / Homogeneity":

        preset = st.selectbox(
            "Choose example or enter your own:",
            [
                "Example A — Gender × Preference (2×3)",
                "Example B — School × Pass/Fail (3×2)",
                "enter my own data"
            ]
        )

        if preset == "Example A — Gender × Preference (2×3)":
            default = "10, 20, 30\n15, 25, 35"
        elif preset == "Example B — School × Pass/Fail (3×2)":
            default = "30, 10\n25, 15\n40, 5"
        else:
            default = ""

        mat = st.text_area("Enter contingency table:", value=default)

        if st.button("▶️ Run Independence Test"):
            try:
                matrix = parse_matrix(mat)
                chi2_independence(matrix, alpha, decimals)
            except Exception as e:
                st.error(e)


# ==========================================================
# Run app
# ==========================================================
if __name__ == "__main__":
    run()

# For integration with main MIND app
run_chi_square_tool = run

