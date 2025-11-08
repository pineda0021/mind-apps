# ==========================================================
# chi_square_tests_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
from scipy.stats import chi2
import pandas as pd

# ==========================================================
# Helper Functions
# ==========================================================
def round_value(value, decimals=4):
    """Safely round floats."""
    try:
        return round(float(value), decimals)
    except Exception:
        return value


def parse_matrix(input_text):
    """Parse text input into a numeric 2D matrix."""
    try:
        lines = input_text.strip().split("\n")
        matrix = [[float(x) for x in line.replace(",", " ").split()] for line in lines]
        row_lengths = [len(row) for row in matrix]
        if len(set(row_lengths)) != 1:
            raise ValueError("All rows must have the same number of columns.")
        return np.array(matrix)
    except Exception as e:
        raise ValueError(f"Matrix parsing error: {e}. Use commas/spaces and newlines correctly.")


def step_box(text):
    """Stylized step display box."""
    st.markdown(
        f"""
        <div style="background-color:#f0f6ff;padding:10px;border-radius:10px;
        border-left:5px solid #007acc;margin-bottom:10px;">
        <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# Report Generator
# ==========================================================
def print_report(title, chi2_stat, p_value, critical_value, df, expected_matrix, alpha, decimals, observed=None):
    """Display results in a structured, pedagogical format."""
    st.markdown(f"## {title}")
    st.markdown("---")

    # -----------------------------
    st.markdown("### 🧩 Hypotheses")
    if "Goodness-of-Fit" in title:
        st.latex(r"H_0: \text{Observed frequencies follow the expected distribution}")
        st.latex(r"H_a: \text{Observed frequencies do not follow the expected distribution}")
    else:
        st.latex(r"H_0: \text{The variables are independent}")
        st.latex(r"H_a: \text{The variables are dependent (associated)}")

    # -----------------------------
    step_box("**Step 1:** Compute the Chi-Squared Test Statistic")
    st.latex(r"\chi^2 = \sum \frac{(O - E)^2}{E}")
    st.write(f"Computed value: **χ² = {round_value(chi2_stat, decimals)}**")

    step_box("**Step 2:** Determine the Degrees of Freedom")
    st.write(f"**df = {df}**")

    step_box("**Step 3:** Find the Critical Value and P-Value")
    st.write(f"Critical Value (χ²₍₁₋α₎,df): **{round_value(critical_value, decimals)}**")
    st.write(f"P-Value: **{round_value(p_value, decimals)}**")

    step_box("**Step 4:** Decision Rule")
    st.markdown(rf"If **p ≤ α = {alpha}**, reject H₀. Otherwise, fail to reject H₀.")

    reject = p_value <= alpha
    decision = "✅ **Reject the null hypothesis.**" if reject else "❌ **Do not reject the null hypothesis.**"
    st.markdown(decision)

    step_box("**Step 5:** Expected Frequencies")
    st.dataframe(np.round(expected_matrix, decimals))

    # Optional visual comparison
    if observed is not None:
        step_box("**Step 6:** Observed vs Expected Comparison")
        comparison_df = pd.DataFrame({
            "Observed (O)": observed.flatten(),
            "Expected (E)": expected_matrix.flatten(),
            "O−E": np.round(observed.flatten() - expected_matrix.flatten(), decimals),
            "(O−E)²/E": np.round(((observed - expected_matrix) ** 2 / expected_matrix).flatten(), decimals)
        })
        st.dataframe(comparison_df)

    step_box("**Step 7:** Interpretation")
    if "Goodness-of-Fit" in title:
        interpretation = (
            "The observed frequencies "
            + ("**differ significantly**" if reject else "**do not differ significantly**")
            + " from the expected distribution."
        )
    else:
        interpretation = (
            "There **is evidence of an association** between the variables."
            if reject
            else "There is **no evidence of association** between the variables (they appear independent)."
        )
    st.success(interpretation)


# ==========================================================
# Core Chi-Square Tests
# ==========================================================
def chi_squared_gof(observed, expected_perc, alpha, decimals):
    observed = np.array(observed)
    expected = np.array(expected_perc) * np.sum(observed)
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = len(observed) - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)
    critical_value = chi2.ppf(1 - alpha, df)
    print_report(
        "📊 Chi-Squared Goodness-of-Fit Test (Non-Uniform)",
        chi2_stat, p_value, critical_value, df, expected, alpha, decimals, observed
    )


def chi_squared_uniform(observed, alpha, decimals):
    observed = np.array(observed)
    k = len(observed)
    expected = np.full(k, np.sum(observed) / k)
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = k - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)
    critical_value = chi2.ppf(1 - alpha, df)
    print_report(
        "📈 Chi-Squared Goodness-of-Fit Test (Uniform)",
        chi2_stat, p_value, critical_value, df, expected, alpha, decimals, observed
    )


def chi_squared_independence(matrix, alpha, decimals):
    observed = np.array(matrix)
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    total = observed.sum()
    expected = np.outer(row_totals, col_totals) / total
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    p_value = 1 - chi2.cdf(chi2_stat, df)
    critical_value = chi2.ppf(1 - alpha, df)
    print_report(
        "🔢 Chi-Squared Test of Independence / Homogeneity",
        chi2_stat, p_value, critical_value, df, expected, alpha, decimals, observed
    )


# ==========================================================
# Streamlit App
# ==========================================================
def run():
    st.header("🧮 Chi-Squared Test Suite")

    test_choice = st.selectbox(
        "Choose a Chi-Squared Test:",
        [
            "Goodness-of-Fit Test (with expected percentages)",
            "Goodness-of-Fit Test (uniform distribution)",
            "Chi-Square Test of Independence / Homogeneity",
        ],
        index=None,
        placeholder="Select a Chi-Squared Test to begin...",
    )

    if not test_choice:
        st.info("👆 Please select a Chi-Squared test to begin.")
        return

    alpha = st.number_input("Significance level (α)", min_value=0.001, max_value=0.5, value=0.05)
    decimals = st.number_input("Decimal places for rounding", 1, 10, 4)

    st.markdown(
        """
        ⚠️ **Reminder:** Enter data separated by **spaces or commas**.  
        For matrices, use **new lines** to separate rows.
        """
    )

    # ----------------------------------------------------------
    if test_choice == "Goodness-of-Fit Test (with expected percentages)":
        st.subheader("📊 Input Data for Non-Uniform Distribution")
        st.caption("Enter observed frequencies and expected **percentages** (which must sum to 1.0).")

        obs = st.text_area("Observed frequencies", placeholder="50, 30, 20")
        exp = st.text_area("Expected percentages", placeholder="0.5, 0.3, 0.2")

        if st.button("▶️ Run GOF (Non-Uniform)"):
            try:
                observed = list(map(float, obs.replace(",", " ").split()))
                expected = list(map(float, exp.replace(",", " ").split()))
                if not np.isclose(sum(expected), 1.0):
                    st.error("Expected percentages must sum to 1.0.")
                else:
                    chi_squared_gof(observed, expected, alpha, decimals)
            except Exception as e:
                st.error(f"❌ Input Error: {e}")

    # ----------------------------------------------------------
    elif test_choice == "Goodness-of-Fit Test (uniform distribution)":
        st.subheader("📈 Input Data for Uniform Distribution")
        st.caption("Enter observed frequencies separated by spaces or commas.")
        obs = st.text_area("Observed frequencies", placeholder="10, 15, 20, 5")

        if st.button("▶️ Run GOF (Uniform)"):
            try:
                observed = list(map(float, obs.replace(",", " ").split()))
                chi_squared_uniform(observed, alpha, decimals)
            except Exception as e:
                st.error(f"❌ Input Error: {e}")

    # ----------------------------------------------------------
    elif test_choice == "Chi-Square Test of Independence / Homogeneity":
        st.subheader("🔢 Input Contingency Table Data")
        st.caption(
            "Each row represents a category or group. "
            "Separate values by **commas or spaces**, and separate rows by **newlines**."
        )

        with st.expander("📘 Example Input Guide"):
            st.markdown(
                """
                **Example 1 – 2×3 Table (Gender vs Course Preference)**  
                ```
                10, 20, 30
                15, 25, 35
                ```
                **Example 2 – 3×2 Table (School vs Pass/Fail)**  
                ```
                30, 10
                25, 15
                40, 5
                ```
                **Example 3 – 3×3 Table (Region vs Income Category)**  
                ```
                25, 35, 40
                20, 30, 50
                15, 40, 45
                ```
                ---
                ✅ **The app automatically:**
                - Splits rows by **newlines**
                - Splits values in each row by **commas or spaces**
                - Calculates all totals, expected values, χ², df, and p-value automatically
                """
            )

        mat = st.text_area("Enter your contingency table:", placeholder="10, 20, 30\n15, 25, 35")

        if st.button("▶️ Run Test of Independence"):
            try:
                matrix = parse_matrix(mat)
                chi_squared_independence(matrix, alpha, decimals)
            except Exception as e:
                st.error(f"❌ Input Error: {e}")


# ==========================================================
# Run Script (with backward compatibility)
# ==========================================================
if __name__ == "__main__":
    run()

# ✅ Backward compatibility for main app
run_chi_square_tool = run
