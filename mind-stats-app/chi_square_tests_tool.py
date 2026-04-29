# ==========================================================
# chi_square_tests_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# Updated with 5-Step Hypothesis Testing Format
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

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
def themed_box(text):
    st.markdown(
        f"""
        <style>
            .themed-box {{
                padding: 12px;
                border-radius: 10px;
                margin-bottom: 12px;
                border-left: 5px solid #4aa3ff;
            }}
            @media (prefers-color-scheme: light) {{
                .themed-box {{
                    background-color: #e6f3ff;
                    color: black;
                }}
            }}
            @media (prefers-color-scheme: dark) {{
                .themed-box {{
                    background-color: #2b2b2b;
                    color: white;
                }}
            }}
        </style>
        <div class="themed-box"><b>{text}</b></div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# Universal Decision Box
# ==========================================================
def decision_box(reject: bool):
    if reject:
        st.markdown(
            """
            <div style='display:flex; align-items:center; gap:8px;
                padding:10px; border-radius:8px;
                background-color:#c8f7c5; margin:10px 0;'>
                <span style='font-size:22px; color:#2ecc71;'>✅</span>
                <span style='font-size:18px; color:black;'><b>Decision: Reject H₀</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
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
# Plot Helper
# ==========================================================
def plot_chi_square_rejection_region(chi2_stat, df, alpha):
    fig, ax = plt.subplots(figsize=(8, 4))

    reject_color = "#d62728"
    accept_color = "#2ca02c"
    curve_color = "black"
    stat_color = "#1f77b4"

    xmax = max(chi2.ppf(0.995, df), chi2_stat + 3)
    x = np.linspace(0.001, xmax, 700)
    y = chi2.pdf(x, df)

    crit = chi2.ppf(1 - alpha, df)

    x_accept = np.linspace(0.001, crit, 350)
    y_accept = chi2.pdf(x_accept, df)
    ax.fill_between(x_accept, y_accept, 0, color=accept_color, alpha=0.55, zorder=1)

    x_reject = np.linspace(crit, xmax, 350)
    y_reject = chi2.pdf(x_reject, df)
    ax.fill_between(x_reject, y_reject, 0, color=reject_color, alpha=0.75, zorder=2)

    ax.plot(x, y, color=curve_color, linewidth=2, zorder=3)
    ax.axvline(crit, color="black", linestyle="--", linewidth=2, zorder=4)
    ax.axvline(chi2_stat, color=stat_color, linestyle="-", linewidth=3, zorder=5)

    ax.set_title(f"Classical Method: Chi-Square Rejection Region (df={df})")
    ax.set_xlabel(r"$\chi^2$")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    st.caption("🟥 Red = Reject H₀ region   |   🟩 Green = Do not reject H₀ region")


# ==========================================================
# Report Generator
# ==========================================================
def print_report(title, chi2_stat, p_value, crit_val, df, expected_matrix, alpha, decimals, observed=None):
    reject = p_value <= alpha

    st.markdown(f"## {title}")
    st.markdown("---")

    # Step 1: Hypotheses
    themed_box("**Step 1: Hypotheses**")
    if "Goodness-of-Fit" in title:
        st.latex(r"H_0: \text{Observed frequencies follow the expected distribution}")
        st.latex(r"H_a: \text{Observed frequencies differ from the expected distribution}")
    else:
        st.latex(r"H_0: \text{The variables are independent}")
        st.latex(r"H_a: \text{The variables are dependent}")

    # Step 2: Test Statistic
    themed_box("**Step 2: Test Statistic**")
    st.latex(r"\chi^2 = \sum \frac{(O - E)^2}{E}")
    st.write(f"χ² = **{round_value(chi2_stat, decimals)}**")
    st.write(f"df = **{df}**")

    # Step 3: Classical Method
    themed_box("**Step 3: Classical Method**")
    st.write(f"Critical value: **{round_value(crit_val, decimals)}**")
    st.markdown(f"Decision rule: Reject H₀ if **χ² > {round_value(crit_val, decimals)}**.")
    st.markdown(f"Observed test statistic: **χ² = {round_value(chi2_stat, decimals)}**")
    st.markdown(f"Classical method decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**")
    plot_chi_square_rejection_region(chi2_stat, df, alpha)

    # Step 4: P-value Approach
    themed_box("**Step 4: P-value Approach**")
    st.write(f"P-value: **{round_value(p_value, decimals)}**")
    st.write(f"α = **{round_value(alpha, decimals)}**")
    st.markdown(f"P-value approach decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**")
    decision_box(reject=reject)

    # Step 5: Conclusion
    themed_box("**Step 5: Conclusion**")
    if "Goodness-of-Fit" in title:
        if reject:
            st.write("Since the p-value is less than α, we reject H₀. There is sufficient evidence that the observed frequencies do not follow the expected distribution.")
        else:
            st.write("Since the p-value is greater than or equal to α, we do not reject H₀. There is not sufficient evidence to conclude that the observed frequencies differ from the expected distribution.")
    else:
        if reject:
            st.write("Since the p-value is less than α, we reject H₀. There is sufficient evidence of an association between the variables.")
        else:
            st.write("Since the p-value is greater than or equal to α, we do not reject H₀. There is not sufficient evidence of an association between the variables.")

    # Extra instructional tables
    themed_box("**Expected Frequencies**")
    st.dataframe(pd.DataFrame(np.round(expected_matrix, decimals)))

    if observed is not None:
        themed_box("**Observed vs Expected Comparison**")
        comp = pd.DataFrame({
            "Observed (O)": observed.flatten(),
            "Expected (E)": expected_matrix.flatten(),
            "O−E": np.round(observed.flatten() - expected_matrix.flatten(), decimals),
            "(O−E)²/E": np.round(((observed - expected_matrix) ** 2 / expected_matrix).flatten(), decimals)
        })
        st.dataframe(comp)


# ==========================================================
# Core Test Functions
# ==========================================================
def chi_squared_gof(observed, expected_pct, alpha, decimals):
    observed = np.array(observed, dtype=float)
    expected = np.array(expected_pct, dtype=float) * np.sum(observed)

    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = len(observed) - 1
    p_val = 1 - chi2.cdf(chi2_stat, df)
    crit_val = chi2.ppf(1 - alpha, df)

    print_report(
        "📊 Chi-Squared Goodness-of-Fit Test (Non-Uniform)",
        chi2_stat, p_val, crit_val, df, expected, alpha, decimals, observed
    )


def chi_squared_uniform(observed, alpha, decimals):
    observed = np.array(observed, dtype=float)
    k = len(observed)
    expected = np.full(k, np.sum(observed) / k)

    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = k - 1
    p_val = 1 - chi2.cdf(chi2_stat, df)
    crit_val = chi2.ppf(1 - alpha, df)

    print_report(
        "📈 Chi-Squared Goodness-of-Fit Test (Uniform)",
        chi2_stat, p_val, crit_val, df, expected, alpha, decimals, observed
    )


def chi_squared_independence(matrix, alpha, decimals):
    observed = np.array(matrix, dtype=float)
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    total = observed.sum()

    expected = np.outer(row_totals, col_totals) / total
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    p_val = 1 - chi2.cdf(chi2_stat, df)
    crit_val = chi2.ppf(1 - alpha, df)

    print_report(
        "🔢 Chi-Squared Test of Independence / Homogeneity",
        chi2_stat, p_val, crit_val, df, expected, alpha, decimals, observed
    )


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

                if len(observed) != len(expected):
                    st.error("Observed and expected lists must have the same length.")
                    return

                if not np.isclose(sum(expected), 1.0):
                    st.error("Expected percentages must sum to 1.0.")
                    return

                if any(v <= 0 for v in observed):
                    st.error("Observed frequencies must be positive.")
                    return

                if any(v <= 0 for v in expected):
                    st.error("Expected percentages must all be positive.")
                    return

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

                if any(v <= 0 for v in observed):
                    st.error("Observed frequencies must be positive.")
                    return

                chi_squared_uniform(observed, alpha, decimals)

            except Exception as e:
                st.error(str(e))

    # ------------------------------------------------------
    # CHI-SQUARE — INDEPENDENCE
    # ------------------------------------------------------
    elif test_choice == "Chi-Square Test of Independence / Homogeneity":
        mat = st.text_area(
            "Enter contingency table:",
            value="10, 20, 30\n15, 25, 35"
        )

        if st.button("▶️ Run Test of Independence"):
            try:
                matrix = parse_matrix(mat)

                if np.any(matrix <= 0):
                    st.error("All observed counts must be positive.")
                    return

                chi_squared_independence(matrix, alpha, decimals)

            except Exception as e:
                st.error(str(e))


# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":
    run()

run_chi_square_tool = run
