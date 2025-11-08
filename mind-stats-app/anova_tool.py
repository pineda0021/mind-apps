# ==========================================================
# anova_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import f

# ==========================================================
# Helper Functions
# ==========================================================
def step_box(text):
    """Stylized step box for explanations."""
    st.markdown(
        f"""
        <div style="background-color:#f0f6ff;padding:10px;border-radius:10px;
        border-left:5px solid #007acc;margin-bottom:10px;">
        <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


def parse_groups(input_text):
    """Parse semicolon-separated groups like: 12,15,18; 25,28,29; 10,12,14"""
    try:
        groups = [list(map(float, group.strip().replace(" ", "").split(",")))
                  for group in input_text.strip().split(";")]
        return [g for g in groups if len(g) > 0]
    except Exception as e:
        st.error(f"Error parsing groups: {e}")
        return None


# ==========================================================
# One-Way ANOVA Calculation
# ==========================================================
def one_way_anova(groups, alpha, decimals):
    """Perform One-Way ANOVA and display results."""
    st.markdown("## 📊 One-Way ANOVA Test")
    st.markdown("---")

    # Step 0: Hypotheses
    st.markdown("### 🧩 Hypotheses")
    st.latex(r"H_0: \mu_1 = \mu_2 = \dots = \mu_k")
    st.latex(r"H_a: \text{At least one population mean differs}")

    # Step 1: Compute group statistics
    step_box("**Step 1:** Compute Group Statistics")
    group_means = [np.mean(g) for g in groups]
    group_vars = [np.var(g, ddof=1) for g in groups]
    group_sizes = [len(g) for g in groups]
    overall_mean = np.mean(np.concatenate(groups))

    summary_df = pd.DataFrame({
        "Group": [f"Group {i+1}" for i in range(len(groups))],
        "n": group_sizes,
        "Mean": np.round(group_means, decimals),
        "Variance": np.round(group_vars, decimals)
    })
    st.dataframe(summary_df)

    # Step 2: Calculate SSB, SSW, MSB, MSW
    step_box("**Step 2:** Compute Sums of Squares")
    ssb = sum(n * (m - overall_mean) ** 2 for n, m in zip(group_sizes, group_means))
    ssw = sum(sum((x - m) ** 2 for x in g) for g, m in zip(groups, group_means))
    df_between = len(groups) - 1
    df_within = sum(group_sizes) - len(groups)
    msb = ssb / df_between
    msw = ssw / df_within

    # Step 3: Compute F statistic
    step_box("**Step 3:** Compute F Statistic")
    f_stat = msb / msw
    p_value = 1 - f.cdf(f_stat, df_between, df_within)
    critical_value = f.ppf(1 - alpha, df_between, df_within)

    # Step 4: ANOVA table
    step_box("**Step 4:** Construct the ANOVA Table")
    anova_df = pd.DataFrame({
        "Source": ["Between Groups", "Within Groups", "Total"],
        "SS": [round(ssb, decimals), round(ssw, decimals), round(ssb + ssw, decimals)],
        "df": [df_between, df_within, df_between + df_within],
        "MS": [round(msb, decimals), round(msw, decimals), ""],
        "F": [round(f_stat, decimals), "", ""]
    })
    st.dataframe(anova_df)

    # Step 5: Decision and Interpretation
    step_box("**Step 5:** Decision and Interpretation")
    st.write(f"**F-statistic:** {round(f_stat, decimals)}")
    st.write(f"**P-value:** {round(p_value, decimals)}")
    st.write(f"**Critical Value (F₍₁₋α, df₁, df₂₎):** {round(critical_value, decimals)}")
    st.write(f"**Degrees of Freedom:** df₁ = {df_between}, df₂ = {df_within}")

    if p_value <= alpha:
        st.success("✅ Reject H₀: There is a significant difference among group means.")
    else:
        st.info("❌ Fail to Reject H₀: No significant difference among group means.")

    step_box("**Step 6:** Summary Interpretation")
    interpretation = (
        "Since the p-value is "
        + ("less" if p_value <= alpha else "greater")
        + f" than α = {alpha}, "
        + ("we reject" if p_value <= alpha else "we fail to reject")
        + " the null hypothesis. "
        + ("At least one mean differs significantly." if p_value <= alpha else "We conclude the group means are not significantly different.")
    )
    st.write(interpretation)


# ==========================================================
# Streamlit Interface
# ==========================================================
def run():
    st.header("📊 One-Way ANOVA Test")

    st.markdown("""
    This tool tests whether **three or more group means are equal** using the F-test.
    ---
    **Reminder:**  
    - Separate values within each group by commas.  
    - Separate groups by semicolons (;).  
    - Example: `12,15,18; 25,28,29; 10,12,14`
    """)

    input_method = st.radio(
        "Choose data input method:",
        [
            "Enter all group data in one line (semicolon-separated)",
            "Enter number of groups and input each group separately"
        ]
    )

    groups = []
    if input_method == "Enter all group data in one line (semicolon-separated)":
        input_text = st.text_area("Enter group data:", placeholder="12,15,18; 25,28,29; 10,12,14")
        if input_text:
            groups = parse_groups(input_text)

    elif input_method == "Enter number of groups and input each group separately":
        num_groups = st.number_input("Number of groups", min_value=2, step=1)
        for i in range(num_groups):
            group_text = st.text_input(f"Group {i+1} data (comma-separated)", key=f"group_{i}")
            if group_text:
                try:
                    groups.append(list(map(float, group_text.strip().split(','))))
                except Exception:
                    st.error(f"⚠️ Invalid input in Group {i+1}. Use commas to separate values (e.g., 10,12,14).")

    alpha = st.number_input("Significance level (α)", min_value=0.001, max_value=0.5, value=0.05)
    decimals = st.number_input("Decimal places for rounding", 1, 10, 4)

    if st.button("▶️ Run ANOVA Test"):
        if not groups or any(len(g) < 2 for g in groups):
            st.error("⚠️ Please enter at least two groups, each with two or more values.")
        else:
            one_way_anova(groups, alpha, decimals)


# ==========================================================
# Run Script
# ==========================================================
if __name__ == "__main__":
    run()

# ✅ Compatibility for main app
run_anova_tool = run

