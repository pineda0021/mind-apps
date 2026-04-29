# ==========================================================
# anova_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# Updated with 5-Step Hypothesis Testing Format
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f

# ==========================================================
# Helper Functions
# ==========================================================
def themed_box(text):
    st.markdown(
        f"""
        <style>
            .themed-box {{
                padding: 12px;
                border-radius: 10px;
                margin-bottom: 12px;
                border-left: 5px solid #007acc;
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
        unsafe_allow_html=True,
    )


def parse_groups(input_text):
    """Parse semicolon-separated groups like: 12,15,18; 25,28,29; 10,12,14"""
    try:
        groups = [
            list(map(float, group.strip().replace(" ", "").split(",")))
            for group in input_text.strip().split(";")
        ]
        return [g for g in groups if len(g) > 0]
    except Exception as e:
        st.error(f"Error parsing groups: {e}")
        return None


def load_uploaded_data():
    """Upload CSV or Excel and extract groups for ANOVA."""
    uploaded_file = st.file_uploader(
        "📂 Upload CSV or Excel file (wide or long format)",
        type=["csv", "xlsx"]
    )
    if not uploaded_file:
        return None

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### 📄 Preview of Uploaded Data")
        st.dataframe(df.head())

        lower_cols = [str(c).lower() for c in df.columns]

        if df.shape[1] == 2 and "group" in lower_cols and "value" in lower_cols:
            group_col = df.columns[lower_cols.index("group")]
            value_col = df.columns[lower_cols.index("value")]
            groups = [grp[value_col].dropna().tolist() for _, grp in df.groupby(group_col)]
            st.success("✅ Detected long format with 'Group' and 'Value' columns.")
            return groups

        elif df.shape[1] >= 2:
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if len(numeric_cols) < 2:
                st.error("⚠️ Need at least two numeric columns for wide format.")
                return None
            groups = [df[col].dropna().tolist() for col in numeric_cols]
            st.success("✅ Detected wide format (each numeric column = one group).")
            return groups

        else:
            st.error("⚠️ File must contain at least two numeric columns or a 'Group'-'Value' pair.")
            return None

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


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
# Accessibility Function
# ==========================================================
def accessibility_summary(groups, decimals):
    st.markdown("### ♿ Accessibility Summary (Text-Only Interpretation)")
    for i, g in enumerate(groups):
        st.markdown(
            f"""
**Group {i+1} Summary**
- n = {len(g)}
- Min = {np.min(g):.{decimals}f}
- Q1 = {np.percentile(g, 25):.{decimals}f}
- Median = {np.median(g):.{decimals}f}
- Q3 = {np.percentile(g, 75):.{decimals}f}
- Max = {np.max(g):.{decimals}f}
"""
        )


# ==========================================================
# Plot Helpers
# ==========================================================
def plot_horizontal_boxplots(groups):
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(groups, vert=False, patch_artist=True)

    for box in bp["boxes"]:
        box.set(facecolor="#dbeafe", edgecolor="black", linewidth=1.5)
    for whisker in bp["whiskers"]:
        whisker.set(color="black", linewidth=1.2)
    for cap in bp["caps"]:
        cap.set(color="black", linewidth=1.2)
    for median in bp["medians"]:
        median.set(color="#d62728", linewidth=2)

    ax.set_xlabel("Values")
    ax.set_yticks(range(1, len(groups) + 1))
    ax.set_yticklabels([f"Group {i+1}" for i in range(len(groups))])
    ax.set_title("Horizontal Boxplots of Groups")
    ax.grid(alpha=0.2)

    st.pyplot(fig)


def plot_f_rejection_region(f_stat, df_between, df_within, alpha):
    fig, ax = plt.subplots(figsize=(8, 4))

    reject_color = "#d62728"
    accept_color = "#2ca02c"
    curve_color = "black"
    stat_color = "#1f77b4"

    xmax = max(6, f.ppf(0.995, df_between, df_within), f_stat + 2)
    x = np.linspace(0.001, xmax, 700)
    y = f.pdf(x, df_between, df_within)

    crit = f.ppf(1 - alpha, df_between, df_within)

    x_accept = np.linspace(0.001, crit, 350)
    y_accept = f.pdf(x_accept, df_between, df_within)
    ax.fill_between(x_accept, y_accept, 0, color=accept_color, alpha=0.55, zorder=1)

    x_reject = np.linspace(crit, xmax, 350)
    y_reject = f.pdf(x_reject, df_between, df_within)
    ax.fill_between(x_reject, y_reject, 0, color=reject_color, alpha=0.75, zorder=2)

    ax.plot(x, y, color=curve_color, linewidth=2, zorder=3)
    ax.axvline(crit, color="black", linestyle="--", linewidth=2, zorder=4)
    ax.axvline(f_stat, color=stat_color, linestyle="-", linewidth=3, zorder=5)

    ax.set_title(f"Classical Method: F Rejection Region (df1={df_between}, df2={df_within})")
    ax.set_xlabel("F")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    st.caption("🟥 Red = Reject H₀ region   |   🟩 Green = Do not reject H₀ region")


# ==========================================================
# One-Way ANOVA Calculation
# ==========================================================
def one_way_anova(groups, alpha, decimals):
    st.markdown("## 📊 One-Way ANOVA Test")
    st.markdown("---")

    all_values = np.concatenate(groups)
    group_means = [np.mean(g) for g in groups]
    group_vars = [np.var(g, ddof=1) for g in groups]
    group_sizes = [len(g) for g in groups]
    overall_mean = np.mean(all_values)

    ssb = sum(n * (m - overall_mean) ** 2 for n, m in zip(group_sizes, group_means))
    ssw = sum(sum((x - m) ** 2 for x in g) for g, m in zip(groups, group_means))
    df_between = len(groups) - 1
    df_within = sum(group_sizes) - len(groups)
    msb = ssb / df_between
    msw = ssw / df_within
    f_stat = msb / msw
    p_value = 1 - f.cdf(f_stat, df_between, df_within)
    critical_value = f.ppf(1 - alpha, df_between, df_within)
    reject = p_value <= alpha

    # Step 1
    themed_box("**Step 1: Hypotheses**")
    st.latex(r"H_0: \mu_1 = \mu_2 = \dots = \mu_k")
    st.latex(r"H_a: \text{At least one population mean differs}")

    # Step 2
    themed_box("**Step 2: Test Statistic Components**")
    summary_df = pd.DataFrame({
        "Group": [f"Group {i+1}" for i in range(len(groups))],
        "n": group_sizes,
        "Mean": np.round(group_means, decimals),
        "Variance": np.round(group_vars, decimals)
    })
    st.dataframe(summary_df)

    st.latex(r"SSB = \sum n_i(\bar{x}_i - \bar{x})^2")
    st.latex(r"SSW = \sum \sum (x_{ij} - \bar{x}_i)^2")
    st.latex(r"MSB = \frac{SSB}{df_{\text{between}}}")
    st.latex(r"MSW = \frac{SSW}{df_{\text{within}}}")
    st.latex(r"F = \frac{MSB}{MSW}")

    st.write(f"Overall Mean = **{overall_mean:.{decimals}f}**")
    st.write(f"SSB = **{ssb:.{decimals}f}**")
    st.write(f"SSW = **{ssw:.{decimals}f}**")
    st.write(f"df between = **{df_between}**")
    st.write(f"df within = **{df_within}**")
    st.write(f"MSB = **{msb:.{decimals}f}**")
    st.write(f"MSW = **{msw:.{decimals}f}**")
    st.write(f"F statistic = **{f_stat:.{decimals}f}**")

    themed_box("**Step 2A: Visualize Groups (Horizontal Boxplots)**")
    plot_horizontal_boxplots(groups)
    accessibility_summary(groups, decimals)

    # Step 3
    themed_box("**Step 3: Classical Method**")
    st.write(f"Critical value = **{critical_value:.{decimals}f}**")
    st.markdown(f"Decision rule: Reject H₀ if **F > {critical_value:.{decimals}f}**.")
    st.markdown(f"Observed test statistic: **F = {f_stat:.{decimals}f}**")
    st.markdown(f"Classical method decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**")
    plot_f_rejection_region(f_stat, df_between, df_within, alpha)

    # Step 4
    themed_box("**Step 4: P-value Approach**")
    st.write(f"P-value = **{p_value:.{decimals}f}**")
    st.write(f"α = **{alpha:.{decimals}f}**")
    st.markdown(f"P-value approach decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**")
    decision_box(reject)

    # Step 5
    themed_box("**Step 5: Conclusion**")
    interpretation = (
        "Since the p-value is "
        + ("less than" if reject else "greater than or equal to")
        + f" α = {alpha}, "
        + ("we reject " if reject else "we do not reject ")
        + "H₀. "
        + (
            "There is sufficient evidence that at least one group mean differs."
            if reject
            else "There is not sufficient evidence to conclude that the group means differ."
        )
    )
    st.write(interpretation)

    # ANOVA Table
    themed_box("**ANOVA Table**")
    anova_df = pd.DataFrame({
        "Source": ["Between Groups", "Within Groups", "Total"],
        "SS": [round(ssb, decimals), round(ssw, decimals), round(ssb + ssw, decimals)],
        "df": [df_between, df_within, df_between + df_within],
        "MS": [round(msb, decimals), round(msw, decimals), ""],
        "F": [round(f_stat, decimals), "", ""]
    })
    st.dataframe(anova_df)


# ==========================================================
# Streamlit Interface
# ==========================================================
def run():
    st.header("📊 One-Way ANOVA Test (Enhanced Version)")

    st.markdown("""
This tool tests whether **three or more group means are equal** using the F-test.

---
**Input Options:**
- Manual entry
- Upload CSV/Excel
""")

    input_method = st.radio(
        "Choose data input method:",
        ["📋 Manual Entry", "📂 Upload CSV/Excel File"]
    )

    groups = []
    if input_method == "📋 Manual Entry":
        mode = st.radio(
            "Manual Entry Method:",
            [
                "Enter all group data in one line (semicolon-separated)",
                "Enter number of groups and input each group separately"
            ]
        )

        if mode == "Enter all group data in one line (semicolon-separated)":
            input_text = st.text_area(
                "Enter group data:",
                placeholder="12,15,18; 25,28,29; 10,12,14"
            )
            if input_text:
                groups = parse_groups(input_text)

        else:
            num_groups = st.number_input("Number of groups", min_value=2, step=1)
            for i in range(num_groups):
                group_text = st.text_input(f"Group {i+1} data (comma-separated)", key=f"group_{i}")
                if group_text:
                    try:
                        groups.append(list(map(float, group_text.strip().split(','))))
                    except Exception:
                        st.error(f"⚠️ Invalid input in Group {i+1}. Use commas to separate values.")

    elif input_method == "📂 Upload CSV/Excel File":
        groups = load_uploaded_data()

    alpha = st.number_input("Significance level (α)", min_value=0.001, max_value=0.5, value=0.05)
    decimals = st.number_input("Decimal places for rounding", 1, 10, 4)

    if st.button("▶️ Run ANOVA Test"):
        if not groups or len(groups) < 2:
            st.error("⚠️ Please enter at least two groups.")
            return

        if any(len(g) < 2 for g in groups):
            st.error("⚠️ Each group must contain at least two values.")
            return

        one_way_anova(groups, alpha, decimals)


# ==========================================================
# Run Script
# ==========================================================
if __name__ == "__main__":
    run()

# Compatibility for main app
run_anova_tool = run
