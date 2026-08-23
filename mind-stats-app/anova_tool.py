# ==========================================================
# anova_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# Updated with 5-Step Hypothesis Testing Format
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import f


# ==========================================================
# Helper Functions
# ==========================================================
def themed_box(text):

    # Remove Markdown ** symbols inside HTML box
    text = text.replace("**", "")

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

<div class="themed-box">
<b>{text}</b>
</div>
        """,
        unsafe_allow_html=True,
    )


def parse_groups(input_text):
    """Parse semicolon-separated groups like: 12,15,18; 25,28,29; 10,12,14"""

    try:
        groups = [
            list(
                map(
                    float,
                    group.strip().replace(" ", "").split(",")
                )
            )
            for group in input_text.strip().split(";")
        ]

        return [
            g for g in groups
            if len(g) > 0
        ]

    except Exception as e:

        st.error(
            f"Error parsing groups: {e}"
        )

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

        st.write(
            "### 📄 Preview of Uploaded Data"
        )

        st.dataframe(
            df.head()
        )

        lower_cols = [
            str(c).lower()
            for c in df.columns
        ]

        # Long format
        if (
            df.shape[1] == 2
            and "group" in lower_cols
            and "value" in lower_cols
        ):

            group_col = df.columns[
                lower_cols.index("group")
            ]

            value_col = df.columns[
                lower_cols.index("value")
            ]

            groups = [
                grp[value_col].dropna().tolist()
                for _, grp in df.groupby(group_col)
            ]

            st.success(
                "✅ Detected long format with 'Group' and 'Value' columns."
            )

            return groups

        # Wide format
        elif df.shape[1] >= 2:

            numeric_cols = [
                col
                for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col])
            ]

            if len(numeric_cols) < 2:

                st.error(
                    "⚠️ Need at least two numeric columns for wide format."
                )

                return None

            groups = [
                df[col].dropna().tolist()
                for col in numeric_cols
            ]

            st.success(
                "✅ Detected wide format "
                "(each numeric column = one group)."
            )

            return groups

        else:

            st.error(
                "⚠️ File must contain at least two numeric columns "
                "or a 'Group'-'Value' pair."
            )

            return None

    except Exception as e:

        st.error(
            f"Error reading file: {e}"
        )

        return None


# ==========================================================
# Decision Box
# ==========================================================
def decision_box(reject: bool):

    if reject:

        html = """
<div style="
    display:flex;
    align-items:center;
    gap:10px;
    padding:14px;
    border-radius:10px;
    background-color:#c8f7c5;
    margin:10px 0;
">
<span style="font-size:22px;">✅</span>
<span style="font-size:18px; color:black;">
<b>Decision: Reject H₀</b>
</span>
</div>
"""

    else:

        html = """
<div style="
    display:flex;
    align-items:center;
    gap:10px;
    padding:14px;
    border-radius:10px;
    background-color:#f7c5c5;
    margin:10px 0;
">
<span style="font-size:22px;">❌</span>
<span style="font-size:18px; color:black;">
<b>Decision: Do not reject H₀</b>
</span>
</div>
"""

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ==========================================================
# Plot Helpers — Plotly
# ==========================================================
def plot_horizontal_boxplots(groups):

    fig = go.Figure()

    for i, group in enumerate(groups):

        fig.add_trace(
            go.Box(
                x=group,
                name=f"Group {i+1}",
                orientation="h",
                boxmean=True
            )
        )

    fig.update_layout(
        title="Horizontal Boxplots of Groups",
        xaxis_title="Values",
        yaxis_title="Groups",
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def plot_f_rejection_region(
    f_stat,
    df_between,
    df_within,
    alpha
):

    xmax = max(
        6,
        f.ppf(
            0.995,
            df_between,
            df_within
        ),
        f_stat + 2
    )

    x = np.linspace(
        0.001,
        xmax,
        700
    )

    y = f.pdf(
        x,
        df_between,
        df_within
    )

    crit = f.ppf(
        1 - alpha,
        df_between,
        df_within
    )

    # ------------------------------------------------------
    # Do Not Reject Region
    # ------------------------------------------------------
    x_accept = np.linspace(
        0.001,
        crit,
        350
    )

    y_accept = f.pdf(
        x_accept,
        df_between,
        df_within
    )

    # ------------------------------------------------------
    # Reject Region
    # ------------------------------------------------------
    x_reject = np.linspace(
        crit,
        xmax,
        350
    )

    y_reject = f.pdf(
        x_reject,
        df_between,
        df_within
    )

    fig = go.Figure()

    # F Distribution Curve
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="F Distribution",
            line=dict(
                color="black",
                width=3
            )
        )
    )

    # Do Not Reject Region
    fig.add_trace(
        go.Scatter(
            x=x_accept,
            y=y_accept,
            mode="lines",
            fill="tozeroy",
            name="Do not reject H₀",
            line=dict(
                color="green"
            ),
            fillcolor="rgba(44,160,44,0.45)"
        )
    )

    # Reject Region
    fig.add_trace(
        go.Scatter(
            x=x_reject,
            y=y_reject,
            mode="lines",
            fill="tozeroy",
            name="Reject H₀",
            line=dict(
                color="red"
            ),
            fillcolor="rgba(214,39,40,0.55)"
        )
    )

    # Critical F Value
    fig.add_vline(
        x=crit,
        line_dash="dash",
        line_width=2,
        line_color="black",
        annotation_text=f"Critical F = {crit:.4f}",
        annotation_position="top"
    )

    # Observed F Statistic
    fig.add_vline(
        x=f_stat,
        line_width=3,
        line_color="blue",
        annotation_text=f"Observed F = {f_stat:.4f}",
        annotation_position="top right"
    )

    fig.update_layout(
        title=(
            "Classical Method: "
            "F Rejection Region "
            f"(df1={df_between}, df2={df_within})"
        ),
        xaxis_title="F",
        yaxis_title="Density",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "🟥 Red = Reject H₀ region   |   "
        "🟩 Green = Do not reject H₀ region"
    )


# ==========================================================
# One-Way ANOVA Calculation
# ==========================================================
def one_way_anova(
    groups,
    alpha,
    decimals
):

    st.markdown(
        "## 📊 One-Way ANOVA Test"
    )

    st.markdown("---")

    # ------------------------------------------------------
    # Basic Calculations
    # ------------------------------------------------------
    all_values = np.concatenate(
        groups
    )

    group_means = [
        np.mean(g)
        for g in groups
    ]

    group_vars = [
        np.var(
            g,
            ddof=1
        )
        for g in groups
    ]

    group_sizes = [
        len(g)
        for g in groups
    ]

    overall_mean = np.mean(
        all_values
    )

    # Sum of Squares Between
    ssb = sum(
        n * (m - overall_mean) ** 2
        for n, m
        in zip(
            group_sizes,
            group_means
        )
    )

    # Sum of Squares Within
    ssw = sum(
        sum(
            (x - m) ** 2
            for x in g
        )
        for g, m
        in zip(
            groups,
            group_means
        )
    )

    df_between = (
        len(groups) - 1
    )

    df_within = (
        sum(group_sizes)
        - len(groups)
    )

    msb = (
        ssb / df_between
    )

    msw = (
        ssw / df_within
    )

    f_stat = (
        msb / msw
    )

    p_value = (
        1
        - f.cdf(
            f_stat,
            df_between,
            df_within
        )
    )

    critical_value = f.ppf(
        1 - alpha,
        df_between,
        df_within
    )

    reject = (
        p_value <= alpha
    )


    # ======================================================
    # Step 1
    # ======================================================
    themed_box(
        "Step 1: Hypotheses"
    )

    st.latex(
        r"H_0: \mu_1 = \mu_2 = \dots = \mu_k"
    )

    st.latex(
        r"H_a: \text{At least one population mean differs}"
    )


    # ======================================================
    # Step 2
    # ======================================================
    themed_box(
        "Step 2: Test Statistic Components"
    )

    summary_df = pd.DataFrame(
        {
            "Group": [
                f"Group {i+1}"
                for i in range(
                    len(groups)
                )
            ],

            "n": group_sizes,

            "Mean": np.round(
                group_means,
                decimals
            ),

            "Variance": np.round(
                group_vars,
                decimals
            )
        }
    )

    st.dataframe(
        summary_df,
        use_container_width=True
    )

    st.latex(
        r"SSB = \sum n_i(\bar{x}_i - \bar{x})^2"
    )

    st.latex(
        r"SSW = \sum \sum (x_{ij} - \bar{x}_i)^2"
    )

    st.latex(
        r"MSB = \frac{SSB}{df_{\text{between}}}"
    )

    st.latex(
        r"MSW = \frac{SSW}{df_{\text{within}}}"
    )

    st.latex(
        r"F = \frac{MSB}{MSW}"
    )

    st.write(
        f"Overall Mean = "
        f"**{overall_mean:.{decimals}f}**"
    )

    st.write(
        f"SSB = "
        f"**{ssb:.{decimals}f}**"
    )

    st.write(
        f"SSW = "
        f"**{ssw:.{decimals}f}**"
    )

    st.write(
        f"df between = "
        f"**{df_between}**"
    )

    st.write(
        f"df within = "
        f"**{df_within}**"
    )

    st.write(
        f"MSB = "
        f"**{msb:.{decimals}f}**"
    )

    st.write(
        f"MSW = "
        f"**{msw:.{decimals}f}**"
    )

    st.write(
        f"F statistic = "
        f"**{f_stat:.{decimals}f}**"
    )


    # ======================================================
    # Step 2A
    # ======================================================
    themed_box(
        "Step 2A: Visualize Groups (Horizontal Boxplots)"
    )

    plot_horizontal_boxplots(
        groups
    )


    # ======================================================
    # Step 3
    # ======================================================
    themed_box(
        "Step 3: Classical Method"
    )

    st.write(
        f"Critical value = "
        f"**{critical_value:.{decimals}f}**"
    )

    st.markdown(
        f"Decision rule: Reject H₀ if "
        f"**F > {critical_value:.{decimals}f}**."
    )

    st.markdown(
        f"Observed test statistic: "
        f"**F = {f_stat:.{decimals}f}**"
    )

    st.markdown(
        f"Classical method decision: "
        f"**{'Reject H₀' if reject else 'Do not reject H₀'}**"
    )

    plot_f_rejection_region(
        f_stat,
        df_between,
        df_within,
        alpha
    )


    # ======================================================
    # Step 4
    # ======================================================
    themed_box(
        "Step 4: P-value Approach"
    )

    st.write(
        f"P-value = "
        f"**{p_value:.{decimals}f}**"
    )

    st.write(
        f"α = "
        f"**{alpha:.{decimals}f}**"
    )

    st.markdown(
        f"P-value approach decision: "
        f"**{'Reject H₀' if reject else 'Do not reject H₀'}**"
    )

    decision_box(
        reject
    )


    # ======================================================
    # Step 5
    # ======================================================
    themed_box(
        "Step 5: Conclusion"
    )

    interpretation = (
        "Since the p-value is "
        + (
            "less than"
            if reject
            else "greater than or equal to"
        )
        + f" α = {alpha}, "
        + (
            "we reject "
            if reject
            else "we do not reject "
        )
        + "H₀. "
        + (
            "There is sufficient evidence that at least one group mean differs."
            if reject
            else
            "There is not sufficient evidence to conclude that the group means differ."
        )
    )

    st.write(
        interpretation
    )


    # ======================================================
    # ANOVA Table
    # ======================================================
    themed_box(
        "ANOVA Table"
    )

    anova_df = pd.DataFrame(
        {
            "Source": [
                "Between Groups",
                "Within Groups",
                "Total"
            ],

            "SS": [
                round(
                    ssb,
                    decimals
                ),

                round(
                    ssw,
                    decimals
                ),

                round(
                    ssb + ssw,
                    decimals
                )
            ],

            "df": [
                df_between,
                df_within,
                df_between + df_within
            ],

            "MS": [
                round(
                    msb,
                    decimals
                ),

                round(
                    msw,
                    decimals
                ),

                ""
            ],

            "F": [
                round(
                    f_stat,
                    decimals
                ),

                "",

                ""
            ]
        }
    )

    st.dataframe(
        anova_df,
        use_container_width=True
    )


# ==========================================================
# Streamlit Interface
# ==========================================================
def run():

    st.header(
        "📊 One-Way ANOVA Test"
    )

    st.markdown(
        """
This tool tests whether **three or more group means are equal** using the F-test.

---
**Input Options:**
- Manual entry
- Upload CSV/Excel
"""
    )

    input_method = st.radio(
        "Choose data input method:",
        [
            "📋 Manual Entry",
            "📂 Upload CSV/Excel File"
        ]
    )

    groups = []

    # ======================================================
    # Manual Entry
    # ======================================================
    if input_method == "📋 Manual Entry":

        mode = st.radio(
            "Manual Entry Method:",
            [
                "Enter all group data in one line (semicolon-separated)",
                "Enter number of groups and input each group separately"
            ]
        )

        # --------------------------------------------------
        # One-line input
        # --------------------------------------------------
        if mode == "Enter all group data in one line (semicolon-separated)":

            input_text = st.text_area(
                "Enter group data:",
                placeholder=(
                    "12,15,18; "
                    "25,28,29; "
                    "10,12,14"
                )
            )

            if input_text:

                groups = parse_groups(
                    input_text
                )

        # --------------------------------------------------
        # Separate Groups
        # --------------------------------------------------
        else:

            num_groups = st.number_input(
                "Number of groups",
                min_value=2,
                step=1
            )

            for i in range(
                num_groups
            ):

                group_text = st.text_input(
                    f"Group {i+1} data (comma-separated)",
                    key=f"group_{i}"
                )

                if group_text:

                    try:

                        groups.append(
                            list(
                                map(
                                    float,
                                    group_text.strip().split(",")
                                )
                            )
                        )

                    except Exception:

                        st.error(
                            f"⚠️ Invalid input in Group {i+1}. "
                            "Use commas to separate values."
                        )


    # ======================================================
    # Upload CSV / Excel
    # ======================================================
    elif input_method == "📂 Upload CSV/Excel File":

        groups = load_uploaded_data()


    # ======================================================
    # Settings
    # ======================================================
    alpha = st.number_input(
        "Significance level (α)",
        min_value=0.001,
        max_value=0.5,
        value=0.05
    )

    decimals = st.number_input(
        "Decimal places for rounding",
        min_value=1,
        max_value=10,
        value=4
    )


    # ======================================================
    # Run ANOVA Button
    # ======================================================
    if st.button(
        "▶️ Run ANOVA Test"
    ):

        if not groups or len(groups) < 2:

            st.error(
                "⚠️ Please enter at least two groups."
            )

            return

        if any(
            len(g) < 2
            for g in groups
        ):

            st.error(
                "⚠️ Each group must contain at least two values."
            )

            return

        one_way_anova(
            groups,
            alpha,
            decimals
        )


# ==========================================================
# Run Script
# ==========================================================
if __name__ == "__main__":
    run()


# Compatibility for main app
run_anova_tool = run
