# ==========================================================
# galton_regression_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ==========================================================
# Helper Functions
# ==========================================================
def round_value(value, decimals=4):
    try:
        return round(float(value), decimals)
    except Exception:
        return value

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

def print_summary(model, decimals=4):
    """Display regression summary key results."""
    st.markdown("### 📄 Regression Summary")
    st.text(model.summary())

    st.markdown("### 📈 Model Statistics")
    st.write(f"**R²:** {round_value(model.rsquared, decimals)}")
    st.write(f"**Adjusted R²:** {round_value(model.rsquared_adj, decimals)}")
    st.write(f"**F-statistic:** {round_value(model.fvalue, decimals)}")
    st.write(f"**p-value (F-test):** {round_value(model.f_pvalue, decimals)}")

    st.markdown("### 📄 Coefficients Table")
    coef_table = model.summary2().tables[1]
    st.dataframe(np.round(coef_table, decimals))

def plot_regression(y, y_hat, y_label="y", title="Observed vs Predicted"):
    fig, ax = plt.subplots()
    ax.scatter(y_hat, y, color="#007acc", edgecolor="black", alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="Perfect Fit")
    ax.set_xlabel("Predicted (ŷ)")
    ax.set_ylabel(f"Observed ({y_label})")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# ==========================================================
# Main App
# ==========================================================
def run():
    st.header("👨‍🏫 Galton Family Regression Analysis")

    st.markdown("""
    This tool explores **bivariate** and **multiple** regression models  
    using the historic **Galton family dataset (19th century)**.  
    It estimates models of the form:

    \[
    \hat{y} = b_0 + b_1x_1 + b_2x_2 + \dots + b_kx_k
    \]

    using the **Ordinary Least Squares (OLS)** method.
    """)

    st.subheader("📂 Upload or Paste Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    example_data = """son,father,mother,siblings
70,70,65,5
73,70,66.5,10
71,65.5,63,3
72,71,65.5,5
79,70,65,6
69.7,67,65,5
63,65,66,3
70.5,71.7,65.5,0
64,68,60,6
72,66,65.5,4
"""

    st.caption("Or paste your dataset below (comma-separated):")
    text_data = st.text_area("Paste Data:", value=example_data, height=200)

    decimals = st.number_input("Decimal places for output", 1, 10, 4)

    # ----------------------------------------------------------
    # Load the data
    # ----------------------------------------------------------
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            return
    else:
        try:
            df = pd.read_csv(pd.io.common.StringIO(text_data))
        except Exception as e:
            st.error("❌ Could not parse pasted data. Make sure it's comma-separated.")
            return

    st.markdown("### 📊 Data Preview")
    st.dataframe(df.head())

    # ----------------------------------------------------------
    # Variable Selection
    # ----------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Dataset must have at least two numeric columns.")
        return

    st.markdown("### 🧩 Variable Selection")
    y_col = st.selectbox("Select dependent variable (y):", numeric_cols, index=0)
    X_cols = st.multiselect("Select independent variable(s) (x):", [c for c in numeric_cols if c != y_col])

    # ----------------------------------------------------------
    # Run Bivariate Regression (first predictor only)
    # ----------------------------------------------------------
    if X_cols and st.button("▶️ Run Bivariate Regression (y ~ first x)"):
        try:
            x1 = X_cols[0]
            X = sm.add_constant(df[x1])
            y = df[y_col]
            model = sm.OLS(y, X).fit()

            step_box(f"**Step 1:** Fit the bivariate model predicting {y_col} from {x1}")
            st.latex(r"\hat{y} = b_0 + b_1x_1")

            step_box("**Step 2:** Model Summary and Fit Statistics")
            print_summary(model, decimals)

            plot_regression(y, model.fittedvalues, y_label=y_col, title=f"{y_col} vs Predicted ({x1})")

        except Exception as e:
            st.error(f"❌ Error: {e}")

    # ----------------------------------------------------------
    # Run Multiple Regression
    # ----------------------------------------------------------
    if len(X_cols) > 1 and st.button("▶️ Run Multiple Regression (y ~ all x’s)"):
        try:
            X = sm.add_constant(df[X_cols])
            y = df[y_col]
            model = sm.OLS(y, X).fit()

            step_box(f"**Step 1:** Fit the multiple regression model for {y_col}")
            eq = " + ".join([f"b{i+1}{x}" for i, x in enumerate(X_cols)])
            st.latex(fr"\hat{{y}} = b_0 + {eq}")

            step_box("**Step 2:** Model Summary and Fit Statistics")
            print_summary(model, decimals)

            plot_regression(y, model.fittedvalues, y_label=y_col, title=f"{y_col} vs Predicted (All Predictors)")

        except Exception as e:
            st.error(f"❌ Error: {e}")

    # ----------------------------------------------------------
    # Model Comparison (Nested Models)
    # ----------------------------------------------------------
    if len(X_cols) > 1:
        with st.expander("⚖️ Model Comparison (Nested F-test)"):
            st.caption("Compare a smaller model to a larger model using the F-test (NHST).")

            smaller_x = st.multiselect(
                "Select predictors for the smaller model (nested):",
                X_cols,
                help="The smaller model should be a subset of the full model predictors."
            )

            if smaller_x and st.button("Run Model Comparison (F-test)"):
                try:
                    y = df[y_col]
                    full_X = sm.add_constant(df[X_cols])
                    small_X = sm.add_constant(df[smaller_x])

                    full_model = sm.OLS(y, full_X).fit()
                    small_model = sm.OLS(y, small_X).fit()

                    df_num = full_model.df_model - small_model.df_model
                    df_den = full_model.df_resid
                    ssr_diff = small_model.ssr - full_model.ssr
                    msr_diff = ssr_diff / df_num
                    mse_full = full_model.ssr / df_den
                    F_stat = msr_diff / mse_full
                    p_val = 1 - sm.stats.f.cdf(F_stat, df_num, df_den)

                    st.markdown("### 🔍 Model Comparison Results")
                    st.write(f"F = **{round_value(F_stat, decimals)}**, df₁ = {int(df_num)}, df₂ = {int(df_den)}, p = **{round_value(p_val, decimals)}**")

                    if p_val <= 0.05:
                        st.success("✅ Reject H₀: The larger model significantly improves fit.")
                    else:
                        st.info("❌ Fail to reject H₀: The smaller model is sufficient.")

                except Exception as e:
                    st.error(f"❌ Error running comparison: {e}")


# ==========================================================
# Run Script
# ==========================================================
if __name__ == "__main__":
    run()

# ✅ Compatibility alias for MIND suite integration
run_multiple_regression_tool = run


