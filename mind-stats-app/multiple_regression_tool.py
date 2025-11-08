# ==========================================================
# multiple_regression_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

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


def parse_matrix(input_text):
    """Parse user-entered matrix data."""
    try:
        rows = [r for r in input_text.strip().split("\n") if r.strip()]
        data = [list(map(float, r.replace(",", " ").split())) for r in rows]
        data = np.array(data, dtype=float)
        if data.shape[1] < 2:
            raise ValueError("Each row must include at least one predictor variable (x).")
        X = data[:, 1:]
        y = data[:, 0]
        return X, y
    except Exception:
        raise ValueError("Invalid format: please check commas, spaces, and newlines.")


# ==========================================================
# Regression Report
# ==========================================================
def print_regression_summary(model, decimals):
    st.subheader("📄 Regression Summary")
    st.text(model.summary())

    st.markdown("### 📈 Model Fit Statistics")
    st.write(f"**R² (Coefficient of Determination):** {round_value(model.rsquared, decimals)}")
    st.write(f"**Adjusted R²:** {round_value(model.rsquared_adj, decimals)}")
    st.write(f"**F-statistic:** {round_value(model.fvalue, decimals)}")
    st.write(f"**p-value (F-test):** {round_value(model.f_pvalue, decimals)}")

    st.markdown("### 📄 Coefficients Table")
    coef_table = model.summary2().tables[1]
    st.dataframe(np.round(coef_table, decimals))


def plot_residuals(model):
    residuals = model.resid
    fitted = model.fittedvalues

    st.markdown("### 📉 Residual Diagnostics")

    fig1, ax1 = plt.subplots()
    ax1.scatter(fitted, residuals, color="#007acc", edgecolor="black")
    ax1.axhline(y=0, color="gray", linestyle="--")
    ax1.set_xlabel("Fitted Values (ŷ)")
    ax1.set_ylabel("Residuals (y - ŷ)")
    ax1.set_title("Residuals vs. Fitted Values")
    ax1.grid(True)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.hist(residuals, bins=10, color="#72bcd4", edgecolor="black")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Histogram of Residuals")
    st.pyplot(fig2)


# ==========================================================
# Main App
# ==========================================================
def run():
    st.header("👨‍🏫 Multiple Regression Analysis")

    st.markdown(
        """
        ### 📘 Data Input Instructions
        Each **row** represents one observation (student, case, or trial).  
        **Format rules:**
        - The **first value** in each row is the dependent variable *(y)*.  
        - The **remaining values** are the independent variables *(x₁, x₂, …, xₖ)*.  
        - Separate values with **commas or spaces**.  
        - Press **Enter** (new line) after each observation.
        
        **Example:** Predict exam score (y) using hours studied (x₁) and quiz average (x₂):
        ```
        85, 10, 90
        78, 8, 85
        92, 12, 95
        70, 5, 75
        ```
        **Here:**
        - y = exam score (dependent variable)
        - x₁ = hours studied
        - x₂ = quiz average
        
        ---
        📏 **Format summary:**
        ```
        y, x₁, x₂, …, xₖ   ← order of variables per row
        k = number of predictors (independent variables)
        ```
        """
    )

    uploaded_file = st.file_uploader("📂 Upload CSV or Excel file (optional)", type=["csv", "xlsx"])
    raw_data = st.text_area(
        "Or manually enter your data below (follow the format above):",
        placeholder="85, 10, 90\n78, 8, 85\n92, 12, 95\n70, 5, 75",
        height=200,
    )

    decimals = st.number_input("Decimal places for rounding", min_value=1, max_value=10, value=4, step=1)

    if st.button("👨‍💻 Run Multiple Regression"):
        try:
            if uploaded_file:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.dataframe(df.head())

                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(num_cols) < 2:
                    st.error("File must contain at least two numeric columns (y + predictors).")
                    return

                y_col = st.selectbox("Select Dependent Variable (y)", num_cols, index=0)
                X_cols = [c for c in num_cols if c != y_col]
                X = df[X_cols].dropna().to_numpy(dtype=float)
                y = df.loc[X.index, y_col].to_numpy(dtype=float)
            else:
                X, y = parse_matrix(raw_data)

            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            step_box("**Step 1:** Compute regression model using OLS")
            st.latex(r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k + \varepsilon")

            step_box("**Step 2:** Review Model Summary and Fit Statistics")
            print_regression_summary(model, decimals)

            step_box("**Step 3:** Examine Residuals for Randomness and Normality")
            plot_residuals(model)

            step_box("**Step 4:** Predict New Values")
            st.caption("Enter predictor values (x₁, x₂, …, xₖ) separated by commas:")
            new_x_input = st.text_input("Example: 9, 87  → 9 hours studied, 87 quiz average")
            if new_x_input.strip():
                try:
                    new_x = np.array(list(map(float, new_x_input.replace(",", " ").split())))
                    if len(new_x) != X.shape[1] - 1:
                        st.error(f"Please enter {X.shape[1]-1} predictor values.")
                    else:
                        y_hat = model.predict([np.insert(new_x, 0, 1)])[0]
                        st.success(f"Predicted y = **{round_value(y_hat, decimals)}**")
                except Exception as e:
                    st.error(f"Error: {e}")

        except Exception as e:
            st.error(f"❌ Error: {e}")


# ==========================================================
# Run Script
# ==========================================================
if __name__ == "__main__":
    run()

# ✅ Compatibility alias for main suite integration
run_multiple_regression_tool = run

