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


# ==========================================================
# Regression Summary & Plots
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

    st.markdown("""
    This tool estimates the best-fitting line  
    \\[
    \\hat{y} = b_0 + b_1x_1 + b_2x_2 + \\dots + b_kx_k
    \\]
    using the **Ordinary Least Squares (OLS)** method.
    """)

    # --- File upload section ---
    st.subheader("📘 Data Input")
    uploaded_file = st.file_uploader("📂 Upload CSV or Excel file (optional)", type=["csv", "xlsx"])

    st.markdown("### ✏️ Or enter data manually")
    st.caption("""
    Each row represents **one observation (student, case, or trial)**.  
    - The **first value** in each row is the dependent variable *(y)*  
    - The remaining values are the independent variables *(x₁, x₂, …, xₖ)*  
    - Separate values with **commas**, and press **Enter** for each new row.

    Example (predicting exam score `y` using hours studied `x₁` and quiz average `x₂`):
    ```
    85, 10, 90
    78, 8, 85
    92, 12, 95
    70, 5, 75
    ```
    """)

    raw_data = st.text_area("Enter your data below:", height=200, placeholder="y, x₁, x₂ ...")

    decimals = st.number_input("Decimal places for rounding", min_value=1, max_value=10, value=4, step=1)

    # ======================================================
    # Case 1: File upload
    # ======================================================
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("📊 Data Preview:")
            st.dataframe(df.head())

            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) < 2:
                st.error("File must contain at least two numeric columns (y and predictors).")
                return

            st.markdown("### 🧩 Variable Selection")
            y_col = st.selectbox("Select dependent variable (y):", num_cols, index=0)
            X_cols = st.multiselect("Select independent variable(s) (x):", [c for c in num_cols if c != y_col])

            if st.button("👨‍💻 Run Regression (Uploaded Data)"):
                if not X_cols:
                    st.error("Please select at least one independent variable.")
                    return

                data = df.dropna(subset=[y_col] + X_cols)
                y = data[y_col].to_numpy(dtype=float)
                X = data[X_cols].to_numpy(dtype=float)
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()

                step_box("**Step 1:** Fit the multiple regression model")
                st.latex(r"y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_kx_k + \varepsilon")

                step_box("**Step 2:** Review Model Summary and Fit Statistics")
                print_regression_summary(model, decimals)

                step_box("**Step 3:** Examine Residuals")
                plot_residuals(model)

                step_box("**Step 4:** Predict New Values")
                new_x_input = st.text_input("Enter predictor values (x₁, x₂, …, xₖ):", placeholder="9, 87")
                if new_x_input.strip():
                    try:
                        new_x = np.array(list(map(float, new_x_input.replace(",", " ").split())))
                        if len(new_x) != len(X_cols):
                            st.error(f"Please enter {len(X_cols)} predictor values.")
                        else:
                            y_hat = model.predict([np.insert(new_x, 0, 1)])[0]
                            st.success(f"Predicted y = **{round_value(y_hat, decimals)}**")
                    except Exception as e:
                        st.error(f"Error: {e}")

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

    # ======================================================
    # Case 2: Manual Data Entry
    # ======================================================
    elif raw_data.strip() and st.button("👨‍💻 Run Regression (Manual Entry)"):
        try:
            rows = [r.strip() for r in raw_data.strip().split("\n") if r.strip()]
            data = [list(map(float, r.replace(",", " ").split())) for r in rows]
            data = np.array(data)

            if data.shape[1] < 2:
                st.error("Each row must include at least one predictor variable (x).")
                return

            y = data[:, 0]
            X = data[:, 1:]
            X = sm.add_constant(X)
            k = X.shape[1] - 1

            st.success(f"✅ Data entered successfully: {len(y)} observations and {k} predictor(s).")

            model = sm.OLS(y, X).fit()

            step_box("**Step 1:** Fit the multiple regression model")
            st.latex(r"y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_kx_k + \varepsilon")

            step_box("**Step 2:** Review Model Summary and Fit Statistics")
            print_regression_summary(model, decimals)

            step_box("**Step 3:** Examine Residuals")
            plot_residuals(model)

            step_box("**Step 4:** Predict New Values")
            new_x_input = st.text_input(f"Enter {k} predictor value(s) (x₁, …, xₖ):", placeholder="10, 90")
            if new_x_input.strip():
                try:
                    new_x = np.array(list(map(float, new_x_input.replace(",", " ").split())))
                    if len(new_x) != k:
                        st.error(f"Please enter {k} predictor values.")
                    else:
                        y_hat = model.predict([np.insert(new_x, 0, 1)])[0]
                        st.success(f"Predicted y = **{round_value(y_hat, decimals)}**")
                except Exception as e:
                    st.error(f"Error: {e}")

        except Exception as e:
            st.error(f"❌ Error: {e}")

    else:
        st.info("👆 Upload a dataset or enter data manually to begin.")


# ==========================================================
# Run Script
# ==========================================================
if __name__ == "__main__":
    run()

# ✅ Compatibility alias for integration with the main suite
run_multiple_regression_tool = run

