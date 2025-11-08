# ====================================================
# 📊 MIND: Multiple Regression (Advanced)
# Professor Edward Pineda-Castro, Los Angeles City College
# ====================================================

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ====================================================
# MAIN FUNCTION
# ====================================================
def run():
    st.header("👨‍🏫 Multiple Regression (Advanced)")

    # ---------- Intro Section ----------
    st.markdown("""
    This advanced module allows students to:

    - Select which variable is the dependent variable (y)
    - Select one or more independent variables (x₁, x₂, …)
    - Run both **bivariate** and **multiple** regression
    - Perform **model comparison (nested F-test)**

    The model is estimated using **Ordinary Least Squares (OLS):**
    """)
    st.latex(r"\hat{y} = b_0 + b_1x_1 + b_2x_2 + \cdots + b_kx_k")

    # ---------- File Upload Section ----------
    st.markdown("### 📂 Upload CSV or Excel File (optional)")
    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx"])

    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success("✅ Data successfully loaded!")
            st.dataframe(df.head())

            # Select dependent and independent variables
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 2:
                y_col = st.selectbox("Select dependent variable (y)", num_cols)
                x_cols = st.multiselect(
                    "Select one or more independent variables (x₁, x₂, …)",
                    [col for col in num_cols if col != y_col],
                )

                if y_col and x_cols:
                    y = df[y_col]
                    X = df[x_cols]
                    X = sm.add_constant(X)
                else:
                    st.warning("Please select both y and at least one x variable.")
                    return
            else:
                st.error("❌ The dataset must have at least two numeric columns.")
                return

        except Exception as e:
            st.error(f"Error loading file: {e}")
            return

    else:
        # ---------- Manual Data Entry Section ----------
        st.markdown("### ✍️ Enter Data Manually")
        st.markdown("""
        You can manually enter your data below:
        - **Dependent variable (y):** comma-separated values  
          Example: `1, 2, 3, 5`
        - **Independent variables (x’s):** each separated by semicolons  
          Example: `2, 3, 4, 5; 1, 2, 3, 5`
        """)

        y_input = st.text_input("Enter dependent variable (y) values:")
        x_input = st.text_area("Enter independent variable (x) values:")

        y, X = None, None
        if y_input.strip() and x_input.strip():
            try:
                y = np.array([float(v) for v in y_input.replace(",", " ").split()])
                X_lists = [
                    [float(x) for x in row.replace(",", " ").split()]
                    for row in x_input.strip().split(";")
                ]
                X = np.column_stack(X_lists)
                if len(y) != X.shape[0]:
                    st.error("❌ Number of y values must match number of rows in X.")
                    return
                X = sm.add_constant(X)
            except Exception as e:
                st.error(f"⚠️ Error parsing manual data: {e}")
                return
        else:
            st.info("👆 Upload a dataset above or enter data manually to continue.")
            return

    # ---------- Regression Computation ----------
    try:
        model = sm.OLS(y, X).fit()
        st.subheader("📄 Regression Summary")
        st.text(model.summary())

        # ---------- Residual Plot ----------
        st.subheader("📉 Residual Plot (Residuals vs Fitted Values)")
        residuals = model.resid
        fitted = model.fittedvalues
        fig, ax = plt.subplots()
        ax.scatter(fitted, residuals)
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        ax.grid(True)
        st.pyplot(fig)

        # ---------- Optional Model Comparison ----------
        st.markdown("### 🔍 Optional: Compare with Simpler Model")
        st.caption("Enter the variables for a smaller (nested) model to test whether adding predictors improves fit.")

        smaller_x = st.text_input("Enter smaller model x’s (comma-separated, must exist in dataset or manual entry):")
        if smaller_x.strip():
            try:
                if df is not None:
                    smaller_vars = [x.strip() for x in smaller_x.split(",")]
                    X_small = sm.add_constant(df[smaller_vars])
                    model_small = sm.OLS(df[y_col], X_small).fit()
                else:
                    idxs = [int(i) - 1 for i in smaller_x.replace(",", " ").split()]
                    X_small = sm.add_constant(X[:, [i + 1 for i in idxs]])
                    model_small = sm.OLS(y, X_small).fit()

                f_test = model.compare_f_test(model_small)
                st.markdown("**F-Test for Model Comparison (Nested Models)**")
                st.write(f"F = {round(f_test[0], 4)}, p = {round(f_test[1], 4)}")

                if f_test[1] <= 0.05:
                    st.success("✅ The larger model significantly improves the fit (reject H₀).")
                else:
                    st.info("❌ The larger model does not significantly improve the fit (fail to reject H₀).")
            except Exception as e:
                st.error(f"⚠️ Could not run model comparison: {e}")

    except Exception as e:
        st.error(f"⚠️ Regression Error: {e}")
