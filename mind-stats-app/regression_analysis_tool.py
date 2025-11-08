# ==========================================================
# regression_analysis_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# ==========================================================
# Helper Function
# ==========================================================
def step_box(text):
    """Stylized explanation box for clarity."""
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
# SIMPLE LINEAR REGRESSION
# ==========================================================
def run_simple_regression_tool():
    st.header("👨‍🏫 Simple Linear Regression")

    if "simple_reg" not in st.session_state:
        st.session_state.simple_reg = {}

    # --- Introduction
    st.markdown("This tool estimates the best-fitting line using the **Ordinary Least Squares (OLS)** method.")
    st.latex(r"\hat{y} = b_0 + b_1x")

    # --- Data input
    uploaded_file = st.file_uploader("📂 Upload CSV or Excel file (optional)", type=["csv", "xlsx"])
    y_input = st.text_area("Or enter dependent variable (y) values (comma-separated):")
    x_input = st.text_area("Or enter independent variable (x) values (comma-separated):")
    decimals = st.number_input("Decimal places for rounding", 0, 10, 4)

    df = None
    x_col = y_col = None

    # --- File upload handling
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            st.write("📋 Data Preview:")
            st.dataframe(df.head())

            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) < 2:
                st.error("The file must contain at least two numeric columns.")
                return

            c1, c2 = st.columns(2)
            with c1:
                x_col = st.selectbox("Select X (independent variable)", num_cols)
            with c2:
                y_col = st.selectbox("Select Y (dependent variable)", num_cols, index=1 if len(num_cols) > 1 else 0)
        except Exception as e:
            st.error(f"⚠️ File error: {e}")
            return

    # --- Run regression
    if st.button("👨‍💻 Run Simple Regression"):
        try:
            if df is not None and x_col and y_col:
                data_xy = df[[x_col, y_col]].dropna()
                x = data_xy[x_col].to_numpy()
                y = data_xy[y_col].to_numpy()
            else:
                y = np.array(list(map(float, y_input.replace(",", " ").split())))
                x = np.array(list(map(float, x_input.replace(",", " ").split())))

            if len(x) != len(y) or len(x) == 0:
                st.error("x and y must have the same length and contain valid numeric data.")
                return

            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            y_pred = model.predict(X)

            st.session_state.simple_reg = {
                "fitted": True,
                "intercept": float(model.params[0]),
                "slope": float(model.params[1]),
                "summary": model.summary().as_text(),
                "x": x,
                "y": y,
                "y_pred": y_pred,
                "residuals": model.resid,
                "fitted_vals": model.fittedvalues,
                "decimals": int(decimals)
            }

        except Exception as e:
            st.error(f"Error fitting model: {e}")
            return

    # --- Display results
    if st.session_state.simple_reg.get("fitted"):
        x = st.session_state.simple_reg["x"]
        y = st.session_state.simple_reg["y"]
        y_pred = st.session_state.simple_reg["y_pred"]
        residuals = st.session_state.simple_reg["residuals"]
        intercept = st.session_state.simple_reg["intercept"]
        slope = st.session_state.simple_reg["slope"]
        decimals = st.session_state.simple_reg["decimals"]

        st.markdown("---")
        st.markdown("### 🧩 Hypotheses")
        st.latex(r"H_0: \beta_1 = 0 \quad\text{(no linear relationship)}")
        st.latex(r"H_a: \beta_1 \neq 0 \quad\text{(linear relationship exists)}")

        # Step 1: Correlation
        step_box("**Step 1:** Compute correlation and assess linearity")
        corr_coef, p_val = stats.pearsonr(x, y)
        r2 = corr_coef ** 2
        summary_df = pd.DataFrame({
            "Statistic": ["r", "r²", "p-value", "Slope (b₁)", "Intercept (b₀)"],
            "Value": [
                round(corr_coef, decimals),
                round(r2, decimals),
                round(p_val, decimals),
                round(slope, decimals),
                round(intercept, decimals)
            ]
        })
        st.dataframe(summary_df)

        # Step 2: Regression Equation
        step_box("**Step 2:** Compute regression coefficients")
        st.latex(r"\hat{y} = b_0 + b_1x")
        st.write(f"**Estimated model:** ŷ = {round(intercept, decimals)} + {round(slope, decimals)}·x")
        st.text(st.session_state.simple_reg["summary"])

        # Step 3: Scatter plot
        step_box("**Step 3:** Visualize regression line")
        order = np.argsort(x)
        fig, ax = plt.subplots()
        ax.scatter(x, y, color="#007acc", label="Observed Data")
        ax.plot(x[order], y_pred[order], color="red", label="Regression Line")
        ax.set_xlabel("X (Independent Variable)")
        ax.set_ylabel("Y (Dependent Variable)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # Step 4: Residual plot
        step_box("**Step 4:** Analyze residuals for randomness")
        fig2, ax2 = plt.subplots()
        ax2.scatter(st.session_state.simple_reg["fitted_vals"], residuals, color="#007acc")
        ax2.axhline(y=0, linestyle="--", color="red")
        ax2.set_xlabel("Fitted Values")
        ax2.set_ylabel("Residuals")
        ax2.grid(True)
        st.pyplot(fig2)

        # Step 5: Prediction
        step_box("**Step 5:** Make predictions and interpret results")
        st.caption("Regression equation for prediction:")
        st.latex(r"\hat{y} = b_0 + b_1x")
        st.write(f"**ŷ = {round(intercept, decimals)} + {round(slope, decimals)}·x**")

        new_x = st.number_input("Enter a new x value:", value=0.0, format="%.4f")
        y_hat = intercept + slope * new_x
        st.success(f"Predicted ŷ = **{round(y_hat, decimals)}**")

        actual_y = st.text_input("Optional: Enter actual y to compute residual:")
        if actual_y.strip():
            try:
                residual_new = float(actual_y) - y_hat
                st.info(f"Residual (Actual − Predicted) = **{round(residual_new, decimals)}**")
            except ValueError:
                st.error("Please enter a valid numeric value for actual y.")

        if st.button("🧹 Clear Fitted Model"):
            st.session_state.simple_reg = {}

# ==========================================================
# MULTIPLE LINEAR REGRESSION
# ==========================================================
def run_multiple_regression_tool():
    st.header("👨‍🏫 Multiple Linear Regression")

    st.markdown("This tool estimates a multiple regression model using the **Ordinary Least Squares (OLS)** method.")
    st.latex(r"\hat{y} = b_0 + b_1x_1 + b_2x_2 + \dots + b_kx_k")

    uploaded_file = st.file_uploader("📂 Upload CSV or Excel file (optional)", type=["csv", "xlsx"])
    raw_matrix = st.text_area(
        "Or enter matrix data (each row = observation, last column = dependent variable y):\n"
        "Example:\n5, 7, 2\n6, 8, 3\n7, 9, 4\n8, 10, 5"
    )
    decimals = st.number_input("Decimal places for output", 0, 10, 4)

    if st.button("👨‍💻 Run Multiple Regression"):
        try:
            if uploaded_file:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                st.write("📋 Data Preview:")
                st.dataframe(df.head())

                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(num_cols) < 2:
                    st.error("File must contain at least one predictor and one dependent variable.")
                    return

                y_col = st.selectbox("Select dependent variable (Y)", num_cols, index=len(num_cols) - 1)
                X_cols = [col for col in num_cols if col != y_col]
                X = df[X_cols].dropna()
                y = df.loc[X.index, y_col].to_numpy()
                X = X.to_numpy()
            else:
                rows = [r for r in raw_matrix.strip().split("\n") if r.strip()]
                data = np.array([list(map(float, r.split(","))) for r in rows])
                if data.shape[1] < 2:
                    st.error("Matrix must have ≥2 columns (predictors + response).")
                    return
                X, y = data[:, :-1], data[:, -1]

            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            st.subheader("📄 Regression Summary")
            st.text(model.summary())

            st.subheader("📉 Residual Plot (Residuals vs Fitted Values)")
            residuals = model.resid
            fitted = model.fittedvalues
            fig, ax = plt.subplots()
            ax.scatter(fitted, residuals, color="#007acc")
            ax.axhline(y=0, linestyle="--", color="red")
            ax.set_xlabel("Fitted Values")
            ax.set_ylabel("Residuals")
            ax.grid(True)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================================
# End of Module
# ==========================================================


