# regression_analysis_tool.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# -----------------------------
# SIMPLE LINEAR REGRESSION
# -----------------------------
def run_simple_regression_tool():
    st.header("👨‍🏫 Simple Linear Regression")

    # Keep state for fitted model params so prediction works across reruns
    if "simple_reg" not in st.session_state:
        st.session_state.simple_reg = {}

    st.markdown("### Data Input")
    uploaded_file = st.file_uploader("Upload CSV or Excel file (optional)", type=["csv", "xlsx"])
    y_input = st.text_area("Or enter dependent variable (y) values, separated by commas:")
    x_input = st.text_area("Or enter independent variable (x) values, separated by commas:")
    decimals = st.number_input("Decimal places for output", min_value=0, max_value=10, value=4, step=1)

    df = None
    x_col = y_col = None

    # If a file is uploaded, show preview and numeric column selectors
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("Data Preview:")
            st.dataframe(df.head())

            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) < 2:
                st.error("The file must contain at least two numeric columns.")
                return

            c1, c2 = st.columns(2)
            with c1:
                x_col = st.selectbox("Select X (independent) column", num_cols, key="simple_x_col")
            with c2:
                # default to the second numeric column if present
                default_y_idx = 1 if len(num_cols) > 1 else 0
                y_col = st.selectbox("Select Y (dependent) column", num_cols, index=default_y_idx, key="simple_y_col")

        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

    # Fit model button
    if st.button("👨‍💻 Run Simple Regression"):
        try:
            # Build x, y based on chosen input method
            if df is not None and x_col is not None and y_col is not None:
                # drop rows with NaNs in either column to keep them aligned
                data_xy = df[[x_col, y_col]].dropna()
                x = data_xy[x_col].to_numpy(dtype=float)
                y = data_xy[y_col].to_numpy(dtype=float)
            else:
                # Parse manual text areas
                y = np.array(list(map(float, y_input.replace(",", " ").split())))
                x = np.array(list(map(float, x_input.replace(",", " ").split())))

            if x.size == 0 or y.size == 0:
                st.error("Please provide data via upload or text inputs.")
                return

            if len(y) != len(x):
                st.error("x and y must have the same length.")
                return

            # Fit OLS model
            X = sm.add_constant(x)  # shape (n, 2)
            model = sm.OLS(y, X).fit()
            y_pred = model.predict(X)

            # Save minimal info for future predictions (no need to keep the full model object)
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
                "decimals": int(decimals),
            }

        except Exception as e:
            st.error(f"Error: {e}")
            return

    # If we have a fitted model in session state, show results & tools
    if st.session_state.simple_reg.get("fitted"):
        x = st.session_state.simple_reg["x"]
        y = st.session_state.simple_reg["y"]
        y_pred = st.session_state.simple_reg["y_pred"]
        residuals = st.session_state.simple_reg["residuals"]
        fitted_vals = st.session_state.simple_reg["fitted_vals"]
        intercept = st.session_state.simple_reg["intercept"]
        slope = st.session_state.simple_reg["slope"]
        decimals = st.session_state.simple_reg.get("decimals", 4)

        # Correlation analysis
        corr_coef, p_value = stats.pearsonr(x, y)
        st.subheader("📊 Correlation Analysis")
        st.write(f"**Correlation Coefficient (r):** {round(corr_coef, decimals)}")
        st.write(f"**P-value:** {round(p_value, decimals)}")

        # Regression summary
        st.subheader("📄 Regression Summary")
        st.text(st.session_state.simple_reg["summary"])

        # Scatter plot with regression line
        st.subheader("📈 Scatter Plot with Regression Line")
        order = np.argsort(x)
        fig1, ax1 = plt.subplots()
        ax1.scatter(x, y, label="Observed")
        ax1.plot(x[order], y_pred[order], label="Regression Line")
        for i in range(len(x)):
            ax1.text(x[i], y[i], f"({x[i]}, {y[i]})", fontsize=8, ha="right")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.grid(True)
        ax1.legend()
        st.pyplot(fig1)

        # Residual plot
        st.subheader("📉 Residual Plot (Residuals vs Fitted Values)")
        fig2, ax2 = plt.subplots()
        ax2.scatter(fitted_vals, residuals)
        ax2.axhline(y=0, linestyle="--")
        ax2.set_xlabel("Fitted Values")
        ax2.set_ylabel("Residuals")
        ax2.grid(True)
        st.pyplot(fig2)

        # Prediction & residual calculation (no extra button, works across reruns)
        st.subheader("🔮 Predict New Value")
        st.caption(f"Regression equation:  ŷ = {round(intercept, decimals)} + {round(slope, decimals)}·x")

        new_x = st.number_input("Enter a new x to predict y:", value=0.0, format="%.6f", key="simple_new_x")
        y_hat = intercept + slope * new_x
        st.success(f"Predicted y for x = {new_x}: **{round(y_hat, decimals)}**")

        actual_y_str = st.text_input("Optional: enter the actual y value to compute residual:", key="simple_actual_y")
        if actual_y_str.strip():
            try:
                actual_y = float(actual_y_str)
                residual_new = actual_y - y_hat
                st.info(f"Residual (Actual − Predicted) = **{round(residual_new, decimals)}**")
            except ValueError:
                st.error("Please enter a valid numeric value for the actual y.")

        # Reset/clear
        if st.button("🧹 Clear Fitted Model"):
            st.session_state.simple_reg = {}

# -----------------------------
# MULTIPLE REGRESSION
# -----------------------------
def run_multiple_regression_tool():
    st.header("👨‍🏫 Multiple Regression")

    st.markdown("### Data Input")
    uploaded_file = st.file_uploader("Upload CSV or Excel file (optional)", type=["csv", "xlsx"])
    raw_matrix = st.text_area(
        "Or enter your data as a matrix (rows = observations, columns = independent variables; last column = y):\n"
        "Example:\n5, 7, 2\n6, 8, 3\n7, 9, 4\n8, 10, 5"
    )
    decimals = st.number_input("Decimal places for output", min_value=0, max_value=10, value=4, step=1)

    if st.button("👨‍💻 Run Multiple Regression"):
        try:
            if uploaded_file is not None:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.write("Data Preview:")
                st.dataframe(df.head())

                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(num_cols) < 2:
                    st.error("File must contain at least 2 numeric columns (predictors + response).")
                    return

                # Let user pick response column (default last numeric col)
                y_col = st.selectbox("Select Y (dependent) column", num_cols, index=len(num_cols) - 1, key="multi_y_col")
                X_cols = [c for c in num_cols if c != y_col]
                X = df[X_cols].dropna()
                y = df.loc[X.index, y_col].to_numpy(dtype=float)
                X = X.to_numpy(dtype=float)

            else:
                # Parse matrix text
                rows = [r for r in raw_matrix.strip().split("\n") if r.strip()]
                data = [list(map(float, r.split(","))) for r in rows]
                data = np.array(data, dtype=float)
                if data.shape[1] < 2:
                    st.error("Need at least 1 independent variable and 1 dependent variable (>=2 columns).")
                    return
                X = data[:, :-1]
                y = data[:, -1]

            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            st.subheader("Regression Summary")
            st.text(model.summary())

            st.subheader("Residual Plot (Residuals vs Fitted Values)")
            residuals = model.resid
            fitted = model.fittedvalues
            fig, ax = plt.subplots()
            ax.scatter(fitted, residuals)
            ax.axhline(y=0, linestyle="--")
            ax.set_xlabel("Fitted Values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residual Plot")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")
