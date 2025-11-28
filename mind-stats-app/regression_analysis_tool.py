# ==========================================================
# regression_analysis_tool.py
# Updated for Universal Readability (Dark & Light Mode Safe)
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# ==========================================================
# UNIVERSAL READABILITY COLORS
# ==========================================================
BACKGROUND = "#2B2B2B"
TEXT = "white"
ACCENT = "#4da3ff"

plt.rcParams.update({
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.edgecolor": TEXT,
})

# ==========================================================
# Helper Step Box
# ==========================================================
def step_box(text):
    st.markdown(
        f"""
        <div style="
            background-color:{BACKGROUND};
            padding:12px;
            border-radius:10px;
            border-left:6px solid {ACCENT};
            margin-bottom:12px;">
            <p style="color:{TEXT};margin:0;font-weight:bold;">{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================
# SIMPLE LINEAR REGRESSION
# ==========================================================
def run_simple_regression_tool():
    # MAIN HEADER (Option B)
    st.markdown(
        f"<h1 style='color:{TEXT};'>🧠 MIND: Regression Analysis — Universal Readability</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<h2 style='color:{TEXT};'>📈 Simple Linear Regression</h2>",
        unsafe_allow_html=True,
    )

    if "simple_reg" not in st.session_state:
        st.session_state.simple_reg = {}

    st.markdown(f"<p style='color:{TEXT};'>This tool estimates the best-fitting line using the Ordinary Least Squares (OLS) method.</p>", unsafe_allow_html=True)
    st.latex(r"\hat{y} = b_0 + b_1x")

    # --- Inputs ---
    uploaded_file = st.file_uploader("📂 Upload CSV or Excel file (optional)", type=["csv", "xlsx"])
    y_input = st.text_area("Or enter dependent variable (y) values (comma-separated):")
    x_input = st.text_area("Or enter independent variable (x) values (comma-separated):")
    decimals = st.number_input("Decimal places for rounding", 0, 10, 4)

    df = None
    x_col = y_col = None

    # --- File Upload ---
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

            st.markdown(f"<p style='color:{TEXT};'>📋 Data Preview</p>", unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)

            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) < 2:
                st.error("The file must contain at least two numeric columns.")
                return

            c1, c2 = st.columns(2)
            with c1:
                x_col = st.selectbox("Select X (independent variable)", num_cols)
            with c2:
                y_col = st.selectbox("Select Y (dependent variable)", num_cols, index=1)
        except Exception as e:
            st.error(f"⚠️ File error: {e}")
            return

    # --- Run Regression ---
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

    # =============================
    # DISPLAY RESULTS
    # =============================
    if st.session_state.simple_reg.get("fitted"):
        x = st.session_state.simple_reg["x"]
        y = st.session_state.simple_reg["y"]
        y_pred = st.session_state.simple_reg["y_pred"]
        residuals = st.session_state.simple_reg["residuals"]
        intercept = st.session_state.simple_reg["intercept"]
        slope = st.session_state.simple_reg["slope"]
        decimals = st.session_state.simple_reg["decimals"]

        st.markdown("---")

        # Hypotheses
        st.markdown(f"<h3 style='color:{TEXT};'>🧩 Hypotheses</h3>", unsafe_allow_html=True)
        st.latex(r"H_0: \beta_1 = 0")
        st.latex(r"H_a: \beta_1 \neq 0")

        # Step 1: Correlation
        step_box("Step 1: Compute correlation and assess linearity")
        corr_coef, p_val = stats.pearsonr(x, y)
        r2 = corr_coef ** 2

        summary_df = pd.DataFrame({
            "Statistic": ["r", "r²", "p-value", "Slope (b₁)", "Intercept (b₀)"],
            "Value": [
                round(corr_coef, decimals),
                round(r2, decimals),
                round(p_val, decimals),
                round(slope, decimals),
                round(intercept, decimals),
            ],
        })
        st.dataframe(summary_df, use_container_width=True)

        # Step 2: Regression Equation
        step_box("Step 2: Compute regression coefficients")
        st.latex(r"\hat{y} = b_0 + b_1x")
        st.markdown(
            f"<p style='color:{TEXT};'>Estimated model: ŷ = {round(intercept, decimals)} + {round(slope, decimals)}·x</p>",
            unsafe_allow_html=True,
        )
        st.text(st.session_state.simple_reg["summary"])

        # Step 3: Regression Line Plot
        step_box("Step 3: Visualize regression line")
        order = np.argsort(x)
        fig, ax = plt.subplots(facecolor=BACKGROUND)
        fig.patch.set_facecolor(BACKGROUND)

        ax.scatter(x, y, color=ACCENT, label="Observed Data")
        ax.plot(x[order], y_pred[order], color="red", label="Regression Line")

        ax.set_xlabel("X (Independent Variable)")
        ax.set_ylabel("Y (Dependent Variable)")
        ax.legend()
        ax.grid(True, color="#555555")

        st.pyplot(fig)

        # Step 4: Residual Plot
        step_box("Step 4: Analyze residuals for randomness")
        fig2, ax2 = plt.subplots(facecolor=BACKGROUND)
        fig2.patch.set_facecolor(BACKGROUND)

        ax2.scatter(st.session_state.simple_reg["fitted_vals"], residuals, color=ACCENT)
        ax2.axhline(y=0, linestyle="--", color="red")

        ax2.set_xlabel("Fitted Values")
        ax2.set_ylabel("Residuals")
        ax2.grid(True, color="#555555")

        st.pyplot(fig2)

        # Step 5: Prediction
        step_box("Step 5: Make predictions and interpret results")
        st.latex(r"\hat{y} = b_0 + b_1x")

        new_x = st.number_input("Enter a new x value:", value=0.0, format="%.4f")
        y_hat = intercept + slope * new_x
        st.success(f"Predicted ŷ = {round(y_hat, decimals)}")

        actual_y = st.text_input("Optional: Enter actual y value to compute residual:")
        if actual_y.strip():
            try:
                residual_new = float(actual_y) - y_hat
                st.info(f"Residual = {round(residual_new, decimals)}")
            except ValueError:
                st.error("Please enter a valid numeric value.")

        if st.button("🧹 Clear Fitted Model"):
            st.session_state.simple_reg = {}

# ==========================================================
# MULTIPLE REGRESSION
# ==========================================================
def run_multiple_regression_tool():

    st.markdown(
        f"<h2 style='color:{TEXT};'>📊 Multiple Linear Regression</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<p style='color:{TEXT};'>This tool estimates a multiple regression model using the OLS method.</p>",
        unsafe_allow_html=True,
    )

    st.latex(r"\hat{y} = b_0 + b_1x_1 + b_2x_2 + \dots + b_kx_k")

    uploaded_file = st.file_uploader("📂 Upload CSV or Excel file", type=["csv", "xlsx"])
    raw_matrix = st.text_area(
        "Or enter matrix data (each row = observation, last column = y):\n"
        "Example:\n5, 7, 2\n6, 8, 3\n7, 9, 4"
    )
    decimals = st.number_input("Decimal places for output", 0, 10, 4)

    if st.button("👨‍💻 Run Multiple Regression"):
        try:
            # --- File Upload ---
            if uploaded_file:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

                st.markdown(f"<p style='color:{TEXT};'>📋 Data Preview</p>", unsafe_allow_html=True)
                st.dataframe(df.head(), use_container_width=True)

                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(num_cols) < 2:
                    st.error("File must contain at least one predictor and one dependent variable.")
                    return

                y_col = st.selectbox("Select dependent variable (Y)", num_cols, index=len(num_cols) - 1)
                X_cols = [col for col in num_cols if col != y_col]

                X = df[X_cols].dropna().to_numpy()
                y = df.loc[df[X_cols].dropna().index, y_col].to_numpy()

            # --- Manual Matrix Input ---
            else:
                rows = [r.strip() for r in raw_matrix.strip().split("\n") if r.strip()]
                data = np.array([list(map(float, r.split(","))) for r in rows])

                if data.shape[1] < 2:
                    st.error("Matrix must have at least 2 columns (predictor(s) + response y).")
                    return

                X = data[:, :-1]
                y = data[:, -1]

            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            st.markdown(f"<h3 style='color:{TEXT};'>📄 Regression Summary</h3>", unsafe_allow_html=True)
            st.text(model.summary())

            # Residual Plot
            st.markdown(f"<h3 style='color:{TEXT};'>📉 Residual Plot</h3>", unsafe_allow_html=True)
            residuals = model.resid
            fitted = model.fittedvalues

            fig, ax = plt.subplots(facecolor=BACKGROUND)
            fig.patch.set_facecolor(BACKGROUND)

            ax.scatter(fitted, residuals, color=ACCENT)
            ax.axhline(0, linestyle="--", color="red")

            ax.set_xlabel("Fitted Values")
            ax.set_ylabel("Residuals")
            ax.grid(True, color="#555555")

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================================
# End of Module
# ==========================================================


