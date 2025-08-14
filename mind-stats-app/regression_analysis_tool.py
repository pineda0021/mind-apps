import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm


def run_simple_regression_tool():
    st.header("👨‍🏫 Simple Linear Regression")

    # Data input
    st.markdown("### Data Input")
    uploaded_file = st.file_uploader("Upload CSV or Excel file (optional)", type=["csv", "xlsx"])
    y_input = st.text_area("Or enter dependent variable (y) values, separated by commas:")
    x_input = st.text_area("Or enter independent variable (x) values, separated by commas:")

    if st.button("Run Simple Regression"):
        try:
            # Read data
            if uploaded_file is not None:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(df.head())
                if df.shape[1] < 2:
                    st.error("The file must contain at least 2 columns: x and y.")
                    return
                x = df.iloc[:, 0].values
                y = df.iloc[:, 1].values
            else:
                y = np.array(list(map(float, y_input.replace(',', ' ').split())))
                x = np.array(list(map(float, x_input.replace(',', ' ').split())))
                if len(y) != len(x):
                    st.error("x and y must have the same length.")
                    return

            # Regression model
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            y_pred = model.predict(X)

            # Correlation analysis
            corr_coef, p_value = stats.pearsonr(x, y)
            st.subheader("📊 Correlation Analysis")
            st.write(f"**Correlation Coefficient:** {round(corr_coef, 4)}")
            st.write(f"**P-value:** {round(p_value, 4)}")

            # Regression summary
            st.subheader("📄 Regression Summary")
            st.text(model.summary())

            # Scatter plot with regression line
            st.subheader("📈 Scatter Plot with Regression Line")
            fig1, ax1 = plt.subplots()
            ax1.scatter(x, y, color='blue', label='Observed')
            ax1.plot(x, y_pred, color='red', label='Regression Line')
            for i in range(len(x)):
                ax1.text(x[i], y[i], f"({x[i]}, {y[i]})", fontsize=8, ha='right')
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.legend()
            ax1.grid(True)
            st.pyplot(fig1)

            # Residual plot
            st.subheader("📉 Residual Plot (Residuals vs Fitted Values)")
            residuals = model.resid
            fitted = model.fittedvalues
            fig2, ax2 = plt.subplots()
            ax2.scatter(fitted, residuals, color='purple')
            ax2.axhline(y=0, color='gray', linestyle='--')
            ax2.set_xlabel("Fitted Values")
            ax2.set_ylabel("Residuals")
            ax2.grid(True)
            st.pyplot(fig2)

            # Prediction & residual calculation
            st.subheader("🔮 Predict New Value")
            new_x = st.number_input("Enter a new x value to predict y:", value=0.0, format="%.4f")
            if st.button("Predict y"):
                pred_y = model.predict([[1, new_x]])[0]
                st.success(f"Predicted y for x = {new_x}: **{round(pred_y, 4)}**")
                actual_y_input = st.text_input("Enter actual y value (optional):")
                if actual_y_input:
                    actual_y = float(actual_y_input)
                    residual = actual_y - pred_y
                    st.info(f"Residual = {round(residual, 4)}")

        except Exception as e:
            st.error(f"Error: {e}")



def run_multiple_regression_tool():
    st.header("👨‍🏫 Multiple Regression")

    st.markdown("### Data Input")
    uploaded_file = st.file_uploader("Upload CSV or Excel file (optional)", type=["csv", "xlsx"])
    raw_matrix = st.text_area(
        "Or enter your data as a matrix (rows = observations, columns = independent variables; last column = y):\n"
        "Example:\n5, 7, 2\n6, 8, 3\n7, 9, 4\n8, 10, 5"
    )

    if st.button("👨‍💻 Run Multiple Regression"):
        try:
            if uploaded_file is not None:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(df.head())
                if df.shape[1] < 2:
                    st.error("File must contain at least 2 columns: independent variables and dependent variable.")
                    return
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            else:
                rows = raw_matrix.strip().split('\n')
                data = [list(map(float, row.strip().split(','))) for row in rows]
                data = np.array(data)
                if data.shape[1] < 2:
                    st.error("Need at least 1 independent variable and 1 dependent variable.")
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
            ax.scatter(fitted, residuals, color='purple')
            ax.axhline(y=0, color='gray', linestyle='--')
            ax.set_xlabel("Fitted Values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residual Plot")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {e}")

