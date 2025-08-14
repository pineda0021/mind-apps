# regression_analysis_tool.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

def run_regression_tool():
    st.header("📊 Regression Analysis Tool")

    st.write("### Step 1: Enter Data")
    upload_file = st.file_uploader("Upload CSV file (first column = dependent, others = independent)", type=["csv","xlsx"])
    
    if upload_file:
        if upload_file.name.endswith(".csv"):
            data = pd.read_csv(upload_file)
        else:
            data = pd.read_excel(upload_file)
        st.write("Data Preview:")
        st.dataframe(data)
    else:
        y_input = st.text_area("Enter dependent variable (y), comma-separated")
        x_input = st.text_area("Enter independent variable(s) (x), comma-separated columns separated by semicolons")
        data = None
        if y_input and x_input:
            try:
                y = np.array(list(map(float, y_input.split(","))))
                X = [list(map(float, col.split(","))) for col in x_input.split(";")]
                data = pd.DataFrame({f"X{i+1}": X[i] for i in range(len(X))})
                data.insert(0, "Y", y)
                st.write("Data Preview:")
                st.dataframe(data)
            except Exception as e:
                st.error(f"Error parsing input: {e}")

    if data is not None:
        y = data.iloc[:,0]
        X = data.iloc[:,1:]

        # Step 2: Scatter Plot (only for single X)
        if X.shape[1] == 1:
            fig, ax = plt.subplots()
            ax.scatter(X.iloc[:,0], y, color='blue')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("Scatter Plot")
            st.pyplot(fig)

        # Step 3: Correlation (only if single X)
        if X.shape[1] == 1:
            corr_coef, p_val = stats.pearsonr(X.iloc[:,0], y)
            st.write(f"**Correlation Coefficient:** {round(corr_coef,4)}")
            st.write(f"**P-value:** {round(p_val,4)}")

        # Step 4: Regression Summary
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        st.write("### Linear Regression Summary")
        st.text(model.summary())

        # Step 5: Predict & Residual
        st.write("### Predict a Value")
        if X.shape[1] == 1:
            pred_x = st.number_input("Enter X value to predict Y", value=float(X.iloc[0,0]))
            pred_y = model.predict([1, pred_x])[0]
            st.write(f"Predicted Y = {round(pred_y,4)}")
            actual_y = st.text_input("Optional: Enter actual Y to compute residual")
            if actual_y:
                residual = float(actual_y) - pred_y
                st.write(f"Residual = Actual Y - Predicted Y = {round(residual,4)}")
        else:
            st.write("Residual prediction for multiple regression requires vector input; skip or extend as needed.")

        # Step 6: Residual Plot
        residuals = model.resid
        fig, ax = plt.subplots()
        if X.shape[1] == 1:
            ax.scatter(X.iloc[:,0], residuals, color='purple')
            ax.set_xlabel("X")
        else:
            ax.scatter(range(len(residuals)), residuals, color='purple')
            ax.set_xlabel("Observation")
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        st.pyplot(fig)

