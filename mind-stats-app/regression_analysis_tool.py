import streamlit as st
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def run_simple_regression_tool():
    st.header("Simple Linear Regression Tool")
    
    y_input = st.text_area("Enter dependent variable (y) values, separated by commas:")
    x_input = st.text_area("Enter independent variable (x) values, separated by commas:")
    
    if st.button("Run Simple Regression"):
        try:
            y = np.array(list(map(float, y_input.replace(',', ' ').split())))
            x = np.array(list(map(float, x_input.replace(',', ' ').split())))
            if len(y) != len(x):
                st.error("x and y must have the same length.")
                return
            
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            
            st.subheader("Regression Summary")
            st.text(model.summary())
            
            fig, ax = plt.subplots()
            ax.scatter(x, y, color='blue', label='Observed')
            ax.plot(x, model.predict(X), color='red', label='Fitted Line')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Scatter Plot with Regression Line")
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {e}")

def run_multiple_regression_tool():
    st.header("Multiple Regression Tool (TI-84 Style Input)")
    
    st.markdown("""
    Enter your data as a matrix (rows = observations, columns = independent variables; last column can be y if you like):
    
    Example for 4 observations and 3 independent variables:
    ```
    5, 7, 2
    6, 8, 3
    7, 9, 4
    8, 10, 5
    ```
    """)
    
    raw_matrix = st.text_area("Enter data matrix (comma-separated, new line = new observation):")
    
    if st.button("Run Multiple Regression"):
        try:
            rows = raw_matrix.strip().split('\n')
            data = [list(map(float, row.strip().split(','))) for row in rows]
            data = np.array(data)
            
            if data.shape[1] < 2:
                st.error("Need at least 1 independent variable and 1 dependent variable.")
                return
            
            X = data[:, :-1]  # all columns except last
            y = data[:, -1]   # last column = dependent variable
            
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
