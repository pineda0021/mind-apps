pip install statsmodels scipy
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro

st.set_page_config(layout="wide")
st.title("General Linear Regression Model App")

# -----------------------------
# 1. DATA UPLOAD
# -----------------------------
st.header("1. Upload Data")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
google_sheet_url = st.text_input("Or paste Google Sheets share link (optional)")

df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)

elif google_sheet_url:
    try:
        sheet_id = google_sheet_url.split('/d/')[1].split('/')[0]
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        df = pd.read_csv(csv_url)
        st.success("Google Sheet loaded successfully.")
    except:
        st.error("Could not load Google Sheet. Check sharing permissions.")

if df is not None:
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # -----------------------------
    # 2. VARIABLE SELECTION
    # -----------------------------
    st.header("2. Select Variables")

    response = st.selectbox("Select Response Variable (Y)", df.columns)
    predictors = st.multiselect("Select Predictor Variables (X)", 
                                 [col for col in df.columns if col != response])

    categorical_vars = st.multiselect("Select Categorical Variables (Factors)", predictors)

    # Convert selected categorical variables
    for col in categorical_vars:
        df[col] = df[col].astype('category')

    # -----------------------------
    # 3. EXPLORATORY DATA ANALYSIS
    # -----------------------------
    st.header("3. Data Exploration")

    st.subheader("Histogram of Response Variable")
    fig = px.histogram(df, x=response, title=f"Distribution of {response}")
    st.plotly_chart(fig)

    # Shapiro-Wilk Test
    st.subheader("Normality Test (Shapiro-Wilk)")

    if df[response].dtype != "category":
        stat, p = shapiro(df[response].dropna())
        st.write(f"Shapiro-Wilk Statistic: {stat:.4f}")
        st.write(f"p-value: {p:.4f}")

        if p > 0.05:
            st.success("Fail to reject H₀ → Data likely normal.")
        else:
            st.warning("Reject H₀ → Data likely NOT normal.")
    else:
        st.warning("Response variable must be numeric for normality test.")

    # -----------------------------
    # 4. FIT GENERAL LINEAR MODEL
    # -----------------------------
    st.header("4. Fit General Linear Model")

    if predictors:

        # Build formula
        formula = response + " ~ " + " + ".join(predictors)

        model = smf.ols(formula=formula, data=df).fit()

        st.subheader("Model Summary")
        st.text(model.summary())

        # Model Fit Metrics
        st.subheader("Model Fit Metrics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("R-squared", round(model.rsquared, 4))
        col2.metric("Adj R-squared", round(model.rsquared_adj, 4))
        col3.metric("AIC", round(model.aic, 2))
        col4.metric("BIC", round(model.bic, 2))

        st.write(f"Log-Likelihood: {model.llf:.4f}")

        # -----------------------------
        # 5. REDUCED MODEL (Significant Predictors Only)
        # -----------------------------
        st.header("5. Reduced Model (p < 0.05)")

        significant_vars = model.pvalues[model.pvalues < 0.05].index.tolist()
        significant_vars = [var for var in significant_vars if var != "Intercept"]

        if significant_vars:
            reduced_formula = response + " ~ " + " + ".join(significant_vars)
            reduced_model = smf.ols(formula=reduced_formula, data=df).fit()

            st.write("Reduced Model Formula:")
            st.code(reduced_formula)

            st.text(reduced_model.summary())
        else:
            st.info("No statistically significant predictors at α = 0.05.")

        # -----------------------------
        # 6. PREDICTION
        # -----------------------------
        st.header("6. Prediction")

        st.write("Enter values for prediction:")

        input_dict = {}

        for var in predictors:
            if var in categorical_vars:
                input_dict[var] = st.selectbox(f"{var}", df[var].cat.categories)
            else:
                input_dict[var] = st.number_input(f"{var}", 
                                                  value=float(df[var].mean()))

        if st.button("Predict"):
            input_df = pd.DataFrame([input_dict])
            prediction = model.predict(input_df)[0]

            st.success(f"Predicted {response}: {prediction:.4f}")

        # -----------------------------
        # 7. PREDICTION VISUALIZATION
        # -----------------------------
        st.header("7. Prediction Visualization")

        predicted_vals = model.predict(df)

        fig2 = px.scatter(x=predicted_vals,
                          y=df[response],
                          labels={'x': 'Predicted', 'y': 'Actual'},
                          title="Predicted vs Actual Values")

        st.plotly_chart(fig2)

else:
    st.info("Upload a dataset to begin.")
