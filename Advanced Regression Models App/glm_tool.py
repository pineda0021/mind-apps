import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import shapiro

def run():

    st.header("Gaussian Linear Model (OLS)")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="glm")

    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    response = st.selectbox("Response Variable", df.columns)
    predictors = st.multiselect(
        "Predictor Variables",
        [c for c in df.columns if c != response]
    )

    if not predictors:
        return

    formula = response + " ~ " + " + ".join(predictors)

    model = smf.ols(formula=formula, data=df).fit()

    st.text(model.summary())

    residuals = model.resid
    stat, p = shapiro(residuals)
    st.write(f"Shapiro-Wilk p-value: {p:.4f}")
