import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import boxcox
from scipy.special import inv_boxcox

def run():

    st.header("Box-Cox Transformation + OLS")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="boxcox")

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

    if (df[response] <= 0).any():
        st.error("Response must be strictly positive.")
        return

    y_transformed, lambda_bc = boxcox(df[response])
    df["y_bc"] = y_transformed

    formula = "y_bc ~ " + " + ".join(predictors)
    model = smf.ols(formula=formula, data=df).fit()

    st.write(f"Estimated λ: {lambda_bc:.4f}")
    st.text(model.summary())
