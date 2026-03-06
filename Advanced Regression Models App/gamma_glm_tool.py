import streamlit as st
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def run():

    st.header("Gamma GLM (Log Link)")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="gamma")

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

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Gamma(sm.families.links.log())
    ).fit()

    st.text(model.summary())
