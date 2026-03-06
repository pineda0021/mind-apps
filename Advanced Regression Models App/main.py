import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, boxcox
from scipy.special import inv_boxcox

st.set_page_config(layout="wide")
st.title("General Linear Model Laboratory")

# ---------------------------
# 1. DATA UPLOAD
# ---------------------------
st.header("1. Upload Data")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

if df is not None:

    # ---------------------------
    # 2. VARIABLE SELECTION
    # ---------------------------
    st.header("2. Variable Selection")

    response = st.selectbox("Response Variable", df.columns)
    predictors = st.multiselect(
        "Predictor Variables",
        [col for col in df.columns if col != response]
    )

    categorical_vars = st.multiselect(
        "Categorical Variables (Factors)",
        predictors
    )

    for col in categorical_vars:
        df[col] = df[col].astype("category")

    if predictors:

        formula = response + " ~ " + " + ".join(predictors)

        # ---------------------------
        # 3. MODEL TYPE
        # ---------------------------
        st.header("3. Model Type")

        model_type = st.radio(
            "Choose Model",
            ["OLS (Gaussian)", "Box-Cox OLS", "Gamma GLM (log link)"]
        )

        # ---------------------------
        # 4. FIT MODELS
        # ---------------------------
        if model_type == "OLS (Gaussian)":
            model = smf.ols(formula=formula, data=df).fit()

        elif model_type == "Box-Cox OLS":

            if (df[response] <= 0).any():
                st.error("Box-Cox requires strictly positive response.")
                st.stop()

            y_transformed, lambda_bc = boxcox(df[response])
            df["y_boxcox"] = y_transformed

            formula_bc = "y_boxcox ~ " + " + ".join(predictors)
            model = smf.ols(formula=formula_bc, data=df).fit()

            st.write(f"Estimated λ (Box-Cox): {lambda_bc:.4f}")

        elif model_type == "Gamma GLM (log link)":
            model = smf.glm(
                formula=formula,
                data=df,
                family=sm.families.Gamma(sm.families.links.log())
            ).fit()

        # ---------------------------
        # 5. MODEL SUMMARY
        # ---------------------------
        st.header("4. Model Summary")
        st.text(model.summary())

        st.subheader("Model Fit Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("AIC", round(model.aic, 2))
        col2.metric("BIC", round(model.bic, 2))
        col3.metric("Log-Likelihood", round(model.llf, 2))

        # ---------------------------
        # 6. DIAGNOSTICS
        # ---------------------------
        st.header("5. Diagnostics")

        residuals = model.resid_response if hasattr(model, "resid_response") else model.resid

        fig_resid = px.scatter(
            x=model.fittedvalues,
            y=residuals,
            labels={"x": "Fitted", "y": "Residuals"},
            title="Residuals vs Fitted"
        )
        st.plotly_chart(fig_resid)

        # Normality test (OLS only)
        if model_type != "Gamma GLM (log link)":
            stat, p = shapiro(residuals)
            st.write(f"Shapiro-Wilk p-value: {p:.4f}")

        # ---------------------------
        # 7. MODEL COMPARISON OPTION
        # ---------------------------
        st.header("6. Model Comparison (OLS vs Gamma)")

        if st.button("Fit Both Models for Comparison"):

            ols_model = smf.ols(formula=formula, data=df).fit()
            gamma_model = smf.glm(
                formula=formula,
                data=df,
                family=sm.families.Gamma(sm.families.links.log())
            ).fit()

            comparison_df = pd.DataFrame({
                "Model": ["OLS", "Gamma"],
                "AIC": [ols_model.aic, gamma_model.aic],
                "BIC": [ols_model.bic, gamma_model.bic],
                "LogLik": [ols_model.llf, gamma_model.llf]
            })

            st.dataframe(comparison_df)

        # ---------------------------
        # 8. PREDICTION
        # ---------------------------
        st.header("7. Prediction")

        input_dict = {}

        for var in predictors:
            if var in categorical_vars:
                input_dict[var] = st.selectbox(
                    f"{var}", df[var].cat.categories
                )
            else:
                input_dict[var] = st.number_input(
                    f"{var}",
                    value=float(df[var].mean())
                )

        if st.button("Predict"):
            new_df = pd.DataFrame([input_dict])
            pred = model.predict(new_df)[0]

            if model_type == "Box-Cox OLS":
                pred = inv_boxcox(pred, lambda_bc)

            st.success(f"Predicted {response}: {pred:.4f}")
