import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import shapiro, chi2


def run():

    st.title("📘 General Linear Regression Model")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ======================================================
    # 1️⃣ Model Specification
    # ======================================================

    st.header("1️⃣ Model Specification")

    response = st.selectbox("Select Response Variable (Y)", df.columns)

    predictors = st.multiselect(
        "Select Predictor Variables (X)",
        [col for col in df.columns if col != response]
    )

    if not predictors:
        return

    categorical_vars = st.multiselect("Select Categorical Predictors", predictors)

    reference_dict = {}

    for col in categorical_vars:
        df[col] = df[col].astype("category")
        ref = st.selectbox(
            f"Reference level for {col}",
            df[col].cat.categories,
            key=f"ref_{col}"
        )
        reference_dict[col] = ref

    terms = []
    for var in predictors:
        if var in categorical_vars:
            ref = reference_dict[var]
            terms.append(f'C({var}, Treatment(reference="{ref}"))')
        else:
            terms.append(var)

    formula_original = response + " ~ " + " + ".join(terms)
    st.code(formula_original)

    # ======================================================
    # 2️⃣ Transformation (EXPLICIT 1 - 1/y)
    # ======================================================

    st.header("2️⃣ Transformation: 1 - (1 / y)")

    df_model = df.copy()
    df_model[response] = pd.to_numeric(df_model[response], errors="coerce")

    if (df_model[response] <= 0).any():
        st.warning("Response must be strictly positive.")
        return

    if st.checkbox("Apply Transformation (1 - 1/y)"):

        transformed_response = response + "_tr"

        # 🔴 Explicit formula — no λ confusion
        df_model[transformed_response] = 1 - (1 / df_model[response])

        # Drop missing rows
        df_model = df_model.dropna(subset=[transformed_response] + predictors)

        st.subheader("Original vs Transformed Response")
        st.dataframe(df_model[[response, transformed_response]].head(20))

        # Normality of transformed response
        if len(df_model[transformed_response]) >= 3:
            stat_y, p_y = shapiro(df_model[transformed_response])
            st.subheader("Normality Test (Transformed Response)")
            st.write(f"Shapiro-Wilk Statistic: {stat_y:.4f}")
            st.write(f"p-value: {p_y:.4f}")

            if p_y <= 0.05:
                st.warning("Transformed response is NOT normally distributed.")
            else:
                st.success("Transformed response appears normally distributed.")

    else:
        return

    # ======================================================
    # 3️⃣ Model Fitting (Original GLM Preserved)
    # ======================================================

    st.header("3️⃣ Model Fitting")

    formula_final = transformed_response + " ~ " + " + ".join(terms)

    model = smf.glm(
        formula=formula_final,
        data=df_model,
        family=sm.families.Gaussian()
    ).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # 4️⃣ Residual Normality
    # ======================================================

    resid = model.resid_response.dropna()

    if len(resid) >= 3:
        stat_resid, p_resid = shapiro(resid)
        st.subheader("Normality Test (Model Residuals)")
        st.write(f"Shapiro-Wilk Statistic: {stat_resid:.4f}")
        st.write(f"p-value: {p_resid:.4f}")

    # ======================================================
    # 5️⃣ Deviance Test
    # ======================================================

    null_model = smf.glm(
        formula=transformed_response + " ~ 1",
        data=df_model,
        family=sm.families.Gaussian()
    ).fit()

    deviance = -2 * (null_model.llf - model.llf)
    df_diff = model.df_model - null_model.df_model
    p_value = 1 - chi2.cdf(deviance, df_diff)

    st.subheader("Likelihood Ratio (Deviance) Test")
    st.write(f"Deviance: {deviance:.4f}")
    st.write(f"Degrees of Freedom: {df_diff}")
    st.write(f"p-value: {p_value:.4f}")

    # ======================================================
    # 6️⃣ Interpretation
    # ======================================================

    st.header("4️⃣ Interpretation of Coefficients")

    for term in model.params.index:

        coef = model.params[term]
        pval = model.pvalues[term]

        if term == "Intercept":
            interpretation = (
                f"When all predictors are at reference levels or zero, "
                f"the expected transformed response is {coef:.4f}."
            )
        else:
            interpretation = (
                f"A one-unit increase in '{term}' changes the expected "
                f"transformed response by {coef:.4f}, holding other variables constant."
            )

        significance = (
            "Statistically significant."
            if pval <= 0.05
            else "Not statistically significant."
        )

        st.markdown(f"**{term}**  \n"
                    f"- Coefficient: {coef:.4f}  \n"
                    f"- p-value: {pval:.4f}  \n"
                    f"- {interpretation}  \n"
                    f"- {significance}")


if __name__ == "__main__":
    run()
