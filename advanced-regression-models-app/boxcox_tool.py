import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import shapiro, chi2


# ======================================================
# Transformation Helper
# ======================================================

def transformation_info(lam):
    lam_rounded = round(lam, 1)

    transformations = {
        -2.0: {"name": "Inverse Square"},
        -1.0: {"name": "Inverse"},
        -0.5: {"name": "Inverse Square Root"},
        0.0: {"name": "Natural Log"},
        0.5: {"name": "Square Root"},
        1.0: {"name": "Linear"},
        2.0: {"name": "Square"}
    }

    return transformations.get(
        lam_rounded,
        {"name": "Custom λ Transformation"}
    )


# ======================================================
# APP
# ======================================================

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
    # 2️⃣ Box–Cox Transformation
    # ======================================================

    st.header("2️⃣ Box–Cox Transformation")

    df_model = df.copy()
    transformed_response = response
    chosen_lambda = 1.0

    y_numeric = pd.to_numeric(df[response], errors="coerce")

    if (y_numeric <= 0).any():
        st.warning("Box–Cox requires strictly positive response values.")
    else:
        chosen_lambda = st.number_input("Enter λ value", value=0.0, step=0.1)

        info = transformation_info(chosen_lambda)
        st.write(f"Selected transformation: **{info['name']}**")

        if st.checkbox("Apply Transformation"):

            transformed_response = response + "_tr"

            if np.isclose(chosen_lambda, 0):
                df_model[transformed_response] = np.log(y_numeric)
            else:
                df_model[transformed_response] = (
                    y_numeric**chosen_lambda - 1
                ) / chosen_lambda

            # Display both responses
            st.subheader("Original vs Transformed Response")
            display_df = df_model[[response, transformed_response]].dropna()
            st.dataframe(display_df.head(20))

            # Normality of transformed response
            if len(display_df) >= 3:
                stat_y, p_y = shapiro(display_df[transformed_response])

                st.subheader("Normality Test: Transformed Response")
                st.write(f"Shapiro-Wilk Statistic: {stat_y:.4f}")
                st.write(f"p-value: {p_y:.4f}")

                if p_y <= 0.05:
                    st.warning("Transformed response is NOT normally distributed.")
                else:
                    st.success("Transformed response appears normally distributed.")

    # ======================================================
    # 3️⃣ Model Fitting
    # ======================================================

    st.header("3️⃣ Model Fitting")

    formula_final = transformed_response + " ~ " + " + ".join(terms)
    model = smf.ols(formula=formula_final, data=df_model).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # 4️⃣ Residual Normality
    # ======================================================

    resid = model.resid.dropna()

    if len(resid) >= 3:
        stat_resid, p_resid = shapiro(resid)

        st.subheader("Normality Test: Model Residuals")
        st.write(f"Shapiro-Wilk Statistic: {stat_resid:.4f}")
        st.write(f"p-value: {p_resid:.4f}")

        if p_resid <= 0.05:
            st.warning("Residuals are NOT normally distributed.")
        else:
            st.success("Residuals appear normally distributed.")

    # ======================================================
    # 5️⃣ Deviance (Likelihood Ratio) Test
    # ======================================================

    null_model = smf.ols(
        formula=transformed_response + " ~ 1",
        data=df_model
    ).fit()

    deviance = -2 * (null_model.llf - model.llf)
    df_diff = model.df_model - null_model.df_model
    p_dev = 1 - chi2.cdf(deviance, df_diff)

    st.subheader("Likelihood Ratio (Deviance) Test")
    st.write(f"Deviance: {deviance:.4f}")
    st.write(f"Degrees of Freedom: {df_diff}")
    st.write(f"p-value: {p_dev:.4f}")

    if p_dev <= 0.05:
        st.success("The full model significantly improves fit over the null model.")
    else:
        st.warning("The model does NOT significantly improve over the null model.")

    # ======================================================
    # 6️⃣ Interpretation of Coefficients
    # ======================================================

    st.header("4️⃣ Interpretation of Coefficients")

    params = model.params
    pvalues = model.pvalues

    for term in params.index:

        coef = params[term]
        pval = pvalues[term]

        if term == "Intercept":
            interpretation = (
                f"When all predictors are at reference levels or zero, "
                f"the expected value of {transformed_response} is {coef:.4f}."
            )
        else:
            interpretation = (
                f"A one-unit increase in '{term}' is associated with "
                f"a {coef:.4f} change in the expected {transformed_response}, "
                f"holding other variables constant."
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
        

# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    run()
