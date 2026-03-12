import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import shapiro, boxcox_normmax, chi2


# ======================================================
# Box–Cox Helper Functions
# ======================================================

def recommend_lambda(lambda_mle):
    if -2.5 <= lambda_mle < -1.5:
        return -2.0
    elif -1.5 <= lambda_mle < -0.75:
        return -1.0
    elif -0.75 <= lambda_mle < -0.25:
        return -0.5
    elif -0.25 <= lambda_mle < 0.25:
        return 0.0
    elif 0.25 <= lambda_mle < 0.75:
        return 0.5
    elif 0.75 <= lambda_mle < 1.5:
        return 1.0
    elif 1.5 <= lambda_mle <= 2.5:
        return 2.0
    else:
        return lambda_mle


def transformation_info(lam):

    lam_rounded = round(lam,1)

    transformations = {
        -2.0: {"name": "Inverse Square",
               "formula": r"\tilde{y} = \frac{1}{2}\left(1 - \frac{1}{y^2}\right)"},
        -1.0: {"name": "Inverse (Reciprocal)",
               "formula": r"\tilde{y} = 1 - \frac{1}{y}"},
        -0.5: {"name": "Inverse Square Root",
               "formula": r"\tilde{y} = 2\left(1 - \frac{1}{\sqrt{y}}\right)"},
        0.0: {"name": "Natural Log",
              "formula": r"\tilde{y} = \ln(y)"},
        0.5: {"name": "Square Root",
              "formula": r"\tilde{y} = 2(\sqrt{y} - 1)"},
        1.0: {"name": "Linear",
              "formula": r"\tilde{y} = y - 1"},
        2.0: {"name": "Square",
              "formula": r"\tilde{y} = \frac{1}{2}(y^2 - 1)"}
    }

    return transformations.get(
        lam_rounded,
        {"name": "Custom λ",
         "formula": r"\tilde{y} = \frac{y^{\lambda}-1}{\lambda}"}
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

    y_original = pd.to_numeric(df[response], errors="coerce").dropna()

    if len(y_original) >= 3:
        _, p_orig = shapiro(y_original)
        st.write(f"Shapiro-Wilk p-value (original Y): {p_orig:.4f}")

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

            terms.append(
                f'C({var}, Treatment(reference="{ref}"))'
            )

        else:

            terms.append(var)

    formula_original = response + " ~ " + " + ".join(terms)

    st.code(formula_original)

    # ======================================================
    # 2️⃣ Box–Cox Transformation
    # ======================================================

    st.header("2️⃣ Box–Cox Transformation (Optional)")

    st.latex(r"""
    \tilde{y} =
    \begin{cases}
    \dfrac{y^{\lambda}-1}{\lambda}, & \lambda \neq 0 \\
    \ln(y), & \lambda = 0
    \end{cases}
    """)

    transformed = False
    chosen_lambda = None
    df_model = df.copy()

    y_clean = pd.to_numeric(df[response], errors="coerce").dropna()
    y_clean = y_clean[np.isfinite(y_clean)]

    can_boxcox = True

    if (y_clean <= 0).any():
        st.warning("Box–Cox requires strictly positive values.")
        can_boxcox = False

    if can_boxcox:

        lambda_mle = boxcox_normmax(y_clean, brack=(-3,3))

        st.write(f"MLE λ = {lambda_mle:.4f}")

        recommended = recommend_lambda(lambda_mle)

        st.write(f"Recommended λ = {recommended}")

        chosen_lambda = st.number_input(
            "Enter λ value",
            value=float(recommended),
            step=0.1
        )

        info = transformation_info(chosen_lambda)

        st.write(f"Selected transformation: **{info['name']}**")

        st.latex(info["formula"])

        if st.checkbox("Apply Box–Cox Transformation"):

            transformed = True

            y = pd.to_numeric(df[response], errors="coerce")

            transformed_response = response + "_tr"

            if np.isclose(chosen_lambda,0):

                df_model[transformed_response] = np.log(y)

            else:

                df_model[transformed_response] = (
                    y**chosen_lambda - 1
                ) / chosen_lambda

    # ======================================================
    # 3️⃣ Model Fitting
    # ======================================================

    st.header("3️⃣ Model Fitting")

    model_original = smf.ols(
        formula=formula_original,
        data=df
    ).fit()

    st.subheader("Original Model Summary")

    st.text(model_original.summary())

    resid_orig = model_original.resid

    _, p_resid_orig = shapiro(resid_orig)

    st.write(f"Residual Shapiro-Wilk p-value: {p_resid_orig:.4f}")

    model = model_original
    active_response = response

    if transformed and chosen_lambda is not None:

        transformed_response = response + "_tr"

        formula_transformed = (
            transformed_response + " ~ " + " + ".join(terms)
        )

        model_transformed = smf.glm(
            formula=formula_transformed,
            data=df_model,
            family=sm.families.Gaussian()
        ).fit()

        st.subheader("Transformed Model Summary")

        st.text(model_transformed.summary())

        resid_tr = model_transformed.resid_response

        _, p_trans = shapiro(resid_tr)

        st.subheader("Normality Check After Transformation")

        st.write(f"Shapiro-Wilk p-value: {p_trans:.4f}")

        if p_trans > 0.05:
            st.success("Residuals appear approximately normal.")
        else:
            st.warning("Residuals may not be normally distributed.")

        null_model = smf.glm(
            formula=transformed_response + " ~ 1",
            data=df_model,
            family=sm.families.Gaussian()
        ).fit()

        deviance = -2 * (null_model.llf - model_transformed.llf)

        df_diff = model_transformed.df_model

        p_value = 1 - chi2.cdf(deviance, df_diff)

        st.subheader("Deviance Test vs Null")

        st.write(f"Deviance: {deviance:.4f}")
        st.write(f"df: {df_diff}")
        st.write(f"p-value: {p_value:.4f}")

        model = model_transformed
        active_response = transformed_response


if __name__ == "__main__":
    run()
