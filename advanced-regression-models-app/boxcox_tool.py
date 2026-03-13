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
        -1.0: {"name": "Inverse (Reciprocal)"},
        -0.5: {"name": "Inverse Square Root"},
        0.0: {"name": "Natural Log"},
        0.5: {"name": "Square Root"},
        1.0: {"name": "Linear"},
        2.0: {"name": "Square"}
    }

    return transformations.get(
        lam_rounded,
        {"name": "Custom λ"}
    )


# ======================================================
# Equation Builder
# ======================================================

def build_equation(model, response_name):

    params = model.params
    equation = f"\\widehat{{\\mathbb{{E}}}}({response_name}) = {round(params['Intercept'],4)}"

    for name in params.index:

        if name == "Intercept":
            continue

        coef = round(params[name], 4)
        sign = "+" if coef >= 0 else "-"

        if name.startswith("C(") and "T." in name:
            var_name = name.split("[")[0]
            var_name = var_name.replace("C(", "").split(",")[0]
            level = name.split("T.")[1].rstrip("]")
            equation += f" {sign} {abs(coef)} D_{{{var_name}={level}}}"
        else:
            equation += f" {sign} {abs(coef)} \\cdot {name}"

    return equation


# ======================================================
# APP
# ======================================================

def run():

    st.title("📘 Gaussian GLM with Box-Cox Transformation")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ======================================================
    # Variable Selection
    # ======================================================

    st.header("1️⃣ Select Variables")

    response = st.selectbox("Select Response Variable (Y)", df.columns)

    predictors = st.multiselect(
        "Select Predictor Variables (X)",
        [col for col in df.columns if col != response]
    )

    if not predictors:
        return

    categorical_vars = st.multiselect(
        "Select Categorical Variables",
        predictors
    )

    terms = []
    for var in predictors:
        if var in categorical_vars:
            df[var] = df[var].astype("category")
            terms.append(f"C({var})")
        else:
            terms.append(var)

    formula_original = response + " ~ " + " + ".join(terms)
    st.code(formula_original)

    # ======================================================
    # Box-Cox Transformation
    # ======================================================

    st.header("2️⃣ Box-Cox Transformation")

    if not pd.api.types.is_numeric_dtype(df[response]):
        st.error("Response must be numeric.")
        return

    if (df[response] <= 0).any():
        st.error("Box-Cox requires strictly positive response values.")
        return

    chosen_lambda = st.number_input("Enter λ", value=0.0, step=0.1)

    info = transformation_info(chosen_lambda)
    st.write(f"Transformation: **{info['name']}**")

    lam = float(chosen_lambda)

    df_model = df.copy()
    y = pd.to_numeric(df_model[response], errors="coerce")

    transformed_response = response + "_tr"

    if lam == 0:
        df_model[transformed_response] = np.log(y)
    else:
        df_model[transformed_response] = (y**lam - 1) / lam

    # ======================================================
    # Normality Test
    # ======================================================

    st.header("3️⃣ Normality Check (Transformed Response)")

    df_fit = df_model[[transformed_response] + predictors].dropna()
    y_trans_clean = df_fit[transformed_response]

    qq_fig = sm.qqplot(y_trans_clean, line='s')
    st.pyplot(qq_fig.figure)

    stat, p = shapiro(y_trans_clean)

    st.write(f"Shapiro-Wilk Statistic: {stat:.4f}")
    st.write(f"p-value: {p:.4f}")

    # ======================================================
    # Fit GLM
    # ======================================================

    st.header("4️⃣ Fit GLM on Transformed Response")

    formula_transformed = transformed_response + " ~ " + " + ".join(terms)

    model = smf.glm(
        formula=formula_transformed,
        data=df_fit,
        family=sm.families.Gaussian()
    ).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # Fitted Equation
    # ======================================================

    st.subheader("Fitted Regression Equation (Transformed Scale)")
    st.latex(build_equation(model, transformed_response))

    # ======================================================
    # Coefficient Interpretation
    # ======================================================

    st.subheader("Coefficient Interpretation (Transformed Scale)")

    for name in model.params.index:

        coef = round(model.params[name], 4)
        pval = model.pvalues[name]

        if name == "Intercept":
            st.markdown(
                f"**Intercept ({coef})**: Estimated mean of the transformed response "
                f"when all predictors are at reference levels or zero."
            )

        elif name.startswith("C(") and "T." in name:
            var_name = name.split("[")[0]
            var_name = var_name.replace("C(", "").split(",")[0]
            level = name.split("T.")[1].rstrip("]")

            st.markdown(
                f"**{var_name} = {level} (β = {coef})**: "
                f"Mean difference in the transformed response compared to the reference level, "
                f"holding other variables constant. "
                f"{'Statistically significant.' if pval < 0.05 else 'Not statistically significant.'}"
            )

        else:
            st.markdown(
                f"**{name} (β = {coef})**: For each one-unit increase in {name}, "
                f"the transformed response changes by {coef} units, "
                f"holding other predictors constant. "
                f"{'Statistically significant.' if pval < 0.05 else 'Not statistically significant.'}"
            )

    # ======================================================
    # Model Fit Statistics
    # ======================================================

    st.header("5️⃣ Model Fit Evaluation (Transformed Scale)")

    n = model.nobs
    k = model.df_model + 1

    residual_deviance = model.deviance
    null_deviance = model.null_deviance
    aic = model.aic

    if (n - k - 1) > 0:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = float("nan")

    sigma_hat = np.sqrt(model.scale)
    rmse = np.sqrt(np.mean(model.resid_response**2))

    st.write(f"Null Deviance: {null_deviance:.4f}")
    st.write(f"Residual Deviance: {residual_deviance:.4f}")
    st.write(f"AIC: {aic:.4f}")
    st.write(f"AICc: {aicc:.4f}")
    st.write(f"Residual SD (σ̂): {sigma_hat:.4f}")
    st.write(f"RMSE (Transformed Scale): {rmse:.4f}")


if __name__ == "__main__":
    run()
