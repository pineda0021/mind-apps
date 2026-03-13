import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import shapiro, chi2


def run():

    st.title("📘 Gaussian GLM with User-Specified Box-Cox Transformation")

    # ======================================================
    # DATA UPLOAD
    # ======================================================

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ======================================================
    # VARIABLE SELECTION (WITH REFERENCE LEVELS)
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
        "Select Categorical Variables (Factors)",
        predictors
    )

    reference_dict = {}

    for col in categorical_vars:
        df[col] = df[col].astype("category")
        ref = st.selectbox(
            f"Select reference level for {col}",
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
    # BOX-COX TRANSFORMATION (USER λ)
    # ======================================================

    st.header("2️⃣ Box-Cox Transformation")

    if not pd.api.types.is_numeric_dtype(df[response]):
        st.error("Response must be numeric.")
        return

    if (df[response] <= 0).any():
        st.error("Box-Cox requires strictly positive response.")
        return

    lam = st.number_input("Enter λ", value=0.0, step=0.1)

    df_model = df.copy()
    y = pd.to_numeric(df_model[response], errors="coerce")

    transformed_response = response + "_tr"

    if abs(lam) < 1e-8:
        df_model[transformed_response] = np.log(y)
    else:
        df_model[transformed_response] = (y**lam - 1) / lam

    # ======================================================
    # NORMALITY TEST ON TRANSFORMED RESPONSE
    # ======================================================

    st.header("3️⃣ Normality Check (Transformed Response)")

    df_fit = df_model[[transformed_response] + predictors].dropna()
    y_trans = df_fit[transformed_response]

    qq_fig = sm.qqplot(y_trans, line="s")
    st.pyplot(qq_fig.figure)

    stat, p = shapiro(y_trans)

    st.write(f"Shapiro-Wilk Statistic: {stat:.4f}")
    st.write(f"p-value: {p:.4f}")

    # ======================================================
    # FIT GAUSSIAN GLM (TRANSFORMED RESPONSE)
    # ======================================================

    st.header("4️⃣ Fit Gaussian GLM (Transformed Scale)")

    formula_transformed = transformed_response + " ~ " + " + ".join(terms)

    model = smf.glm(
        formula=formula_transformed,
        data=df_fit,
        family=sm.families.Gaussian()
    ).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # CORRECT GAUSSIAN DEVIANCE (R-COMPATIBLE)
    # ======================================================

    st.header("5️⃣ Model Fit Evaluation")

    fitted_vals = model.fittedvalues
    residuals = y_trans - fitted_vals

    # For Gaussian with identity link:
    # Deviance = SSE

    residual_deviance = np.sum(residuals**2)

    y_bar = np.mean(y_trans)
    null_deviance = np.sum((y_trans - y_bar)**2)

    n = len(y_trans)
    k = model.df_model + 1

    aic = model.aic

    if (n - k - 1) > 0:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = float("nan")

    sigma_hat = np.sqrt(residual_deviance / (n - k))
    rmse = np.sqrt(np.mean(residuals**2))

    st.write(f"Null Deviance: {null_deviance:.4f}")
    st.write(f"Residual Deviance: {residual_deviance:.4f}")
    st.write(f"AIC: {aic:.4f}")
    st.write(f"AICc: {aicc:.4f}")
    st.write(f"Residual SD (σ̂): {sigma_hat:.4f}")
    st.write(f"RMSE (Transformed Scale): {rmse:.4f}")

    # ======================================================
    # LIKELIHOOD RATIO TEST (R-STYLE)
    # ======================================================

    null_model = smf.glm(
        transformed_response + " ~ 1",
        data=df_fit,
        family=sm.families.Gaussian()
    ).fit()

    lr_stat = 2 * (model.llf - null_model.llf)
    df_diff = int(model.df_model)
    p_value_lr = chi2.sf(lr_stat, df_diff)

    st.subheader("Likelihood Ratio Test")
    st.write(f"LR Statistic: {lr_stat:.4f}")
    st.write(f"Degrees of Freedom: {df_diff}")
    st.write(f"p-value: {p_value_lr:.6f}")

    # ======================================================
    # COEFFICIENT INTERPRETATION
    # ======================================================

    st.subheader("Coefficient Interpretation (Transformed Scale)")

    for name in model.params.index:

        coef = round(model.params[name], 4)
        pval = model.pvalues[name]

        if name == "Intercept":
            st.markdown(
                f"**Intercept ({coef})**: Mean of the transformed response "
                f"when predictors are at reference levels or zero."
            )

        elif name.startswith("C(") and "T." in name:
            var_name = name.split("[")[0]
            var_name = var_name.replace("C(", "").split(",")[0]
            level = name.split("T.")[1].rstrip("]")
            st.markdown(
                f"**{var_name} = {level} (β = {coef})**: "
                f"Difference in transformed mean compared to reference level. "
                f"{'Statistically significant.' if pval < 0.05 else 'Not statistically significant.'}"
            )

        else:
            st.markdown(
                f"**{name} (β = {coef})**: "
                f"One-unit increase changes transformed response by {coef}. "
                f"{'Statistically significant.' if pval < 0.05 else 'Not statistically significant.'}"
            )


if __name__ == "__main__":
    run()
