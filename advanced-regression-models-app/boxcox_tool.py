import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import shapiro, boxcox_normmax, chi2


# ======================================================
# APP
# ======================================================

def run():

    st.title("📘 Ordinary Least Squares (OLS) Regression Lab")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        return

    df_original = pd.read_csv(uploaded_file)
    df = df_original.copy()

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
    # 2️⃣ Box–Cox Transformation (SAFE)
    # ======================================================

    st.header("2️⃣ Box–Cox Transformation (Optional)")

    transformed = False
    df_model = df.copy()

    # Safe numeric conversion
    y_clean = pd.to_numeric(df[response], errors="coerce").dropna()
    y_clean = y_clean[np.isfinite(y_clean)]

    can_boxcox = True

    if not np.issubdtype(y_clean.dtype, np.number):
        st.warning("Response must be numeric for Box–Cox.")
        can_boxcox = False

    if y_clean.nunique() < 2:
        st.warning("Response has no variation. Box–Cox skipped.")
        can_boxcox = False

    if (y_clean <= 0).any():
        st.warning("Box–Cox requires strictly positive values. Skipped.")
        can_boxcox = False

    if can_boxcox:

        try:
            lambda_mle = boxcox_normmax(y_clean)
        except Exception:
            st.warning("Box–Cox optimization failed for this dataset.")
            can_boxcox = False

    if can_boxcox:

        st.write(f"MLE λ = {lambda_mle:.4f}")

        recommended_lambdas = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
        rounded_lambda = recommended_lambdas[
            np.argmin(np.abs(recommended_lambdas - lambda_mle))
        ]

        st.write(f"Recommended rounded λ = {rounded_lambda}")

        use_exact = st.checkbox("Use exact MLE λ instead of rounded")

        if st.checkbox("Apply Box–Cox Transformation"):

            transformed = True
            chosen_lambda = lambda_mle if use_exact else rounded_lambda
            y = df[response]
            transformed_response = response + "_tr"

            if chosen_lambda == 0:
                df_model[transformed_response] = np.log(y)
            else:
                df_model[transformed_response] = (y**chosen_lambda - 1) / chosen_lambda

            stat_tr, p_tr = shapiro(df_model[transformed_response].dropna())
            st.write(f"Shapiro-Wilk p-value (transformed Y): {p_tr:.4f}")

            formula_transformed = transformed_response + " ~ " + " + ".join(terms)

    # ======================================================
    # 3️⃣ Model Fitting
    # ======================================================

    st.header("3️⃣ Model Fitting")

    model_original = smf.ols(formula=formula_original, data=df).fit()
    st.subheader("Original Model Summary")
    st.text(model_original.summary())

    active_response = response
    model = model_original

    if transformed:

        st.header("Model Fitting for Transform")

        model_transformed = smf.glm(
            formula=formula_transformed,
            data=df_model,
            family=sm.families.Gaussian()
        ).fit()

        st.subheader("Transformed Model Summary")
        st.text(model_transformed.summary())

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

    # ======================================================
    # 4️⃣ Coefficient Interpretation
    # ======================================================

    st.header("4️⃣ Coefficient Interpretation")

    for name in model.params.index:

        coef = round(model.params[name], 4)
        pval = model.pvalues[name]

        if name == "Intercept":
            st.markdown(
                f"**Intercept ({coef})**: Estimated mean of the response "
                f"when all predictors are zero or at reference levels. "
                f"{'Significant.' if pval < 0.05 else 'Not significant.'}"
            )
        elif name.startswith("C(") and "T." in name:
            var_name = name.split("[")[0].replace("C(", "").split(",")[0]
            level = name.split("T.")[1].rstrip("]")
            st.markdown(
                f"**{var_name} = {level} (β = {coef})**: "
                f"Difference from reference level. "
                f"{'Significant.' if pval < 0.05 else 'Not significant.'}"
            )
        else:
            st.markdown(
                f"**{name} (β = {coef})**: "
                f"For one-unit increase in {name}, response changes by {coef}. "
                f"{'Significant.' if pval < 0.05 else 'Not significant.'}"
            )

    # ======================================================
    # 5️⃣ Assumption Checks
    # ======================================================

    st.header("5️⃣ Assumption Checks")

    residuals = model.resid if hasattr(model, "resid") else model.resid_response
    fitted = model.fittedvalues

    fig_resid = px.scatter(
        x=fitted,
        y=residuals,
        labels={'x': 'Fitted', 'y': 'Residuals'},
        title="Residuals vs Fitted"
    )
    fig_resid.add_hline(y=0)
    st.plotly_chart(fig_resid)

    # ======================================================
    # 6️⃣ Predicted vs Actual
    # ======================================================

    st.header("6️⃣ Predicted vs Actual")

    predicted_vals = model.predict(df_model)

    fig2 = px.scatter(
        x=predicted_vals,
        y=df_model[active_response],
        labels={'x': 'Predicted', 'y': 'Actual'},
        title="Predicted vs Actual"
    )

    st.plotly_chart(fig2)


if __name__ == "__main__":
    run()
