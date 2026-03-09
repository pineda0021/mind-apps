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

    formula = response + " ~ " + " + ".join(terms)
    st.code(formula)

    # ======================================================
    # 2️⃣ Box–Cox Transformation
    # ======================================================

    st.header("2️⃣ Box–Cox Transformation (Optional)")

    transformed = False
    df_model = df.copy()
    y_clean = df[response].dropna()

    if (y_clean > 0).all():

        lambda_mle = boxcox_normmax(y_clean)
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

            # Table-based transformations
            if chosen_lambda == -2:
                df_model[response] = 0.5 * (1 - 1 / (y**2))
            elif chosen_lambda == -1:
                df_model[response] = 1 - (1 / y)
            elif chosen_lambda == -0.5:
                df_model[response] = 2 * (1 - 1 / np.sqrt(y))
            elif chosen_lambda == 0:
                df_model[response] = np.log(y)
            elif chosen_lambda == 0.5:
                df_model[response] = 2 * (np.sqrt(y) - 1)
            elif chosen_lambda == 1:
                df_model[response] = y - 1
            elif chosen_lambda == 2:
                df_model[response] = 0.5 * (y**2 - 1)
            else:
                df_model[response] = (y**chosen_lambda - 1) / chosen_lambda

            st.write(f"Using λ = {chosen_lambda:.4f}")

            stat_tr, p_tr = shapiro(df_model[response].dropna())
            st.write(f"Shapiro-Wilk p-value (transformed Y): {p_tr:.4f}")

    else:
        st.warning("Box–Cox requires strictly positive response values.")

    # ======================================================
    # 3️⃣ Model Fitting
    # ======================================================

    st.header("3️⃣ Model Fitting")

    model_original = smf.ols(formula=formula, data=df_model).fit()

    if transformed:
        model = smf.glm(
            formula=formula,
            data=df,
            family=sm.families.Gaussian()
        ).fit()

        null_model = smf.glm(
            formula=response + " ~ 1",
            data=df_model,
            family=sm.families.Gaussian()
        ).fit()

        ll_full = model.llf
        ll_null = null_model.llf

        deviance = -2 * (ll_null - ll_full)
        df_diff = model.df_model
        p_value = 1 - chi2.cdf(deviance, df_diff)

        st.subheader("Deviance Test vs Null Model")

        col1, col2, col3 = st.columns(3)
        col1.metric("Deviance", round(deviance, 4))
        col2.metric("df", int(df_diff))
        col3.metric("p-value", round(p_value, 4))

    else:
        model = smf.ols(formula=formula, data=df_model).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # 4️⃣ AIC Comparison
    # ======================================================

    st.header("4️⃣ Model Comparison (AIC)")

    aic_original = model_original.aic

    if transformed:
        aic_transformed = model.aic

        col1, col2 = st.columns(2)
        col1.metric("AIC (Original)", round(aic_original, 4))
        col2.metric("AIC (Transformed)", round(aic_transformed, 4))
    else:
        st.metric("AIC (Original)", round(aic_original, 4))

    # ======================================================
    # 5️⃣ Assumption Checks
    # ======================================================

    st.header("5️⃣ Assumption Checks")

    if hasattr(model, "resid"):
        residuals = model.resid
    else:
        residuals = model.resid_response

    fitted = model.fittedvalues

    fig_resid = px.scatter(x=fitted, y=residuals,
                           labels={'x': 'Fitted', 'y': 'Residuals'},
                           title="Residuals vs Fitted")
    fig_resid.add_hline(y=0)
    st.plotly_chart(fig_resid)

    if len(residuals) >= 3:
        stat_r, p_r = shapiro(residuals)
        st.write(f"Shapiro-Wilk p-value (residuals): {p_r:.4f}")

    # ======================================================
    # 8️⃣ EQUATION BUILDER
    # ======================================================

    def build_equation(model, response):

        params = model.params
        equation = f"\\widehat{{\\mathbb{{E}}}}({response}) = {round(params['Intercept'],4)}"

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

    st.subheader("Fitted Regression Equation (Full Model)")
    st.latex(build_equation(model, response))

    # ======================================================
    # 6️⃣ Prediction
    # ======================================================

    st.header("6️⃣ Prediction")

    input_dict = {}

    for var in predictors:
        if var in categorical_vars:
            input_dict[var] = st.selectbox(var, df[var].cat.categories)
        else:
            input_dict[var] = st.number_input(var, value=float(df[var].mean()))

    if st.button("Predict"):

        new_df = pd.DataFrame([input_dict])

        for var in categorical_vars:
            new_df[var] = pd.Categorical(
                new_df[var],
                categories=df[var].cat.categories
            )

        prediction_tr = model.predict(new_df)[0]

        if transformed:
            if chosen_lambda == -1:
                prediction = 1 / (1 - prediction_tr)
            elif chosen_lambda == 0:
                prediction = np.exp(prediction_tr)
            elif chosen_lambda == 0.5:
                prediction = ((prediction_tr / 2) + 1)**2
            elif chosen_lambda == 1:
                prediction = prediction_tr + 1
            elif chosen_lambda == 2:
                prediction = np.sqrt(2 * prediction_tr + 1)
            else:
                prediction = (chosen_lambda * prediction_tr + 1)**(1/chosen_lambda)

            st.success(f"Predicted {response} (original scale): {prediction:.4f}")
        else:
            st.success(f"Predicted {response}: {prediction_tr:.4f}")

    # ======================================================
    # 7️⃣ Predicted vs Actual (Original Scale)
    # ======================================================

    st.header("7️⃣ Predicted vs Actual")

    if transformed:
        fitted_tr = model.predict(df_model)

        if chosen_lambda == -1:
            fitted_vals = 1 / (1 - fitted_tr)
        elif chosen_lambda == 0:
            fitted_vals = np.exp(fitted_tr)
        elif chosen_lambda == 0.5:
            fitted_vals = ((fitted_tr / 2) + 1)**2
        elif chosen_lambda == 1:
            fitted_vals = fitted_tr + 1
        elif chosen_lambda == 2:
            fitted_vals = np.sqrt(2 * fitted_tr + 1)
        else:
            fitted_vals = (chosen_lambda * fitted_tr + 1)**(1/chosen_lambda)
    else:
        fitted_vals = model.predict(df_model)

    fig2 = px.scatter(
        x=fitted_vals,
        y=df[response],
        labels={'x': 'Predicted', 'y': 'Actual'},
        title="Predicted vs Actual"
    )

    st.plotly_chart(fig2)


if __name__ == "__main__":
    run()
