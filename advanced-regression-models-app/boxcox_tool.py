import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

from scipy.stats import shapiro, chi2, boxcox, boxcox_normmax


def run():

    st.title("General Linear Regression Model Lab (College-Level)")

    # ======================================================
    # 1️⃣ DATA UPLOAD
    # ======================================================

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ======================================================
    # 2️⃣ VARIABLE SELECTION
    # ======================================================

    st.header("1️⃣ Select Variables")

    response_original = st.selectbox("Select Response Variable (Y)", df.columns)

    predictors = st.multiselect(
        "Select Predictor Variables (X)",
        [col for col in df.columns if col != response_original]
    )

    if not predictors:
        return

    categorical_vars = st.multiselect("Select Categorical Variables (Factors)", predictors)

    reference_dict = {}
    for col in categorical_vars:
        df[col] = df[col].astype("category")
        ref = st.selectbox(f"Reference level for {col}", df[col].cat.categories)
        reference_dict[col] = ref

    # ======================================================
    # 3️⃣ RESPONSE NORMALITY CHECK
    # ======================================================

    st.header("2️⃣ Response Normality Check")

    if not pd.api.types.is_numeric_dtype(df[response_original]):
        st.error("Response must be numeric.")
        return

    stat, p = shapiro(df[response_original].dropna())

    st.write(f"Shapiro-Wilk p-value: {p:.4f}")

    lambda_hat = None
    transformed_response = None

    # ======================================================
    # 3️⃣ BOX-COX (IF NEEDED)
    # ======================================================

    if p <= 0.05 and (df[response_original] > 0).all():

        st.header("3️⃣ Box-Cox Transformation")

        y_original = df[response_original].dropna()
        lambda_hat = boxcox_normmax(y_original, method="mle")

        st.write(f"Estimated λ (MLE): {lambda_hat:.4f}")

        # Apply transformation
        y_transformed = boxcox(y_original, lmbda=lambda_hat)
        transformed_response = f"{response_original}_boxcox"
        df[transformed_response] = y_transformed

        # Show formula
        if abs(lambda_hat) > 1e-6:
            st.latex(
                rf"y^* = \frac{{y^{{{lambda_hat:.3f}}} - 1}}{{{lambda_hat:.3f}}}"
            )
        else:
            st.latex(r"y^* = \ln(y)")

        response = transformed_response
    else:
        response = response_original

    # ======================================================
    # 4️⃣ FIT GENERAL LINEAR MODEL
    # ======================================================

    st.header("4️⃣ Fit General Linear Model")

    terms = []
    for var in predictors:
        if var in categorical_vars:
            ref = reference_dict[var]
            terms.append(f'C({var}, Treatment(reference="{ref}"))')
        else:
            terms.append(var)

    formula = response + " ~ " + " + ".join(terms)

    model = smf.ols(formula=formula, data=df).fit()

    # ------------------------------
    # Fit Statistics
    # ------------------------------

    n = df.shape[0]
    k = int(model.df_model) + 1

    loglik = model.llf
    aic = model.aic
    bic = model.bic

    if (n - k - 1) > 0:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = np.nan

    st.subheader("Model Fit Statistics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Log-Likelihood", round(loglik, 3))
    col2.metric("AIC", round(aic, 3))
    col3.metric("AICc", round(aicc, 3))
    col4.metric("BIC", round(bic, 3))

    # ------------------------------
    # Likelihood Ratio (Deviance Test)
    # ------------------------------

    null_model = smf.ols(response + " ~ 1", data=df).fit()

    LR_stat = 2 * (model.llf - null_model.llf)
    df_diff = model.df_model
    p_value_lr = chi2.sf(LR_stat, df_diff)

    st.subheader("Likelihood Ratio Test (Deviance Test)")

    st.write(f"LR Statistic: {LR_stat:.4f}")
    st.write(f"Degrees of Freedom: {int(df_diff)}")
    st.write(f"p-value: {p_value_lr:.6f}")

    # ------------------------------
    # Fitted Regression Equation
    # ------------------------------

    st.subheader("Fitted Regression Equation (Full Model)")

    coefs = model.params
    equation = f"{response} = "

    terms_eq = []
    for name, coef in coefs.items():
        if name == "Intercept":
            terms_eq.append(f"{coef:.4f}")
        else:
            terms_eq.append(f"{coef:.4f}({name})")

    equation += " + ".join(terms_eq)

    st.code(equation)

    # ------------------------------
    # Transformed Model Equation
    # ------------------------------

    if lambda_hat is not None:
        st.subheader("Transformed Model Equation")
        st.code(equation)
        st.text("Note: Model is fitted on Box-Cox transformed response.")

    # ------------------------------
    # Model Summary
    # ------------------------------

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # 5️⃣ PREDICTION
    # ======================================================

    st.header("5️⃣ Prediction")

    input_dict = {}

    for var in predictors:
        if var in categorical_vars:
            input_dict[var] = st.selectbox(
                var, df[var].astype("category").cat.categories
            )
        else:
            input_dict[var] = st.number_input(var, value=float(df[var].mean()))

    if st.button("Predict"):

        new_df = pd.DataFrame([input_dict])
        pred = model.predict(new_df)[0]

        if lambda_hat is not None:
            if abs(lambda_hat) > 1e-6:
                pred_original = (lambda_hat * pred + 1) ** (1 / lambda_hat)
            else:
                pred_original = np.exp(pred)

            st.success(f"Predicted (transformed scale): {pred:.4f}")
            st.success(f"Predicted (original scale): {pred_original:.4f}")
        else:
            st.success(f"Predicted value: {pred:.4f}")

    # ======================================================
    # 6️⃣ PREDICTED VS ACTUAL
    # ======================================================

    st.header("6️⃣ Predicted vs Actual")

    predicted_vals = model.predict(df)

    fig = px.scatter(
        x=predicted_vals,
        y=df[response],
        labels={"x": "Predicted", "y": "Actual"},
        title="Predicted vs Actual Values"
    )

    st.plotly_chart(fig)
