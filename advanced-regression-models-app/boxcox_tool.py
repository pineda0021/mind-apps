import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

from scipy.stats import shapiro, chi2, boxcox_normmax


def run():

    st.title("General Linear Regression Model")

    # ======================================================
    # 1️⃣ DATA UPLOAD
    # ======================================================

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="glm_upload"
    )

    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ======================================================
    # 2️⃣ VARIABLE SELECTION
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

    # ======================================================
    # 3️⃣ RESPONSE NORMALITY CHECK
    # ======================================================

    st.header("2️⃣ Response Normality Check")

    if not pd.api.types.is_numeric_dtype(df[response]):
        st.error("Response must be numeric.")
        return

    fig = px.histogram(df, x=response, marginal="box")
    st.plotly_chart(fig)

    qq_fig = sm.qqplot(df[response].dropna(), line='s')
    st.pyplot(qq_fig.figure)

    stat, p = shapiro(df[response].dropna())

    st.write(f"Shapiro-Wilk Statistic: {stat:.4f}")
    st.write(f"p-value: {p:.4f}")

    if p > 0.05:
        st.success("Response appears normally distributed.")
    else:
        st.warning("Response does NOT appear normally distributed.")

    # ======================================================
    # 4️⃣ TRANSFORMATION (Recommended λ)
    # ======================================================

    st.header("3️⃣ Transformation (If Needed)")

    lambda_hat = None

    if p <= 0.05:

        if (df[response] <= 0).any():
            st.error("Transformation requires strictly positive response values.")
        else:
            y_original = df[response]

            lambda_mle = boxcox_normmax(y_original, method="mle")
            st.write(f"Estimated λ (MLE): **{lambda_mle:.4f}**")

            recommended_lambdas = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
            lambda_hat = recommended_lambdas[np.argmin(abs(recommended_lambdas - lambda_mle))]

            st.info(f"Using Recommended λ = {lambda_hat}")

            if lambda_hat == -2:
                df[response] = 1 / (y_original ** 2)
                st.latex(r"y^* = \frac{1}{y^2}")

            elif lambda_hat == -1:
                df[response] = 1 / y_original
                st.latex(r"y^* = \frac{1}{y}")

            elif lambda_hat == -0.5:
                df[response] = 1 / np.sqrt(y_original)
                st.latex(r"y^* = \frac{1}{\sqrt{y}}")

            elif lambda_hat == 0:
                df[response] = np.log(y_original)
                st.latex(r"y^* = \ln(y)")

            elif lambda_hat == 0.5:
                df[response] = np.sqrt(y_original)
                st.latex(r"y^* = \sqrt{y}")

            elif lambda_hat == 1:
                st.latex(r"y^* = y")

            elif lambda_hat == 2:
                df[response] = y_original ** 2
                st.latex(r"y^* = y^2")

    # ======================================================
    # 5️⃣ BUILD FORMULA
    # ======================================================

    terms = []

    for var in predictors:
        if var in categorical_vars:
            ref = reference_dict[var]
            terms.append(f'C({var}, Treatment(reference=\"{ref}\"))')
        else:
            terms.append(var)

    formula = response + " ~ " + " + ".join(terms)

    # ======================================================
    # 6️⃣ FIT MODEL
    # ======================================================

    st.header("4️⃣ Fit General Linear Model")

    model = smf.ols(formula=formula, data=df).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # 7️⃣ MODEL FIT STATISTICS
    # ======================================================

    st.subheader("Model Fit Statistics")

    n = df.shape[0]
    k = int(model.df_model) + 1

    loglik = model.llf
    aic = model.aic
    bic = model.bic
    sigma_hat = np.sqrt(model.mse_resid)
    rmse = np.sqrt(np.mean(model.resid ** 2))

    if (n - k - 1) > 0:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = float("nan")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Log-Likelihood", round(loglik, 3))
    col2.metric("AIC", round(aic, 3))
    col3.metric("AICc", round(aicc, 3))
    col4.metric("BIC", round(bic, 3))
    col5.metric("σ̂ (Residual SD)", round(sigma_hat, 4))

    st.metric("RMSE", round(rmse, 4))

    # ======================================================
    # 📘 INTERPRETATION (Same Style, Cleaner Rendering)
    # ======================================================

    st.subheader("Interpretation of Model Fit Metrics")

    st.markdown("**Log-Likelihood (ℓ)**")
    st.latex(r"\ell(\hat{\beta})")

    st.markdown("**AIC**")
    st.latex(r"AIC = -2\ell + 2k")

    st.markdown("**AICc**")
    st.latex(r"AICc = AIC + \frac{2k(k+1)}{n-k-1}")

    st.markdown("**BIC**")
    st.latex(r"BIC = -2\ell + k\ln(n)")

    st.markdown("**Residual Standard Deviation (σ̂)**")
    st.latex(r"\hat{\sigma} = \sqrt{\frac{SSE}{n-k}}")

    st.markdown("**RMSE**")
    st.latex(r"RMSE = \sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}")

    # ======================================================
    # 5️⃣ PREDICTION
    # ======================================================

    st.header("5️⃣ Prediction")

    input_dict = {}

    for var in predictors:
        if var in categorical_vars:
            input_dict[var] = st.selectbox(
                var,
                df[var].astype("category").cat.categories
            )
        else:
            input_dict[var] = st.number_input(
                var,
                value=float(df[var].mean())
            )

    if st.button("Predict"):

        new_df = pd.DataFrame([input_dict])
        pred = model.predict(new_df)[0]

        if lambda_hat is not None:

            if lambda_hat == -2:
                pred_original = 1 / np.sqrt(pred)
            elif lambda_hat == -1:
                pred_original = 1 / pred
            elif lambda_hat == -0.5:
                pred_original = 1 / (pred ** 2)
            elif lambda_hat == 0:
                pred_original = np.exp(pred)
            elif lambda_hat == 0.5:
                pred_original = pred ** 2
            elif lambda_hat == 1:
                pred_original = pred
            elif lambda_hat == 2:
                pred_original = np.sqrt(pred)

            st.success(f"Predicted (transformed scale): {pred:.4f}")
            st.success(f"Predicted (original scale): {pred_original:.4f}")
        else:
            st.success(f"Predicted value: {pred:.4f}")

    # ======================================================
    # 6️⃣ PREDICTED VS ACTUAL
    # ======================================================

    st.header("6️⃣ Predicted vs Actual")

    predicted_vals = model.predict(df)

    fig2 = px.scatter(
        x=predicted_vals,
        y=df[response],
        labels={'x': 'Predicted', 'y': 'Actual'},
        title="Predicted vs Actual Values"
    )

    st.plotly_chart(fig2)
