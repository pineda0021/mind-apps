import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, chi2
import numpy as np


def run():

    st.title("General Linear Regression Model Lab")

    # ======================================================
    # 1. DATA UPLOAD
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
    # 2. VARIABLE SELECTION
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
    # 3. RESPONSE NORMALITY CHECK
    # ======================================================

    st.header("2️⃣ Response Normality Check")

    if not pd.api.types.is_numeric_dtype(df[response]):
        st.error("Response must be numeric.")
        return

    fig = px.histogram(
        df,
        x=response,
        title=f"Histogram of {response}",
        marginal="box"
    )
    st.plotly_chart(fig)

    y_clean = df[response].dropna()

    if len(y_clean) >= 3:
        qq_fig = sm.qqplot(y_clean, line='s')
        st.pyplot(qq_fig.figure)

        stat, p = shapiro(y_clean)

        st.write(f"Shapiro-Wilk Statistic: {stat:.4f}")
        st.write(f"p-value: {p:.4f}")

        if p > 0.05:
            st.success("Response appears normally distributed.")
        else:
            st.warning("Response does NOT appear normally distributed.")
    else:
        st.warning("Not enough data for Shapiro-Wilk test.")

    # ======================================================
    # 4. BUILD FORMULA
    # ======================================================

    terms = []

    for var in predictors:
        if var in categorical_vars:
            ref = reference_dict[var]
            terms.append(f'C({var}, Treatment(reference="{ref}"))')
        else:
            terms.append(var)

    formula = response + " ~ " + " + ".join(terms)

    # ======================================================
    # 5. FIT MODEL
    # ======================================================

    st.header("3️⃣ Fit General Linear Model")

    model = smf.ols(formula=formula, data=df).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # 6. MODEL FIT STATISTICS
    # ======================================================

    st.header("4️⃣ Model Fit Evaluation")

    n = int(model.nobs)
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
    col1.metric("Log-Likelihood", round(loglik, 2))
    col2.metric("AIC", round(aic, 2))
    col3.metric("AICc", round(aicc, 2))
    col4.metric("BIC", round(bic, 2))
    col5.metric("σ̂", round(sigma_hat, 4))

    st.metric("RMSE", round(rmse, 4))

    # ======================================================
    # Interpretation Panel
    # ======================================================

    st.subheader("Interpretation of Model Fit Metrics")

    st.markdown("**Log-Likelihood (ℓ)**")
    st.latex(r"\ell(\hat{\beta})")

    st.markdown("**AIC**")
    st.latex(r"AIC = -2\ell + 2k")

    st.markdown("**AICc**")
    st.latex(r"AIC_c = AIC + \frac{2k(k+1)}{n-k-1}")

    st.markdown("**BIC**")
    st.latex(r"BIC = -2\ell + k\ln(n)")

    st.markdown("**Residual Standard Deviation (σ̂)**")
    st.latex(r"\hat{\sigma} = \sqrt{\frac{SSE}{n-k}}")

    st.markdown("**RMSE**")
    st.latex(r"RMSE = \sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}")

    # ======================================================
    # 7. LIKELIHOOD RATIO TEST
    # ======================================================

    st.subheader("Likelihood Ratio (Deviance) Test")

    null_model = smf.ols(response + " ~ 1", data=df).fit()

    lr_stat = 2 * (model.llf - null_model.llf)
    df_diff = int(model.df_model)
    p_value_lr = chi2.sf(lr_stat, df_diff)

    st.write(f"LR Statistic: {lr_stat:.4f}")
    st.write(f"Degrees of Freedom: {df_diff}")
    st.write(f"p-value: {p_value_lr:.6f}")

    if p_value_lr < 0.05:
        st.success("Full model significantly improves over intercept-only model.")
    else:
        st.warning("Model does not significantly improve over intercept-only model.")

    # ======================================================
    # 8. EQUATION BUILDER
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
    # 9. PREDICTION
    # ======================================================

    st.header("6️⃣ Prediction")

    input_dict = {}

    for var in predictors:

        if var in categorical_vars:
            input_dict[var] = st.selectbox(var, df[var].cat.categories)
        else:
            numeric_series = pd.to_numeric(df[var], errors="coerce")

            if numeric_series.notna().sum() > 0:
                input_dict[var] = st.number_input(
                    var,
                    value=float(numeric_series.mean())
                )
            else:
                df[var] = df[var].astype("category")
                input_dict[var] = st.selectbox(
                    var,
                    df[var].cat.categories
                )

    if st.button("Predict"):

        new_df = pd.DataFrame([input_dict])

        for var in categorical_vars:
            new_df[var] = pd.Categorical(
                new_df[var],
                categories=df[var].cat.categories
            )

        prediction = model.predict(new_df)[0]
        st.success(f"Predicted {response}: {prediction:.4f}")

    # ======================================================
    # 10. PREDICTED VS ACTUAL
    # ======================================================

    st.header("7️⃣ Predicted vs Actual")

    predicted_vals = model.predict(df)

    fig2 = px.scatter(
        x=predicted_vals,
        y=df[response],
        labels={'x': 'Predicted', 'y': 'Actual'},
        title="Predicted vs Actual Values"
    )

    st.plotly_chart(fig2)


if __name__ == "__main__":
    run()
