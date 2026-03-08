import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, chi2, boxcox, boxcox_normmax
import numpy as np


def run():

    st.title("General Linear Regression Model")

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

    fig = px.histogram(df, x=response, marginal="box")
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
    # 3B. BOX-COX FOLLOW-UP
    # ======================================================

    st.subheader("Box-Cox Transformation Follow-Up")

    transformed = False
    lambda_hat = None

    terms = []
    for var in predictors:
        if var in categorical_vars:
            ref = reference_dict[var]
            terms.append(f'C({var}, Treatment(reference=\"{ref}\"))')
        else:
            terms.append(var)

    formula = response + " ~ " + " + ".join(terms)
    original_model = smf.ols(formula=formula, data=df).fit()

    if len(y_clean) >= 3 and (y_clean > 0).all():

        lambda_mle = boxcox_normmax(y_clean, method="mle")
        st.write(f"Estimated λ (MLE): {lambda_mle:.4f}")

        lambdas = np.linspace(-2, 2, 200)
        llf_vals = [boxcox(y_clean, lmbda=l)[1] for l in lambdas]

        fig_lambda = px.line(
            x=lambdas,
            y=llf_vals,
            labels={"x": "Lambda (λ)", "y": "Log-Likelihood"},
            title="Box-Cox Profile Log-Likelihood"
        )

        fig_lambda.add_vline(
            x=lambda_mle,
            line_dash="dash",
            annotation_text=f"λ̂ = {lambda_mle:.3f}"
        )

        st.plotly_chart(fig_lambda)

        suggested_lambdas = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
        suggested = suggested_lambdas[np.argmin(abs(suggested_lambdas - lambda_mle))]
        st.write(f"Suggested transformation λ ≈ {suggested}")

        if st.checkbox("Apply Suggested Transformation"):

            transformed = True
            lambda_hat = suggested
            y_original = df[response].copy()

            if suggested == -2:
                y_transformed = 1 / (y_original ** 2)
                st.latex(r"y^* = \frac{1}{y^2}")
            elif suggested == -1:
                y_transformed = 1 / y_original
                st.latex(r"y^* = \frac{1}{y}")
            elif suggested == -0.5:
                y_transformed = 1 / np.sqrt(y_original)
                st.latex(r"y^* = \frac{1}{\sqrt{y}}")
            elif suggested == 0:
                y_transformed = np.log(y_original)
                st.latex(r"y^* = \ln(y)")
            elif suggested == 0.5:
                y_transformed = np.sqrt(y_original)
                st.latex(r"y^* = \sqrt{y}")
            elif suggested == 2:
                y_transformed = y_original ** 2
                st.latex(r"y^* = y^2")

            df[response] = y_transformed

            # Side-by-side histograms
            st.subheader("Original vs Transformed Distribution")
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(px.histogram(x=y_original, title="Original", marginal="box"))

            with col2:
                st.plotly_chart(px.histogram(x=y_transformed, title="Transformed", marginal="box"))

            # Shapiro again
            st.subheader("Normality Test After Transformation")
            stat_t, p_t = shapiro(y_transformed.dropna())
            st.write(f"Shapiro-Wilk Statistic: {stat_t:.4f}")
            st.write(f"p-value: {p_t:.4f}")

            if p_t > 0.05:
                st.success("Transformed response appears normally distributed.")
            else:
                st.warning("Transformed response still deviates from normality.")

            st.subheader("Interpretation Guidance")
            if suggested == 0:
                st.markdown("Log transformation: coefficients approximate % change.")
            elif suggested == -1:
                st.markdown("Reciprocal transformation: inverse relationship.")
            elif suggested == 0.5:
                st.markdown("Square-root transformation: nonlinear interpretation.")
            else:
                st.markdown("Power transformation: interpretation on transformed scale.")

    # ======================================================
    # 4. FIT MODEL
    # ======================================================

    st.header("3️⃣ Fit General Linear Model")

    model = smf.ols(formula=formula, data=df).fit()
    st.subheader("Model Summary")
    st.text(model.summary())

    # Model comparison
    if transformed:
        st.header("Model Comparison: Original vs Transformed")

        comparison_df = pd.DataFrame({
            "Model": ["Original", "Transformed"],
            "Log-Likelihood": [original_model.llf, model.llf],
            "AIC": [original_model.aic, model.aic],
            "BIC": [original_model.bic, model.bic]
        })

        st.dataframe(comparison_df)

        if model.aic < original_model.aic:
            st.success("Transformed model improves fit based on AIC.")
        else:
            st.info("Original model may be preferable based on AIC.")

    # ======================================================
    # 5. MODEL FIT STATISTICS
    # ======================================================

    st.header("4️⃣ Model Fit Evaluation")

    n = int(model.nobs)
    k = int(model.df_model) + 1

    loglik = model.llf
    aic = model.aic
    bic = model.bic
    sigma_hat = np.sqrt(model.mse_resid)
    rmse = np.sqrt(np.mean(model.resid ** 2))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Log-Likelihood", round(loglik, 2))
    col2.metric("AIC", round(aic, 2))
    col3.metric("BIC", round(bic, 2))
    col4.metric("σ̂", round(sigma_hat, 4))
    col5.metric("RMSE", round(rmse, 4))

    # ======================================================
    # 6. PREDICTED VS ACTUAL
    # ======================================================

    st.header("5️⃣ Predicted vs Actual")

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
