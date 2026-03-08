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

    fig = px.histogram(
        df,
        x=response,
        title=f"Histogram of {response}",
        marginal="box"
    )
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
    # 4️⃣ BOX-COX TRANSFORMATION (AUTO IF NEEDED)
    # ======================================================

    st.header("3️⃣ Box-Cox Transformation (If Needed)")

    transformed_response = None
    lambda_hat = None
    original_model = None

    if p <= 0.05:

        st.warning(
            "The response is not normally distributed. "
            "A Box-Cox transformation will be applied."
        )

        if (df[response] <= 0).any():
            st.error("Box-Cox requires strictly positive response values.")
        else:
            y_original = df[response].dropna()

            lambda_hat = boxcox_normmax(y_original, method="mle")
            lambdas = np.linspace(-2, 2, 200)
            llf_vals = [boxcox(y_original, lmbda=l)[1] for l in lambdas]

            st.subheader("Lambda Optimization (Profile Log-Likelihood)")

            fig_lambda = px.line(
                x=lambdas,
                y=llf_vals,
                labels={"x": "Lambda (λ)", "y": "Log-Likelihood"},
                title="Box-Cox Lambda Optimization Curve"
            )

            fig_lambda.add_vline(
                x=lambda_hat,
                line_dash="dash",
                annotation_text=f"λ̂ = {lambda_hat:.3f}"
            )

            st.plotly_chart(fig_lambda)

            st.write(f"Estimated λ (MLE): **{lambda_hat:.4f}**")

            # --- NEW: Show transformation formula ---
            if abs(lambda_hat) > 1e-6:
                st.latex(
                    rf"y^* = \frac{{y^{{{lambda_hat:.3f}}} - 1}}{{{lambda_hat:.3f}}}"
                )
            else:
                st.latex(r"y^* = \ln(y)")
            # ---------------------------------------

            # Apply transformation
            y_transformed = boxcox(y_original, lmbda=lambda_hat)
            transformed_response = f"{response}_boxcox"
            df[transformed_response] = y_transformed

            # Re-check normality
            st.subheader("Normality Check After Box-Cox")

            fig_bc = px.histogram(
                df,
                x=transformed_response,
                title=f"Histogram of Box-Cox Transformed {response}",
                marginal="box"
            )
            st.plotly_chart(fig_bc)

            qq_fig_bc = sm.qqplot(y_transformed, line='s')
            st.pyplot(qq_fig_bc.figure)

            stat_bc, p_bc = shapiro(y_transformed)

            st.write(f"Shapiro-Wilk Statistic: {stat_bc:.4f}")
            st.write(f"p-value: {p_bc:.4f}")

            if p_bc > 0.05:
                st.success("The transformed response now appears normally distributed.")
            else:
                st.warning("The transformed response still deviates from normality.")

            # Save original model for comparison later
            original_formula_terms = []
            for var in predictors:
                if var in categorical_vars:
                    ref = reference_dict[var]
                    original_formula_terms.append(f'C({var}, Treatment(reference=\"{ref}\"))')
                else:
                    original_formula_terms.append(var)

            original_formula = df.columns[df.columns.get_loc(response)] + " ~ " + " + ".join(original_formula_terms)
            original_model = smf.ols(formula=original_formula, data=df).fit()

            # Use transformed response moving forward
            response = transformed_response

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

    st.header("5️⃣ Model Fit Evaluation")

    n = df.shape[0]
    k = int(model.df_model) + 1

    loglik = model.llf
    aic = model.aic
    bic = model.bic
    sigma_hat = model.mse_resid ** 0.5
    rmse = (model.resid ** 2).mean() ** 0.5

    if (n - k - 1) > 0:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = float("nan")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Log-Likelihood", round(loglik, 2))
    col2.metric("AIC", round(aic, 2))
    col3.metric("AICc", round(aicc, 2))
    col4.metric("BIC", round(bic, 2))
    col5.metric("σ̂ (Residual SD)", round(sigma_hat, 4))
    col6.metric("RMSE", round(rmse, 4))

    # --- NEW: AIC Comparison ---
    if original_model is not None:
        st.subheader("Model Comparison: Original vs Box-Cox")

        comparison = pd.DataFrame({
            "Model": ["Original", "Box-Cox"],
            "AIC": [original_model.aic, model.aic],
            "BIC": [original_model.bic, model.bic],
            "Log-Likelihood": [original_model.llf, model.llf]
        })

        st.dataframe(comparison)

        if model.aic < original_model.aic:
            st.success("Box-Cox model shows improved fit (lower AIC).")
        else:
            st.info("Original model may be preferable based on AIC.")
    # ---------------------------------------

    # ======================================================
    # 8️⃣ PREDICTION
    # ======================================================

    st.header("6️⃣ Prediction")

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
        pred_transformed = model.predict(new_df)[0]

        # --- NEW: Back-transformation ---
        if lambda_hat is not None:
            if abs(lambda_hat) > 1e-6:
                pred_original = (lambda_hat * pred_transformed + 1) ** (1 / lambda_hat)
            else:
                pred_original = np.exp(pred_transformed)

            st.success(f"Predicted {response} (transformed): {pred_transformed:.4f}")
            st.success(f"Predicted original scale: {pred_original:.4f}")
        else:
            st.success(f"Predicted {response}: {pred_transformed:.4f}")
        # --------------------------------

    # ======================================================
    # 9️⃣ PREDICTED VS ACTUAL
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
