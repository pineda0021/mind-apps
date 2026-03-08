import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, boxcox_normmax, boxcox_llf

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
    # Model Specification
    # ======================================================

    st.header("1️⃣ Model Specification")

    response = st.selectbox("Select Response Variable (Y)", df.columns)

    predictors = st.multiselect(
        "Select Predictor Variables (X)",
        [col for col in df.columns if col != response]
    )

    if not predictors:
        return

    categorical_vars = st.multiselect(
        "Select Categorical Predictors",
        predictors
    )

    reference_dict = {}

    for col in categorical_vars:
        df[col] = df[col].astype("category")
        ref = st.selectbox(
            f"Reference level for {col}",
            df[col].cat.categories,
            key=f"ref_{col}"
        )
        reference_dict[col] = ref

    # Build formula
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
    # Box–Cox Transformation
    # ======================================================

    st.header("2️⃣ Box–Cox Transformation (Optional)")

    transformed = False
    df_model = df.copy()

    y_clean = df[response].dropna()

    if (y_clean > 0).all():

        lambda_mle = boxcox_normmax(y_clean)
        st.write(f"MLE λ = {lambda_mle:.4f}")

        lambdas = np.linspace(-2.5, 2.5, 400)
        llf_vals = [boxcox_llf(l, y_clean) for l in lambdas]

        fig_lambda = px.line(
            x=lambdas,
            y=llf_vals,
            labels={"x": "Lambda (λ)", "y": "Log-Likelihood"},
            title="Box–Cox Profile Log-Likelihood"
        )

        fig_lambda.add_vline(
            x=lambda_mle,
            line_dash="dash",
            annotation_text=f"λ̂ = {lambda_mle:.3f}"
        )

        st.plotly_chart(fig_lambda)

        recommended_lambdas = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
        rounded_lambda = recommended_lambdas[
            np.argmin(np.abs(recommended_lambdas - lambda_mle))
        ]

        st.write(f"Recommended rounded λ = {rounded_lambda}")

        use_exact = st.checkbox("Use exact MLE λ instead of rounded")

        if st.checkbox("Apply Box–Cox Transformation"):

            transformed = True
            chosen_lambda = lambda_mle if use_exact else rounded_lambda

            if chosen_lambda == 0:
                y_transformed = np.log(df[response])
            else:
                y_transformed = (df[response] ** chosen_lambda - 1) / chosen_lambda

            df_model[response] = y_transformed

            st.write(f"Using λ = {chosen_lambda:.4f}")

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.histogram(x=df[response], title="Original"))
            with col2:
                st.plotly_chart(px.histogram(x=y_transformed, title="Transformed"))

    else:
        st.warning("Box–Cox requires strictly positive response values.")

    # ======================================================
    # Fit Model
    # ======================================================

    st.header("3️⃣ Fit OLS Model")

    model_original = smf.ols(formula=formula, data=df).fit()
    model = smf.ols(formula=formula, data=df_model).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # Model Comparison
    # ======================================================

    if transformed:
        st.subheader("Model Comparison")

        comparison_df = pd.DataFrame({
            "Model": ["Original", "Transformed"],
            "Log-Likelihood": [model_original.llf, model.llf],
            "AIC": [model_original.aic, model.aic],
            "BIC": [model_original.bic, model.bic]
        })

        st.dataframe(comparison_df)

    # ======================================================
    # Residual Diagnostics
    # ======================================================

    st.header("4️⃣ Assumption Checks")

    residuals = model.resid
    fitted = model.fittedvalues

    fig_resid = px.scatter(
        x=fitted,
        y=residuals,
        labels={'x': 'Fitted Values', 'y': 'Residuals'},
        title="Residuals vs Fitted"
    )
    fig_resid.add_hline(y=0)
    st.plotly_chart(fig_resid)

    stat_r, p_r = shapiro(residuals)
    st.write(f"Residual Shapiro-Wilk p-value: {p_r:.4f}")

    # ======================================================
    # Predicted vs Observed
    # ======================================================

    st.header("5️⃣ Predicted vs Observed")

    predicted_vals = model.predict(df_model)

    fig2 = px.scatter(
        x=predicted_vals,
        y=df_model[response],
        labels={'x': 'Predicted', 'y': 'Observed'},
        title="Predicted vs Observed"
    )

    min_val = min(predicted_vals.min(), df_model[response].min())
    max_val = max(predicted_vals.max(), df_model[response].max())

    fig2.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val
    )

    st.plotly_chart(fig2)

    # ======================================================
    # Fit Metrics
    # ======================================================

    st.header("6️⃣ Model Fit Metrics")

    sigma_hat = np.sqrt(model.mse_resid)

    col1, col2, col3 = st.columns(3)
    col1.metric("R²", round(model.rsquared, 4))
    col2.metric("Adj R²", round(model.rsquared_adj, 4))
    col3.metric("σ̂ (Residual SD)", round(sigma_hat, 4))


if __name__ == "__main__":
    run()
