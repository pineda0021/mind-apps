import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, boxcox, boxcox_normmax, boxcox_llf

# ======================================================
# APP
# ======================================================

def run():

    st.title("📘 Ordinary Least Squares (OLS) Regression Lab")
    st.markdown("""
    This lab walks through the full statistical workflow:

    1. Specify model  
    2. Check assumptions  
    3. Consider Box–Cox transformation  
    4. Fit model  
    5. Diagnose residuals  
    6. Interpret results  

    ⚠ This is an **OLS linear model with normal errors**, not a generalized linear model.
    """)

    # ======================================================
    # 1. DATA UPLOAD
    # ======================================================

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        return

    df_original = pd.read_csv(uploaded_file)
    df = df_original.copy()

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ======================================================
    # 2. VARIABLE SELECTION
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
    # 3️⃣ Box–Cox Transformation (Using Standard Formula)
    # ======================================================

    st.header("3️⃣ Box–Cox Transformation (Optional)")

    transformed = False
    df_model = df.copy()

    y_clean = df[response].dropna()

    if (y_clean > 0).all():

    lambda_mle = boxcox_normmax(y_clean)
    st.write(f"MLE λ = {lambda_mle:.4f}")

    # Profile log-likelihood
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

    # Recommended rounding table
    recommended_lambdas = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    rounded_lambda = recommended_lambdas[
        np.argmin(np.abs(recommended_lambdas - lambda_mle))
    ]

    st.write(f"Recommended rounded λ = {rounded_lambda}")

    use_exact = st.checkbox("Use exact MLE λ instead of rounded")

    if st.checkbox("Apply Box–Cox Transformation"):

        transformed = True

        if use_exact:
            chosen_lambda = lambda_mle
        else:
            chosen_lambda = rounded_lambda

        st.latex(r"""
        \tilde{y} =
        \begin{cases}
        \dfrac{y^{\lambda} - 1}{\lambda}, & \lambda \ne 0 \\
        \ln y, & \lambda = 0
        \end{cases}
        """)

        if chosen_lambda == 0:
            y_transformed = np.log(df[response])
        else:
            y_transformed = (df[response] ** chosen_lambda - 1) / chosen_lambda

        df_model[response] = y_transformed

        st.write(f"Using λ = {chosen_lambda:.4f}")

        # Show distributions
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(px.histogram(x=df[response], title="Original"))

        with col2:
            st.plotly_chart(px.histogram(x=y_transformed, title="Transformed"))

else:
    st.warning("Box–Cox requires strictly positive response values.")

    # ======================================================
    # 4. BOX-COX
    # ======================================================

    st.header("3️⃣ Box–Cox Transformation (Optional)")

    transformed = False
    df_model = df.copy()

    if (y_clean > 0).all():

        lambda_mle = boxcox_normmax(y_clean)
        st.write(f"MLE λ = {lambda_mle:.4f}")

        lambdas = np.linspace(-2, 2, 200)
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

        use_exact = st.checkbox("Use exact MLE λ")

        if st.checkbox("Apply Box–Cox Transformation"):

            transformed = True
            y_original = df[response].copy()

            if use_exact:
                y_transformed = boxcox(y_original, lmbda=lambda_mle)
                st.write(f"Using exact λ = {lambda_mle:.4f}")
            else:
                rounded = round(lambda_mle, 1)
                y_transformed = boxcox(y_original, lmbda=rounded)
                st.write(f"Using rounded λ = {rounded}")

            df_model[response] = y_transformed

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.histogram(x=y_original, title="Original"))
            with col2:
                st.plotly_chart(px.histogram(x=y_transformed, title="Transformed"))

    else:
        st.info("Box–Cox requires strictly positive response values.")

    # ======================================================
    # 5. FIT MODEL
    # ======================================================

    st.header("4️⃣ Fit OLS Model")

    model_original = smf.ols(formula=formula, data=df).fit()
    model = smf.ols(formula=formula, data=df_model).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # Confidence intervals
    st.subheader("95% Confidence Intervals")
    st.dataframe(model.conf_int())

    # ======================================================
    # 6. MODEL COMPARISON
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

        st.caption("Note: AIC comparison across transformed responses should be interpreted cautiously.")

    # ======================================================
    # 7. RESIDUAL DIAGNOSTICS
    # ======================================================

    st.header("5️⃣ Assumption Checks")

    residuals = model.resid
    fitted = model.fittedvalues

    # Residual vs Fitted
    fig_resid = px.scatter(
        x=fitted,
        y=residuals,
        labels={'x': 'Fitted Values', 'y': 'Residuals'},
        title="Residuals vs Fitted"
    )
    fig_resid.add_hline(y=0)
    st.plotly_chart(fig_resid)

    # Residual QQ
    qq_resid = sm.qqplot(residuals, line='s')
    st.pyplot(qq_resid.figure)

    stat_r, p_r = shapiro(residuals)
    st.write(f"Residual Shapiro-Wilk p-value: {p_r:.4f}")

    # Cook's Distance
    influence = model.get_influence()
    cooks = influence.cooks_distance[0]

    fig_cook = px.scatter(
        x=np.arange(len(cooks)),
        y=cooks,
        labels={'x': 'Observation Index', 'y': "Cook's Distance"},
        title="Cook's Distance"
    )
    st.plotly_chart(fig_cook)

    # ======================================================
    # 8. PREDICTED VS ACTUAL
    # ======================================================

    st.header("6️⃣ Predicted vs Actual")

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
    # 9. FIT STATISTICS
    # ======================================================

    st.header("7️⃣ Model Fit Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", round(model.rsquared, 4))
    col2.metric("Adj R²", round(model.rsquared_adj, 4))
    col3.metric("AIC", round(model.aic, 2))
    col4.metric("RMSE", round(np.sqrt(np.mean(residuals**2)), 4))

    st.markdown("""
    ### Interpretation Guide

    - **R²**: Proportion of variance explained.
    - **Residual vs Fitted**: Look for random scatter.
    - **QQ Plot**: Points should follow straight line.
    - **Cook's Distance**: Values > 4/n may indicate influential points.
    """)


if __name__ == "__main__":
    run()
