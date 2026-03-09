import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from scipy.stats import shapiro, boxcox_normmax, boxcox_llf, chi2

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

    st.latex(r"""
    \tilde{y} =
    \begin{cases}
    \dfrac{y^\lambda - 1}{\lambda}, & \lambda \ne 0 \\
    \ln y, & \lambda = 0
    \end{cases}
    """)

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

            if chosen_lambda == 0:
                df_model[response] = np.log(df[response])
            else:
                df_model[response] = (df[response] ** chosen_lambda - 1) / chosen_lambda

            st.write(f"Using λ = {chosen_lambda:.4f}")

            col1, col2 = st.columns(2)

            with col1:
                fig1 = px.histogram(df[response], nbins=30,
                                    title="Original Response")
                st.plotly_chart(fig1)

            with col2:
                fig2 = px.histogram(df_model[response], nbins=30,
                                    title="Transformed Response")
                st.plotly_chart(fig2)

    else:
        st.warning("Box–Cox requires strictly positive response values.")

    # ======================================================
    # 4️⃣ Fit Model
    # ======================================================

    st.header("4️⃣ Fit OLS Model")

    model_original = smf.ols(formula=formula, data=df).fit()
    model = smf.ols(formula=formula, data=df_model).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # 3️⃣ Assumption Checks
    # ======================================================

    st.header("3️⃣ Assumption Checks")

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

    # ---- FIXED SHAPIRO BLOCK ----
    if len(residuals) >= 3:
        stat_r, p_r = shapiro(residuals)
        st.write(f"Shapiro-Wilk p-value: {p_r:.4f}")

        if p_r > 0.05:
            st.success("Fail to reject H₀: Residuals are approximately normal.")
        else:
            st.error("Reject H₀: Residuals are not normally distributed.")
    else:
        st.warning("Not enough data for Shapiro-Wilk test.")

    # ======================================================
    # Model Fit Metrics
    # ======================================================

    st.header("Model Fit Metrics")

    sigma_hat = np.sqrt(model.mse_resid)

    col1, col2, col3 = st.columns(3)
    col1.metric("R²", round(model.rsquared, 4))
    col2.metric("Adj R²", round(model.rsquared_adj, 4))
    col3.metric("σ̂ (Residual SD)", round(sigma_hat, 4))

    # ======================================================
    # Correct Likelihood Ratio Test
    # ======================================================

    if transformed:

        st.subheader("Likelihood Ratio (Deviance) Test")

        ll_bc = boxcox_llf(chosen_lambda, y_clean)
        ll_linear = boxcox_llf(1, y_clean)

        deviance = 2 * (ll_bc - ll_linear)
        df_test = 1  # λ is one parameter
        p_value = 1 - chi2.cdf(deviance, df_test)

        st.write(f"Deviance Statistic (D): {deviance:.4f}")
        st.write(f"Degrees of Freedom: {df_test}")
        st.write(f"p-value: {p_value:.4f}")

        if p_value < 0.05:
            st.success("Transformation significantly improves normality.")
        else:
            st.info("No significant improvement from transformation.")

    # ======================================================
    # Fitted Regression Equation
    # ======================================================

    st.subheader("Fitted Regression Equation")

    params = model.params
    equation = response + " = "

    for i, (name, coef) in enumerate(params.items()):
        if i == 0:
            equation += f"{coef:.4f}"
        else:
            sign = "+" if coef >= 0 else "-"
            equation += f" {sign} {abs(coef):.4f}\\,{name}"

    st.latex(equation)

    # ======================================================
    # Prediction Tool
    # ======================================================

    st.subheader("5️⃣ Prediction")

    input_data = {}

    for var in predictors:
        input_data[var] = st.number_input(f"Enter value for {var}", key=f"pred_{var}")

    if st.button("Predict"):

        new_df = pd.DataFrame([input_data])
        prediction = model.predict(new_df)[0]

        st.success(f"Predicted {response} = {prediction:.4f}")

    # ======================================================
    # Predicted vs Observed
    # ======================================================

    st.header("Predicted vs Observed")

    predicted_vals = model.predict(df_model)

    fig2 = px.scatter(
        x=predicted_vals,
        y=df_model[response],
        labels={'x': 'Predicted', 'y': 'Observed'},
        title="Predicted vs Observed"
    )

    min_val = min(predicted_vals.min(), df_model[response].min())
    max_val = max(predicted_vals.max(), df_model[response].max())

    fig2.add_shape(type="line", x0=min_val, y0=min_val,
                   x1=max_val, y1=max_val)

    st.plotly_chart(fig2)

if __name__ == "__main__":
    run()
