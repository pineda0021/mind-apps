import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
from scipy.stats import shapiro, boxcox_normmax, boxcox_llf, chi2

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
        fig_lambda.add_vline(x=lambda_mle, line_dash="dash")
        st.plotly_chart(fig_lambda)

        recommended = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
        rounded_lambda = recommended[np.argmin(abs(recommended - lambda_mle))]

        use_exact = st.checkbox("Use exact MLE λ")

        if st.checkbox("Apply Box–Cox Transformation"):

            transformed = True
            chosen_lambda = lambda_mle if use_exact else rounded_lambda

            if chosen_lambda == 0:
                df_model[response] = np.log(df[response])
            else:
                df_model[response] = (df[response]**chosen_lambda - 1)/chosen_lambda

            st.write(f"Using λ = {chosen_lambda:.4f}")

    else:
        st.warning("Box–Cox requires strictly positive response values.")

    # ======================================================
    # 3️⃣ Fit Models
    # ======================================================

    st.header("3️⃣ Fit OLS Model")

    model_original = smf.ols(formula=formula, data=df).fit()
    model = smf.ols(formula=formula, data=df_model).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # Likelihood Ratio Test
    # ======================================================

    if transformed:
        st.subheader("Likelihood Ratio (Deviance) Test")

        deviance = 2 * (model.llf - model_original.llf)
        p_value = 1 - chi2.cdf(deviance, df=1)

        st.write(f"Deviance Statistic: {deviance:.4f}")
        st.write(f"p-value: {p_value:.4f}")

        if p_value < 0.05:
            st.success("Transformation significantly improves model fit.")
        else:
            st.info("No significant improvement from transformation.")

    # ======================================================
    # Fitted Regression Equation
    # ======================================================

    st.header("4️⃣ Fitted Regression Equation")

    params = model.params
    equation = f"{response} = "

    for i, (name, coef) in enumerate(params.items()):
        if i == 0:
            equation += f"{coef:.4f}"
        else:
            sign = "+" if coef >= 0 else "-"
            equation += f" {sign} {abs(coef):.4f}·{name}"

    st.latex(equation)

    # ======================================================
    # Coefficient Interpretation
    # ======================================================

    st.header("5️⃣ Coefficient Interpretation")

    coef_table = model.summary2().tables[1]
    interpretation = []

    for idx, row in coef_table.iterrows():
        if idx == "Intercept":
            continue
        direction = "increase" if row["Coef."] > 0 else "decrease"
        sig = "significant" if row["P>|t|"] < 0.05 else "not significant"
        interpretation.append(
            f"{idx}: A one-unit increase leads to a {direction} in Y ({sig})."
        )

    for line in interpretation:
        st.write(line)

    # ======================================================
    # Prediction Tool
    # ======================================================

    st.header("6️⃣ Prediction")

    input_data = {}
    for var in predictors:
        input_data[var] = st.number_input(f"Enter value for {var}")

    if st.button("Predict"):
        new_df = pd.DataFrame([input_data])
        prediction = model.predict(new_df)[0]
        st.success(f"Predicted {response} = {prediction:.4f}")

    # ======================================================
    # Predicted vs Actual
    # ======================================================

    st.header("7️⃣ Predicted vs Actual")

    predicted_vals = model.predict(df_model)

    fig2 = px.scatter(
        x=predicted_vals,
        y=df_model[response],
        labels={'x': 'Predicted', 'y': 'Observed'}
    )

    min_val = min(predicted_vals.min(), df_model[response].min())
    max_val = max(predicted_vals.max(), df_model[response].max())

    fig2.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val)
    st.plotly_chart(fig2)

    # ======================================================
    # Model Fit Metrics
    # ======================================================

    st.header("8️⃣ Model Fit Metrics")

    sigma_hat = np.sqrt(model.mse_resid)

    col1, col2, col3 = st.columns(3)
    col1.metric("R²", round(model.rsquared, 4))
    col2.metric("Adj R²", round(model.rsquared_adj, 4))
    col3.metric("σ̂", round(sigma_hat, 4))


if __name__ == "__main__":
    run()
