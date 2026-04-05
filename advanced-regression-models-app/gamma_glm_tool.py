import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import shapiro, chi2


# ======================================================
# Helper: Rescale Prediction
# ======================================================

def rescale_prediction(value, apply_scale=False, scale_factor=1.0):
    if apply_scale:
        return value * scale_factor
    return value


def run():

    st.title("📘 Gamma Generalized Linear Model (Log Link)")

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
    # 2️⃣ MODEL SPECIFICATION
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
        "Select Categorical Variables (Factors)",
        predictors
    )

    reference_dict = {}

    df_model = df.copy()
    df_model[response] = pd.to_numeric(df_model[response], errors="coerce")

    for var in predictors:
        if var not in categorical_vars:
            df_model[var] = pd.to_numeric(df_model[var], errors="coerce")

    for col in categorical_vars:
        df_model[col] = df_model[col].astype("category")

    model_vars = [response] + predictors
    df_model = df_model.dropna(subset=model_vars)

    if df_model.empty:
        st.error("No complete cases remain after removing missing values.")
        return

    if (df_model[response] <= 0).any():
        st.error("Gamma GLM requires strictly positive response values.")
        return

    for col in categorical_vars:
        ref = st.selectbox(
            f"Select reference level for {col}",
            df_model[col].cat.categories,
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
    # 3️⃣ RESPONSE DIAGNOSTICS
    # ======================================================

    st.header("2️⃣ Response Diagnostics")

    log_response = response + "_log"
    df_model[log_response] = np.log(df_model[response])

    col1, col2 = st.columns(2)

    with col1:
        fig_orig = px.histogram(df_model, x=response, title="Original Response", marginal="box")
        st.plotly_chart(fig_orig)

    with col2:
        fig_log = px.histogram(df_model, x=log_response, title="log(Response)", marginal="box")
        st.plotly_chart(fig_log)

    st.subheader("Distribution Diagnostics")
    st.info("Shapiro test below is exploratory only for Gamma GLM.")

    if len(df_model[log_response]) >= 3:
        stat, p_value = shapiro(df_model[log_response])
        st.write(f"Shapiro Statistic: {stat:.4f}")
        st.write(f"p-value: {p_value:.4f}")

    # ======================================================
    # 4️⃣ MODEL FITTING
    # ======================================================

    st.header("3️⃣ Model Fitting")

    model = smf.glm(
        formula=formula,
        data=df_model,
        family=sm.families.Gamma(link=sm.families.links.Log())
    ).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # 5️⃣ MODEL FIT
    # ======================================================

    st.header("4️⃣ Model Fit Evaluation")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("LogLik", round(model.llf, 2))
    col2.metric("AIC", round(model.aic, 2))
    col3.metric("BIC", round(model.bic, 2))
    col4.metric("Deviance", round(model.deviance, 2))
    col5.metric("Pearson", round(model.pearson_chi2, 2))

    # ======================================================
    # 6️⃣ INTERPRETATION
    # ======================================================

    st.header("5️⃣ Interpretation of Coefficients")

    for term in model.params.index:

        coef = model.params[term]
        pval = model.pvalues[term]
        exp_beta = np.exp(coef)

        st.markdown(f"### {term}")

        if term == "Intercept":
            st.markdown(rf"$e^{{{coef:.4f}}} = {exp_beta:.4f}$")

        elif term.startswith("C("):
            st.markdown(rf"$e^{{{coef:.4f}}}\cdot100\% = {exp_beta*100:.2f}\%$")

        else:
            pct = (exp_beta - 1) * 100
            st.markdown(rf"$(e^{{{coef:.4f}}}-1)\cdot100\% = {pct:.2f}\%$")

        st.write(f"Coefficient = {coef:.4f}")
        st.write(f"p-value = {pval:.4f}")

        if pval <= 0.05:
            st.success("Significant")
        else:
            st.warning("Not significant")

    # ======================================================
    # 7️⃣ PREDICTION
    # ======================================================

    st.header("6️⃣ Prediction")

    apply_scale = st.checkbox("Rescale prediction")
    scale_factor = 1.0

    if apply_scale:
        scale_factor = st.number_input("Scale factor", value=10000.0)

    input_dict = {}

    for var in predictors:
        if var in categorical_vars:
            input_dict[var] = st.selectbox(var, df_model[var].cat.categories)
        else:
            input_dict[var] = st.number_input(var, value=float(df_model[var].mean()))

    if st.button("Predict"):

        new_df = pd.DataFrame([input_dict])

        for var in categorical_vars:
            new_df[var] = pd.Categorical(
                new_df[var],
                categories=df_model[var].cat.categories
            )

        prediction = model.predict(new_df)[0]

        final_prediction = rescale_prediction(
            prediction,
            apply_scale=apply_scale,
            scale_factor=scale_factor
        )

        st.success(f"Prediction: {final_prediction:.4f}")

    # ======================================================
    # 8️⃣ PLOT
    # ======================================================

    st.header("7️⃣ Predicted vs Actual")

    pred = model.predict(df_model)

    fig = px.scatter(x=pred, y=df_model[response])
    st.plotly_chart(fig)


if __name__ == "__main__":
    run()
