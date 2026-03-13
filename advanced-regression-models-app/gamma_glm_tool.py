```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2


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

    categorical_vars = st.multiselect("Select Categorical Predictors", predictors)

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
    # 3️⃣ RESPONSE CHECK
    # ======================================================

    st.header("2️⃣ Response Check (Gamma Requirement)")

    df[response] = pd.to_numeric(df[response], errors="coerce")

    if (df[response] <= 0).any():
        st.error("Gamma GLM requires strictly positive response values.")
        return

    fig = px.histogram(
        df,
        x=response,
        title=f"Histogram of {response}",
        marginal="box"
    )

    st.plotly_chart(fig)

    # ======================================================
    # 4️⃣ MODEL FITTING
    # ======================================================

    st.header("3️⃣ Model Fitting")

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Gamma(link=sm.families.links.Log())
    ).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # 5️⃣ LIKELIHOOD RATIO TEST
    # ======================================================

    null_model = smf.glm(
        formula=response + " ~ 1",
        data=df,
        family=sm.families.Gamma(link=sm.families.links.Log())
    ).fit()

    lr_stat = 2 * (model.llf - null_model.llf)
    df_diff = int(model.df_model)
    p_value = chi2.sf(lr_stat, df_diff)

    st.subheader("Likelihood Ratio (Deviance) Test")

    st.write(f"LR Statistic: {lr_stat:.4f}")
    st.write(f"Degrees of Freedom: {df_diff}")
    st.write(f"p-value: {p_value:.6f}")

    # ======================================================
    # 6️⃣ MODEL FIT EVALUATION
    # ======================================================

    st.header("4️⃣ Model Fit Evaluation")

    n = int(model.nobs)
    k = int(model.df_model) + 1

    loglik = model.llf
    aic = model.aic
    bic = model.bic
    deviance = model.deviance
    pearson = model.pearson_chi2

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Log-Likelihood", round(loglik, 2))
    col2.metric("AIC", round(aic, 2))
    col3.metric("BIC", round(bic, 2))
    col4.metric("Deviance", round(deviance, 2))
    col5.metric("Pearson χ²", round(pearson, 2))

    # ======================================================
    # 7️⃣ EQUATION BUILDER
    # ======================================================

    def build_equation(model, response):

        params = model.params

        equation = f"\\log(\\widehat{{\\mathbb{{E}}}}({response})) = {round(params['Intercept'],4)}"

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

    st.subheader("Fitted Regression Equation")

    st.latex(build_equation(model, response))

    # ======================================================
    # 8️⃣ COEFFICIENT INTERPRETATION
    # ======================================================

    st.header("5️⃣ Interpretation of Coefficients")

    for term in model.params.index:

        coef = model.params[term]
        pval = model.pvalues[term]

        if term == "Intercept":

            interpretation = (
                f"When predictors are zero/reference levels, "
                f"log-mean of {response} is {coef:.4f}."
            )

        else:

            interpretation = (
                f"A one-unit increase in '{term}' multiplies "
                f"the expected response by exp({coef:.4f})."
            )

        significance = (
            "Statistically significant."
            if pval <= 0.05
            else "Not statistically significant."
        )

        st.markdown(
            f"**{term}**  \n"
            f"- Coefficient: {coef:.4f}  \n"
            f"- p-value: {pval:.4f}  \n"
            f"- {interpretation}  \n"
            f"- {significance}"
        )

    # ======================================================
    # 9️⃣ PREDICTION
    # ======================================================

    st.header("6️⃣ Prediction")

    input_dict = {}

    for var in predictors:

        if var in categorical_vars:

            input_dict[var] = st.selectbox(var, df[var].cat.categories)

        else:

            numeric_series = pd.to_numeric(df[var], errors="coerce")

            input_dict[var] = st.number_input(
                var,
                value=float(numeric_series.mean())
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
    # 🔟 PREDICTED VS ACTUAL
    # ======================================================

    st.header("7️⃣ Predicted vs Actual")

    predicted_vals = model.predict(df)

    fig2 = px.scatter(
        x=predicted_vals,
        y=df[response],
        labels={'x': 'Predicted', 'y': 'Actual'},
        title="Predicted vs Actual Values"
    )

    min_val = min(predicted_vals.min(), df[response].min())
    max_val = max(predicted_vals.max(), df[response].max())

    fig2.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="red", dash="dash")
    )

    st.plotly_chart(fig2)


if __name__ == "__main__":
    run()
```
