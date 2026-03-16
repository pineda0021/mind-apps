import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.metrics import roc_curve, auc


def run():

    st.title("📘 Binary Logistic Regression Model")

    # 1️⃣ DATA UPLOAD
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # 2️⃣ MODEL SPECIFICATION
    st.header("1️⃣ Model Specification")

    response_original = st.selectbox(
        "Select Binary Response Variable (Y)", df.columns
    )

    df[response_original] = df[response_original].astype("category")

    ref_level = st.selectbox(
        "Select reference level for response (coded as 0)",
        df[response_original].cat.categories
    )

    df["response_binary"] = (df[response_original] != ref_level).astype(int)

    st.info(
        f"The model estimates the probability that "
        f"{response_original} ≠ '{ref_level}'."
    )

    response = "response_binary"

    predictors = st.multiselect(
        "Select Predictor Variables (X)",
        [c for c in df.columns if c != response_original]
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

    terms = []

    for var in predictors:
        if var in categorical_vars:
            ref = reference_dict[var]
            terms.append(
                f'C({var}, Treatment(reference="{ref}"))'
            )
        else:
            terms.append(var)

    formula = response + " ~ " + " + ".join(terms)

    st.code(formula)

    # 3️⃣ RESPONSE DIAGNOSTICS
    st.header("2️⃣ Response Diagnostics")

    fig = px.histogram(
        df,
        x=response,
        color=response,
        title="Distribution of Binary Response"
    )

    st.plotly_chart(fig)

    # 4️⃣ MODEL FITTING
    st.header("3️⃣ Model Fitting")

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial()
    ).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # 5️⃣ LIKELIHOOD RATIO TEST
    st.subheader("Likelihood Ratio Test")

    null_model = smf.glm(
        response + " ~ 1",
        data=df,
        family=sm.families.Binomial()
    ).fit()

    lr_stat = 2 * (model.llf - null_model.llf)
    df_diff = int(model.df_model)
    p_value = chi2.sf(lr_stat, df_diff)

    st.write(f"LR Statistic: {lr_stat:.4f}")
    st.write(f"Degrees of Freedom: {df_diff}")
    st.write(f"p-value: {p_value:.6f}")

    # 6️⃣ MODEL FIT EVALUATION
    st.header("4️⃣ Model Fit Evaluation")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Log-Likelihood", round(model.llf, 2))
    col2.metric("AIC", round(model.aic, 2))
    col3.metric("BIC", round(model.bic, 2))
    col4.metric("Deviance", round(model.deviance, 2))
    col5.metric("Pearson χ²", round(model.pearson_chi2, 2))

    # 7️⃣ EQUATION
    st.subheader("Logistic Regression Equation")

    params = model.params
    intercept = params.get("Intercept", params.get("const", 0))

    equation = f"log(p/(1-p)) = {round(intercept,4)}"

    for name in params.index:
        if name in ["Intercept", "const"]:
            continue

        coef = round(params[name], 4)
        sign = "+" if coef >= 0 else "-"
        equation += f" {sign} {abs(coef)}·{name}"

    st.latex(equation)

    # 8️⃣ INTERPRETATION
    st.header("5️⃣ Interpretation of Coefficients")

    for term in params.index:

        coef = params[term]
        pval = model.pvalues[term]

        if term in ["Intercept", "const"]:
            interpretation = "Baseline log-odds."
        else:
            odds = np.exp(coef)
            interpretation = (
                f"Odds ratio = exp({coef:.4f}) = {odds:.4f}"
            )

        sig = (
            "Statistically significant"
            if pval <= 0.05
            else "Not statistically significant"
        )

        st.markdown(
            f"**{term}**  \n"
            f"- Coefficient: {coef:.4f}  \n"
            f"- p-value: {pval:.4f}  \n"
            f"- {interpretation}  \n"
            f"- {sig}"
        )

    # 9️⃣ PREDICTION
    st.header("6️⃣ Prediction")

    inputs = {}

    for var in predictors:

        if var in categorical_vars:
            inputs[var] = st.selectbox(
                var, df[var].cat.categories
            )
        else:
            inputs[var] = st.number_input(
                var, value=float(df[var].mean())
            )

    if st.button("Predict Probability"):

        new_df = pd.DataFrame([inputs])

        for var in categorical_vars:
            new_df[var] = pd.Categorical(
                new_df[var],
                categories=df[var].cat.categories
            )

        prob = model.predict(new_df)[0]

        st.success(
            f"Predicted Probability of Y=1: {prob:.4f}"
        )

   # 🔟 SIGMOID FUNCTION
    st.header("7️⃣ Predicted vs Actual")

    predicted_vals = model.predict(df_model)

    fig2 = px.scatter(
        x=predicted_vals,
        y=df_model[response],
        labels={'x': 'Predicted Probability', 'y': 'Actual'},
        title="Predicted Probability vs Actual"
    )

    # reference line
    fig2.add_hline(y=0.5, line_dash="dash")

    # ======================================
    # Sigmoid curve
    # ======================================

    x_vals = np.linspace(0, 1, 200)
    sigmoid = 1 / (1 + np.exp(-10*(x_vals - 0.5)))  # centered sigmoid

    fig2.add_scatter(
        x=x_vals,
        y=sigmoid,
        mode="lines",
        line=dict(color="red", width=3),
        name="Sigmoid Curve"
    )

    st.plotly_chart(fig2)

if __name__ == "__main__":
    run()
