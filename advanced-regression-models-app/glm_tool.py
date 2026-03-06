import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro


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
    # 4. BUILD FORMULA WITH REFERENCE LEVELS
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
    # 6. MODEL FIT EVALUATION
    # ======================================================

    st.header("4️⃣ Model Fit Evaluation")

    loglik = model.llf
    aic = model.aic
    bic = model.bic
    f_pvalue = model.f_pvalue

    st.subheader("Log-Likelihood")

    st.write(f"Log-Likelihood: **{loglik:.4f}**")

    st.markdown("""
The log-likelihood measures how probable the observed data are given the fitted model parameters.

A higher log-likelihood indicates a better fit.

However, the value itself is not directly interpretable.  
It is mainly useful when comparing competing models fitted to the same dataset.
""")

    st.subheader("Information Criteria")

    col1, col2 = st.columns(2)
    col1.metric("AIC", round(aic, 2))
    col2.metric("BIC", round(bic, 2))

    st.markdown("""
AIC and BIC penalize model complexity.

Lower values indicate a better balance between goodness of fit and model complexity.
""")

    st.subheader("Overall Model Significance (F-test)")

    st.write(f"F-statistic p-value: **{f_pvalue:.6f}**")

    if f_pvalue < 0.05:
        st.success(
            "At the 5% level of significance, we reject the null hypothesis "
            "that all slope coefficients are zero. "
            "The fitted model significantly improves over an intercept-only model."
        )
    else:
        st.warning(
            "At the 5% level of significance, we fail to reject the null hypothesis. "
            "The model does not significantly improve over an intercept-only model."
        )

    # ======================================================
    # 7. MATHEMATICAL EQUATION
    # ======================================================

    def build_equation(model, response):

        params = model.params
        equation = f"\\hat{{{response}}} = {round(params['Intercept'],4)}"

        for name in params.index:

            if name == "Intercept":
                continue

            coef = round(params[name], 4)
            sign = "+" if coef >= 0 else "-"

            if "C(" in name:
                var_name = name.split("[")[0]
                var_name = var_name.replace("C(", "").split(",")[0]
                level = name.split("T.")[1].replace("]", "")
                equation += f" {sign} {abs(coef)} D_{{{var_name}={level}}}"
            else:
                equation += f" {sign} {abs(coef)} \\cdot {name}"

        return equation

    st.subheader("Fitted Regression Equation")
    st.latex(build_equation(model, response))

    # ======================================================
    # 8. INTERPRETATION
    # ======================================================

    st.subheader("Interpretation of Coefficients")

    for name, coef in model.params.items():

        if name == "Intercept":
            continue

        coef = round(coef, 4)

        if "C(" in name:
            var_name = name.split("[")[0]
            var_name = var_name.replace("C(", "").split(",")[0]
            level = name.split("T.")[1].replace("]", "")
            ref = reference_dict[var_name]

            direction = "increases" if coef > 0 else "decreases"

            st.write(
                f"For **{var_name} = {level}**, expected **{response}** "
                f"{direction} by **{abs(coef)} units** compared to "
                f"reference group (**{ref}**), holding other variables constant."
            )
        else:
            direction = "increases" if coef > 0 else "decreases"

            st.write(
                f"For each one-unit increase in **{name}**, expected "
                f"**{response}** {direction} by **{abs(coef)} units**, "
                "holding other variables constant."
            )

    # ======================================================
    # 9. PREDICTION
    # ======================================================

    st.header("5️⃣ Prediction")

    input_dict = {}

    for var in predictors:
        if var in categorical_vars:
            input_dict[var] = st.selectbox(var, df[var].cat.categories)
        else:
            input_dict[var] = st.number_input(
                var,
                value=float(df[var].mean())
            )

    if st.button("Predict"):
        new_df = pd.DataFrame([input_dict])
        prediction = model.predict(new_df)[0]
        st.success(f"Predicted {response}: {prediction:.4f}")

    # ======================================================
    # 10. PREDICTED VS ACTUAL
    # ======================================================

    st.header("6️⃣ Predicted vs Actual")

    predicted_vals = model.predict(df)

    fig2 = px.scatter(
        x=predicted_vals,
        y=df[response],
        labels={'x': 'Predicted', 'y': 'Actual'},
        title="Predicted vs Actual Values"
    )

    st.plotly_chart(fig2)
