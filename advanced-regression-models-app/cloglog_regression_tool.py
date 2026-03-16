import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm


def run():

    st.title("📘 Binary Regression Model Comparison")

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

    if not set(df[response].dropna().unique()).issubset({0, 1}):
        st.error("Response variable must be coded 0/1.")
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
            terms.append(f'C({var}, Treatment(reference="{ref}"))')

        else:

            terms.append(var)

    formula = response + " ~ " + " + ".join(terms)

    st.code(formula)

    # ======================================================
    # 3️⃣ MODEL FITTING
    # ======================================================

    st.header("2️⃣ Model Fitting")

    df_model = df.copy()

    logit_model = smf.glm(
        formula=formula,
        data=df_model,
        family=sm.families.Binomial(link=sm.families.links.Logit())
    ).fit()

    probit_model = smf.glm(
        formula=formula,
        data=df_model,
        family=sm.families.Binomial(link=sm.families.links.Probit())
    ).fit()

    cloglog_model = smf.glm(
        formula=formula,
        data=df_model,
        family=sm.families.Binomial(link=sm.families.links.CLogLog())
    ).fit()

    st.subheader("Logit Model Summary")
    st.text(logit_model.summary())

    st.subheader("Probit Model Summary")
    st.text(probit_model.summary())

    st.subheader("Cloglog Model Summary")
    st.text(cloglog_model.summary())

    # ======================================================
    # 4️⃣ INFORMATION CRITERIA
    # ======================================================

    st.header("3️⃣ Model Comparison")

    n = len(df_model)

    def AICc(model):

        k = model.df_model + 1
        aic = model.aic

        return aic + (2*k*(k+1))/(n-k-1)

    comparison = pd.DataFrame({

        "Model": ["Logit","Probit","Cloglog"],

        "AIC": [
            logit_model.aic,
            probit_model.aic,
            cloglog_model.aic
        ],

        "AICc":[
            AICc(logit_model),
            AICc(probit_model),
            AICc(cloglog_model)
        ],

        "BIC":[
            logit_model.bic,
            probit_model.bic,
            cloglog_model.bic
        ]

    })

    st.dataframe(comparison.round(4))

    best_model = comparison.loc[comparison["AIC"].idxmin(),"Model"]

    st.success(f"Best model according to AIC: **{best_model}**")

    # ======================================================
    # 5️⃣ CLOGLOG INTERPRETATION
    # ======================================================

    st.header("4️⃣ Interpretation (Cloglog Model)")

    for term in cloglog_model.params.index:

        coef = cloglog_model.params[term]
        pval = cloglog_model.pvalues[term]

        exp_beta = np.exp(coef)

        if term == "Intercept":

            interpretation = (
                "Intercept corresponds to the baseline complementary log-log "
                "transformation of the probability."
            )

        elif term.startswith("C("):

            var_name = term.split("[")[0]
            var_name = var_name.replace("C(","").split(",")[0]

            level = term.split("T.")[-1].replace("]","")
            reference = reference_dict.get(var_name,"reference")

            interpretation = (
                f"The estimated probability of competition for "
                f"**{var_name} = {level}** equals the probability of competition "
                f"for **{var_name} = {reference}** raised to the power  \n"
                f"**exp({coef:.4f}) = {exp_beta:.4f}**."
            )

        else:

            interpretation = (
                f"If **{term}** increases by one unit, the new estimated probability "
                f"of competition equals the old probability raised to the power  \n"
                f"**exp({coef:.4f}) = {exp_beta:.4f}**."
            )

        significance = (
            "Statistically significant at the 5% level."
            if pval <= 0.05
            else "Not statistically significant."
        )

        st.markdown(f"""
### {term}

Coefficient: **{coef:.4f}**

p-value: **{pval:.4f}**

{interpretation}

**{significance}**
""")

    # ======================================================
    # 6️⃣ PREDICTED VS ACTUAL
    # ======================================================

    st.header("5️⃣ Predicted vs Actual")

    predicted_vals = cloglog_model.predict(df_model)

    fig = px.scatter(
        x=predicted_vals,
        y=df_model[response],
        labels={"x":"Predicted Probability","y":"Actual"},
        title="Predicted Probability vs Actual"
    )

    st.plotly_chart(fig)


if __name__ == "__main__":
    run()
