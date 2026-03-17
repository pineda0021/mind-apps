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

    comparison = compare_models(
        models=[logit_model, probit_model, cloglog_model],
        model_names=["Logit", "Probit", "Cloglog"],
        n=n
    )

    st.dataframe(comparison.round(4))

    best_aic = comparison.loc[comparison["AIC"].idxmin(), "Model"]
    best_bic = comparison.loc[comparison["BIC"].idxmin(), "Model"]

    st.success(f"Best model by AIC: **{best_aic}**")
    st.info(f"Best model by BIC: **{best_bic}**")


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
    # 6️⃣ PREDICTION (Cloglog)
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

    if st.button("Predict Probability (Cloglog)"):
        new_df = pd.DataFrame([input_dict])

        for var in categorical_vars:
            new_df[var] = pd.Categorical(
                new_df[var],
                categories=df[var].cat.categories
            )

        # predicted linear predictor
        eta = cloglog_model.predict(new_df, linear=True)[0]

        # predicted probability
        p_pred = 1 - np.exp(-np.exp(eta))

        st.subheader("Prediction Results")
        st.success(f"Predicted probability of competition: {p_pred:.4f}")
        
    # ======================================================
    # 7️⃣ PREDICTED VS ACTUAL
    # ======================================================

    st.header("7️⃣ Predicted vs Actual")

    predicted_vals = cloglog_model.predict(df_model)

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
