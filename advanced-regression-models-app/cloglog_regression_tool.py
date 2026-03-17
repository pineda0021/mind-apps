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

    response = st.selectbox("Select Binary Response Variable (Y)", df.columns)

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

    df_model = df.copy()

    df_model[response] = pd.to_numeric(df_model[response], errors="coerce")

    # Keep only variables used in model and drop missing rows
    df_model = df_model[[response] + predictors].dropna()

    # Restore categorical dtype for factors
    for col in categorical_vars:
        df_model[col] = df_model[col].astype("category")

    unique_vals = df_model[response].dropna().unique()

    if not set(unique_vals).issubset({0,1}):
        st.error("Binary Logistic Regression requires response values coded as 0 and 1.")
        return

    fig = px.histogram(
        df_model,
        x=response,
        title="Distribution of Binary Response",
        color=response
    )

    st.plotly_chart(fig)

    # ======================================================
    # 4️⃣ MODEL FITTING
    # ======================================================

    st.header("3️⃣ Model Fitting")

    model = smf.glm(
        formula=formula,
        data=df_model,
        family=sm.families.Binomial()
    ).fit()

    st.subheader("Logistic Regression Summary")
    st.text(model.summary())

    # ======================================================
    # 5️⃣ LIKELIHOOD RATIO TEST
    # ======================================================

    st.subheader("Likelihood Ratio Test")

    null_model = smf.glm(
        response + " ~ 1",
        data=df_model,
        family=sm.families.Binomial()
    ).fit()

    lr_stat = 2 * (model.llf - null_model.llf)
    df_diff = int(model.df_model)
    p_value = chi2.sf(lr_stat, df_diff)

    st.write(f"LR Statistic: {lr_stat:.4f}")
    st.write(f"Degrees of Freedom: {df_diff}")
    st.write(f"p-value: {p_value:.6f}")

    # ======================================================
    # 6️⃣ MODEL FIT EVALUATION
    # ======================================================

    st.header("4️⃣ Model Fit Evaluation")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Log-Likelihood", round(model.llf,2))
    col2.metric("AIC", round(model.aic,2))
    col3.metric("BIC", round(model.bic,2))
    col4.metric("Deviance", round(model.deviance,2))
    col5.metric("Pearson χ²", round(model.pearson_chi2,2))

    # ======================================================
    # 7️⃣ EQUATION BUILDER
    # ======================================================

    def build_equation(model):

        params = model.params

        equation = f"\\log\\left(\\frac{{p}}{{1-p}}\\right) = {round(params['Intercept'],4)}"

        for name in params.index:

            if name == "Intercept":
                continue

            coef = round(params[name],4)
            sign = "+" if coef >= 0 else "-"

            equation += f" {sign} {abs(coef)} \\cdot {name}"

        return equation

    st.subheader("Logistic Regression Equation")
    st.latex(build_equation(model))

    # ======================================================
    # 8️⃣ INTERPRETATION
    # ======================================================

    st.header("5️⃣ Interpretation of Coefficients")

    for term in model.params.index:

        coef = model.params[term]
        pval = model.pvalues[term]

        if term == "Intercept":

            interpretation = "Baseline log-odds when predictors are zero/reference."

        else:

            odds_ratio = np.exp(coef)

            interpretation = (
                f"Odds Ratio = exp({coef:.4f}) = {odds_ratio:.4f}"
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
            input_dict[var] = st.number_input(var, value=float(numeric_series.mean()))

    if st.button("Predict Probability"):

        new_df = pd.DataFrame([input_dict])

        for var in categorical_vars:
            new_df[var] = pd.Categorical(new_df[var], categories=df[var].cat.categories)

        prob = model.predict(new_df)[0]

        st.subheader("Prediction Results")
        st.success(f"Predicted Probability of Y=1: {prob:.4f}")

    # ======================================================
    # 🔟 ROC CURVE
    # ======================================================

    st.header("7️⃣ ROC Curve")

    y_true = df_model[response]
    y_pred = model.predict(df_model)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    fig = px.line(
        x=fpr,
        y=tpr,
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
        title=f"ROC Curve (AUC = {roc_auc:.3f})"
    )

    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))

    st.plotly_chart(fig)

    # ======================================================
    # 1️⃣1️⃣ MODEL COMPARISON
    # ======================================================

    st.header("8️⃣ Model Comparison (Logistic vs Probit vs Cloglog)")

    logit_model = model

    probit_model = smf.glm(
        formula=formula,
        data=df_model,
        family=sm.families.Binomial(link=sm.families.links.probit())
    ).fit()

    cloglog_model = smf.glm(
        formula=formula,
        data=df_model,
        family=sm.families.Binomial(link=sm.families.links.cloglog())
    ).fit()

    models = {
        "Logistic": logit_model,
        "Probit": probit_model,
        "Cloglog": cloglog_model
    }

    n = len(df_model)

    rows = []

    for name, m in models.items():

        ll = m.llf
        k = m.df_model + 1

        aic = m.aic
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
        bic = -2 * ll + k * np.log(n)

        rows.append({
            "Model": name,
            "AIC": aic,
            "AICc": aicc,
            "BIC": bic
        })

    comparison_df = pd.DataFrame(rows)

    st.subheader("Information Criteria")

    st.dataframe(
        comparison_df.style.highlight_min(axis=0),
        use_container_width=True
    )

    best_aic = comparison_df.loc[comparison_df["AIC"].idxmin(), "Model"]
    best_bic = comparison_df.loc[comparison_df["BIC"].idxmin(), "Model"]

    st.success(f"Best model by AIC: **{best_aic}**")
    st.success(f"Best model by BIC: **{best_bic}**")

    fig2 = px.bar(
        comparison_df,
        x="Model",
        y=["AIC","AICc","BIC"],
        barmode="group",
        title="Information Criteria Comparison"
    )

    st.plotly_chart(fig2)


if __name__ == "__main__":
    run()
