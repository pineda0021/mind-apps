import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.metrics import roc_curve, auc


def run():

    st.title("📘 Complementary Log–Log Regression Model")

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

    response_original = st.selectbox("Select Binary Response Variable (Y)", df.columns)

    df[response_original] = df[response_original].astype("category")

    ref_level = st.selectbox(
        "Select reference level for response (coded as 0)",
        df[response_original].cat.categories
    )

    df["response_binary"] = (df[response_original] != ref_level).astype(int)

    st.info(f"The model estimates P({response_original} ≠ '{ref_level}').")

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
    # 3️⃣ DATA PREP
    # ======================================================

    st.header("2️⃣ Response Diagnostics")

    df_model = df.copy()

    df_model[response] = pd.to_numeric(df_model[response], errors="coerce")

    df_model = df_model[[response] + predictors].dropna()

    for col in categorical_vars:
        df_model[col] = df_model[col].astype("category")

    fig = px.histogram(df_model, x=response, color=response)
    st.plotly_chart(fig)

    # ======================================================
    # 4️⃣ MODEL FITTING
    # ======================================================

    st.header("3️⃣ Model Fitting")

    model = smf.glm(
        formula=formula,
        data=df_model,
        family=sm.families.Binomial(link=sm.families.links.cloglog())
    ).fit()

    st.text(model.summary())

    # ======================================================
    # 5️⃣ LR TEST
    # ======================================================

    st.subheader("Likelihood Ratio Test")

    null_model = smf.glm(
        response + " ~ 1",
        data=df_model,
        family=sm.families.Binomial(link=sm.families.links.cloglog())
    ).fit()

    lr_stat = 2 * (model.llf - null_model.llf)
    p_value = chi2.sf(lr_stat, int(model.df_model))

    st.write(f"LR Statistic: {lr_stat:.4f}")
    st.write(f"p-value: {p_value:.6f}")

    # ======================================================
    # 6️⃣ MODEL FIT
    # ======================================================

    st.header("4️⃣ Model Fit Evaluation")

    st.write(f"AIC: {model.aic:.2f}")
    st.write(f"BIC: {model.bic:.2f}")

    # ======================================================
    # 7️⃣ EQUATION
    # ======================================================

    st.subheader("Model Equation")

    params = model.params

    eq = f"\\log(-\\log(1-p)) = {params['Intercept']:.4f}"

    for name in params.index:
        if name == "Intercept":
            continue
        coef = params[name]
        sign = "+" if coef >= 0 else "-"
        eq += f" {sign} {abs(coef):.4f} {name}"

    st.latex(eq)

    # ======================================================
    # 8️⃣ INTERPRETATION
    # ======================================================

    st.header("5️⃣ Interpretation")

    st.markdown(
    """
$$
\\log(-\\log(1-p)) = X\\beta
$$

Interpretation uses $e^{\\beta}$:

New probability = old probability raised to $e^{\\beta}$.
"""
    )

    for term in model.params.index:

        coef = model.params[term]
        pval = model.pvalues[term]
        power = np.exp(coef)

        if pval <= 0.05:
            sig = "✅ Significant"
            color = st.success
        else:
            sig = "⚠ Not significant"
            color = st.warning

        if term == "Intercept":

            st.markdown(f"### Intercept")
            st.latex(f"\\log(-\\log(1-p)) = {coef:.4f}")

        elif "C(" in term:

            st.markdown(f"### {term}")
            st.latex(f"e^{{{coef:.4f}}} = {power:.4f}")

            st.write(
                f"Compared to the reference group, probability is raised to power {power:.4f}"
            )

        else:

            st.markdown(f"### {term}")
            st.latex(f"e^{{{coef:.4f}}} = {power:.4f}")

            st.write(
                f"If {term} increases by 1, probability is raised to power {power:.4f}"
            )

        st.write(f"Coefficient = {coef:.4f}")
        st.write(f"p-value = {pval:.4f}")
        color(sig)

    # ======================================================
    # 9️⃣ PREDICTION
    # ======================================================

    st.header("6️⃣ Prediction")

    inputs = {}

    for var in predictors:
        if var in categorical_vars:
            inputs[var] = st.selectbox(var, df[var].cat.categories)
        else:
            inputs[var] = st.number_input(var, value=float(df[var].mean()))

    if st.button("Predict"):

        new_df = pd.DataFrame([inputs])

        for var in categorical_vars:
            new_df[var] = pd.Categorical(new_df[var], categories=df[var].cat.categories)

        prob = model.predict(new_df)[0]

        st.success(f"Predicted probability = {prob:.4f}")

    # ======================================================
    # 🔟 ROC
    # ======================================================

    st.header("7️⃣ ROC Curve")

    y_true = df_model[response]
    y_pred = model.predict(df_model)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    fig = px.line(x=fpr, y=tpr, title=f"AUC = {roc_auc:.3f}")
    st.plotly_chart(fig)

    # ======================================================
    # 1️⃣1️⃣ MODEL COMPARISON
    # ======================================================

    st.header("8️⃣ Model Comparison")

    logit = smf.glm(formula, df_model,
                    family=sm.families.Binomial(link=sm.families.links.logit())).fit()

    probit = smf.glm(formula, df_model,
                     family=sm.families.Binomial(link=sm.families.links.probit())).fit()

    cloglog = model

    n = len(df_model)

    rows = []

    for name, m in [("Logit", logit), ("Probit", probit), ("Cloglog", cloglog)]:

        k = m.df_model + 1
        aic = m.aic
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
        bic = -2 * m.llf + k * np.log(n)

        rows.append({"Model": name, "AIC": aic, "AICc": aicc, "BIC": bic})

    comp = pd.DataFrame(rows)

    st.dataframe(comp.style.highlight_min(axis=0))

    st.write("Best AIC:", comp.loc[comp.AIC.idxmin(), "Model"])
    st.write("Best BIC:", comp.loc[comp.BIC.idxmin(), "Model"])


if __name__ == "__main__":
    run()
