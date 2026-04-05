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
    # 5️⃣  LIKELIHOOD RATIO TEST
    # ======================================================

    st.subheader("Likelihood Ratio (Deviance) Test")

    null_model = smf.glm(
        response + " ~ 1",
        data=df_model,
        family=sm.families.Binomial(link=sm.families.links.cloglog())
    ).fit()

    lr_stat = 2 * (model.llf - null_model.llf)
    df_diff = int(model.df_model)
    p_value = chi2.sf(lr_stat, int(model.df_model))

    st.write(f"LR Statistic: {lr_stat:.4f}")
    st.write(f"Degrees of Freedom: {df_diff}")
    st.write(f"p-value: {p_value:.6f}")

    # ======================================================
    # 6️⃣ MODEL FIT
    # ======================================================

    st.header("4️⃣ Model Fit Evaluation")
    loglik = model.llf
    aic = model.aic
    bic = model.bic_llf
    deviance = model.deviance
    pearson = model.pearson_chi2

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Log-Likelihood", round(loglik, 2))
    col2.metric("AIC", round(aic, 2))
    col3.metric("BIC", round(bic, 2))
    col4.metric("Deviance", round(deviance, 2))
    col5.metric("Pearson χ²", round(pearson, 2))

    # ======================================================
    # 7️⃣ EQUATION
    # ======================================================

    st.subheader("Model Equation")

    def clean_term_label(name):
        if name.startswith("C(") and "T." in name:
            return name.split("T.")[1].rstrip("]")
        return name

    params = model.params

    linear_part = f"{params['Intercept']:.5f}"

    for name in params.index:
        if name == "Intercept":
            continue
        coef = params[name]
        sign = "+" if coef >= 0 else "-"
        label = clean_term_label(name)
        linear_part += f" {sign} {abs(coef):.4f}\\cdot {label}"

    st.markdown("**From the output, the estimated complement log-log model is:**")

    st.latex(
        r"1-\widehat{\pi}"
        r"=1-\widehat{\mathbb{P}}(\mathrm{collaboration})"
        r"=\widehat{\mathbb{P}}(\mathrm{competition})"
        r"=\exp\left\{-\exp\left\{-"
        + linear_part +
        r"\right\}\right\}"
    )

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
            sig = " Significant"
            color = st.success
        else:
            sig = " Not significant"
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
    # 🔟 PREDICTED VS ACTUAL
    # ======================================================

    st.header("6️⃣ Predicted vs Actual")

    predicted_vals = model.predict(df_model)

    fig2 = px.scatter(
        x=predicted_vals,
        y=df_model[response],
        labels={'x': 'Predicted Probability', 'y': 'Actual'},
        title="Predicted Probability vs Actual"
    )

    fig2.add_hline(y=0.5, line_dash="dash")

    x_vals = np.linspace(0, 1, 200)
    sigmoid = 1 / (1 + np.exp(-10 * (x_vals - 0.5)))

    fig2.add_scatter(
        x=x_vals,
        y=sigmoid,
        mode="lines",
        line=dict(color="red", width=3),
        name="Sigmoid Curve"
    )

    st.plotly_chart(fig2, use_container_width=True)

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
