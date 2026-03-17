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
            terms.append(f'C({var}, Treatment(reference="{ref}"))')
        else:
            terms.append(var)

    formula = response + " ~ " + " + ".join(terms)

    st.code(formula)

    # ======================================================
    # 3️⃣ DATA PREPARATION
    # ======================================================

    st.header("2️⃣ Response Diagnostics")

    df_model = df.copy()

    df_model[response] = pd.to_numeric(df_model[response], errors="coerce")

    df_model = df_model[[response] + predictors].dropna()

    for col in categorical_vars:
        df_model[col] = df_model[col].astype("category")

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
        family=sm.families.Binomial(link=sm.families.links.cloglog())
    ).fit()

    st.subheader("Cloglog Regression Summary")
    st.text(model.summary())

    # ======================================================
    # 5️⃣ LIKELIHOOD RATIO TEST
    # ======================================================

    st.subheader("Likelihood Ratio Test")

    null_model = smf.glm(
        response + " ~ 1",
        data=df_model,
        family=sm.families.Binomial(link=sm.families.links.cloglog())
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

    col1.metric("Log-Likelihood", round(model.llf, 2))
    col2.metric("AIC", round(model.aic, 2))
    col3.metric("BIC", round(model.bic, 2))
    col4.metric("Deviance", round(model.deviance, 2))
    col5.metric("Pearson χ²", round(model.pearson_chi2, 2))

    # ======================================================
    # 7️⃣ MODEL EQUATION
    # ======================================================

    st.subheader("Cloglog Regression Equation")

    params = model.params

    equation = f"\\log(-\\log(1-p)) = {round(params['Intercept'],4)}"

    for name in params.index:

        if name == "Intercept":
            continue

        coef = round(params[name], 4)
        sign = "+" if coef >= 0 else "-"

        equation += f" {sign} {abs(coef)} \\cdot {name}"

    st.latex(equation)

    # ======================================================
    # 8️⃣ INTERPRETATION
    # ======================================================

    st.header("5️⃣ Interpretation of Coefficients")

    st.markdown(
    """
    For the complementary log–log model

    $$
    \log(-\log(1-p)) = X\\beta
    $$

    coefficients are interpreted using $e^{\\beta}$.

    If a predictor changes, the **new probability equals the old probability raised to the power $e^{\\beta}$**.
    """
    )

    for term in model.params.index:

        coef = model.params[term]
        pval = model.pvalues[term]
        power = np.exp(coef)

        # significance label
        if pval <= 0.05:
            sig_text = "Statistically significant at the 5% level."
            sig_display = st.success
        else:
            sig_text = "Not statistically significant at the 5% level."
            sig_display = st.warning

        # INTERCEPT
        if term == "Intercept":

        s    t.markdown("### Intercept")

            st.latex(r"\log(-\log(1-p)) = " + f"{coef:.4f}")

            st.markdown(
            f"""
Baseline complementary log–log value when predictors are at their reference levels.

Coefficient = **{coef:.4f}**  
p-value = **{pval:.4f}**
"""
        )

        sig_display(sig_text)

    # CATEGORICAL VARIABLES
    elif "C(" in term:

        st.markdown(f"### {term}")

        st.latex(
            rf"e^{{{coef:.4f}}} = {power:.4f}"
        )

        st.markdown(
        f"""
Compared with the **reference category**, the probability of the event
is multiplied by a power of **{power:.4f}**.

Coefficient = **{coef:.4f}**  
p-value = **{pval:.4f}**
"""
        )

        sig_display(sig_text)

    # NUMERIC VARIABLES
    else:

        st.markdown(f"### {term}")

        st.latex(
            rf"e^{{{coef:.4f}}} = {power:.4f}"
        )

        st.markdown(
        f"""
If **{term} increases by one unit**, the new estimated probability equals
the old probability raised to the power **{power:.4f}**.

Coefficient = **{coef:.4f}**  
p-value = **{pval:.4f}**
"""
        )

        sig_display(sig_text)

  

    # ======================================================
    # 9️⃣ PREDICTION
    # ======================================================

    st.header("6️⃣ Prediction")

    input_dict = {}

    for var in predictors:

        if var in categorical_vars:

            input_dict[var] = st.selectbox(
                var,
                df[var].cat.categories
            )

        else:

            numeric_series = pd.to_numeric(df[var], errors="coerce")

            input_dict[var] = st.number_input(
                var,
                value=float(numeric_series.mean())
            )

    if st.button("Predict Probability"):

        new_df = pd.DataFrame([input_dict])

        for var in categorical_vars:
            new_df[var] = pd.Categorical(
                new_df[var],
                categories=df[var].cat.categories
            )

        prob = model.predict(new_df)[0]

        st.subheader("Prediction Result")

        st.success(
            f"Estimated probability of the event = **{prob:.4f}**"
        )

        st.markdown(
        f"""
Using the complementary log–log model

\\[
p = 1 - e^{{-e^{{X\\beta}}}}
\\]

the predicted probability of the event is **{prob:.4f}**.
"""
        )

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

    logit_model = smf.glm(
        formula=formula,
        data=df_model,
        family=sm.families.Binomial(link=sm.families.links.logit())
    ).fit()

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
