import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2


def run():

    st.title("📘 Logistic Regression Model (Binary Response)")

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

    model = smf.glm(
        formula=formula,
        data=df_model,
        family=sm.families.Binomial(link=sm.families.links.Logit())
    ).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # 4️⃣ LIKELIHOOD RATIO TEST
    # ======================================================

    st.subheader("Likelihood Ratio (Deviance) Test")

    null_model = smf.glm(
        response + " ~ 1",
        data=df_model,
        family=sm.families.Binomial(link=sm.families.links.Logit())
    ).fit()

    lr_stat = 2 * (model.llf - null_model.llf)
    df_diff = int(model.df_model)
    p_value = chi2.sf(lr_stat, df_diff)

    st.write(f"LR Statistic: {lr_stat:.4f}")
    st.write(f"Degrees of Freedom: {df_diff}")
    st.write(f"p-value: {p_value:.6f}")

    # ======================================================
    # 5️⃣ MODEL FIT EVALUATION
    # ======================================================

    st.header("3️⃣ Model Fit Evaluation")

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
    # 6️⃣ EQUATION BUILDER
    # ======================================================

    def clean_term_label(name):
        if name.startswith("C(") and "T." in name:
            var_name = name.split("[")[0]
            var_name = var_name.replace("C(", "").split(",")[0]
            level = name.split("T.")[1].rstrip("]")
            return f"{level}"
        return name

    def build_equation(model):
        params = model.params

        equation = r"\log\left(\frac{\pi}{1-\pi}\right)="
        equation += f"{params['Intercept']:.5f}"

        for name in params.index:
            if name == "Intercept":
                continue

            coef = params[name]
            sign = "+" if coef >= 0 else "-"
            label = clean_term_label(name)

            equation += f"{sign}{abs(coef):.5f}\\cdot {label}"

        return equation

    st.subheader(r"The fitted model for $\pi=\mathbb{P}(\mathrm{collaboration})$ is:")
    st.latex(build_equation(model))

    # ======================================================
    # 7️⃣ INTERPRETATION
    # ======================================================

    st.header("4️⃣ Interpretation of Coefficients")

    for term in model.params.index:

        coef = model.params[term]
        pval = model.pvalues[term]

        if term == "Intercept":

            interpretation_latex = (
                rf"When all predictors are at their reference levels or equal to zero, "
                rf"the log-odds of $\pi=\mathbb{{P}}(\mathrm{{collaboration}})$ is "
                rf"${coef:.4f}$."
            )

        elif term.startswith("C("):

            var_name = term.split("[")[0]
            var_name = var_name.replace("C(", "").split(",")[0]

            level = term.split("T.")[-1].replace("]", "")
            reference = reference_dict.get(var_name, "reference")

            if coef >= 0:
                interpretation_latex = (
                    rf"For {var_name} = {level}, the log-odds of "
                    rf"$\pi=\mathbb{{P}}(\mathrm{{collaboration}})$ is "
                    rf"${coef:.4f}$ higher than for {var_name} = {reference}, "
                    rf"holding the other predictors fixed."
                )
            else:
                interpretation_latex = (
                    rf"For {var_name} = {level}, the log-odds of "
                    rf"$\pi=\mathbb{{P}}(\mathrm{{collaboration}})$ is "
                    rf"${abs(coef):.4f}$ lower than for {var_name} = {reference}, "
                    rf"holding the other predictors fixed."
                )

        else:

            if coef >= 0:
                interpretation_latex = (
                    rf"For every one-unit increase in ${term}$, the log-odds of "
                    rf"$\pi=\mathbb{{P}}(\mathrm{{collaboration}})$ increases by "
                    rf"${coef:.4f}$, holding the other predictors fixed."
                )
            else:
                interpretation_latex = (
                    rf"For every one-unit increase in ${term}$, the log-odds of "
                    rf"$\pi=\mathbb{{P}}(\mathrm{{collaboration}})$ decreases by "
                    rf"${abs(coef):.4f}$, holding the other predictors fixed."
                )

        if pval <= 0.05:
            significance = r"\text{Statistically significant at the 5\% level.}"
            box_color = "#d4edda"
            border_color = "#155724"
        else:
            significance = r"\text{Not statistically significant at the 5\% level.}"
            box_color = "#fff3cd"
            border_color = "#856404"

        st.markdown(
            f"""
            <div style="
                background-color: {box_color};
                border-left: 6px solid {border_color};
                padding: 14px;
                border-radius: 8px;
                margin-bottom: 14px;
            ">
                <h4 style="margin-top: 0;">{term}</h4>
                <p><b>Coefficient:</b> {coef:.4f}</p>
                <p><b>p-value:</b> {pval:.4f}</p>
                <p><b>Interpretation:</b></p>
                <p>{interpretation_latex}</p>
                <p><b>Statistical significance:</b> ${significance}$</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ======================================================
    # 8️⃣ PREDICTION
    # ======================================================

    st.header("5️⃣ Prediction")

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

        st.subheader("Prediction Results")

        st.success(f"Predicted probability that {response}=1: {prediction:.4f}")

    # ======================================================
    # 9️⃣ PREDICTED VS ACTUAL
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


if __name__ == "__main__":
    run()
