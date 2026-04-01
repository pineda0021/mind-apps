import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2


def run():

    st.title("📘 Poisson Regression Model (Count Response)")

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
        "Select Count Response Variable (Y)",
        df.columns
    )

    df[response_original] = pd.to_numeric(df[response_original], errors="coerce")

    if (df[response_original].dropna() < 0).any():
        st.error("The Poisson response variable must be nonnegative.")
        return

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
        df[col] = df[col].astype(str).astype("category")

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

    formula = response_original + " ~ " + " + ".join(terms)

    st.code(formula)

    st.info(
        "This model uses the **log link**, so coefficients describe multiplicative effects "
        "on the expected count."
    )

    # ======================================================
    # 3️⃣ MODEL FITTING
    # ======================================================

    st.header("2️⃣ Model Fitting")

    df_model = df[[response_original] + predictors].copy()

    for var in predictors:
        if var not in categorical_vars:
            df_model[var] = pd.to_numeric(df_model[var], errors="coerce")

    df_model = df_model.dropna()

    if df_model.empty:
        st.error("No valid rows remain after removing missing values.")
        return

    try:
        model = smf.glm(
            formula=formula,
            data=df_model,
            family=sm.families.Poisson(link=sm.families.links.Log())
        )

        res = model.fit()

    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        return

    st.subheader("Model Summary")
    st.text(res.summary())

    # ======================================================
    # 4️⃣ LIKELIHOOD RATIO TEST
    # ======================================================

    st.subheader("Likelihood Ratio Test")

    try:
        null_model = smf.glm(
            formula=response_original + " ~ 1",
            data=df_model,
            family=sm.families.Poisson(link=sm.families.links.Log())
        )

        res_null = null_model.fit()

        ll_null = res_null.llf
        ll_model = res.llf

        dev_null = -2 * ll_null
        dev_model = -2 * ll_model
        lr_stat = dev_null - dev_model

        df_diff = len(res.params) - len(res_null.params)
        p_value = chi2.sf(lr_stat, df=df_diff)

        st.write(f"Null Deviance: {dev_null:.4f}")
        st.write(f"Model Deviance: {dev_model:.4f}")
        st.write(f"LR Statistic: {lr_stat:.4f}")
        st.write(f"Degrees of Freedom: {df_diff}")
        st.write(f"p-value: {p_value:.6f}")

    except Exception as e:
        st.warning(f"Could not compute likelihood ratio test: {e}")
        dev_model = -2 * res.llf

    # ======================================================
    # 5️⃣ MODEL FIT EVALUATION
    # ======================================================

    st.header("3️⃣ Model Fit Evaluation")

    loglik = res.llf
    aic = res.aic
    bic = res.bic_llf

    p = len(res.params)
    n = res.nobs

    if n - p - 1 > 0:
        aicc = aic + ((2 * p * (p + 1)) / (n - p - 1))
    else:
        aicc = np.nan

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Log-Likelihood", round(loglik, 2))
    col2.metric("AIC", round(aic, 2))
    col3.metric("AICc", round(aicc, 2) if pd.notna(aicc) else "N/A")
    col4.metric("BIC", round(bic, 2))
    col5.metric("Model Deviance", round(dev_model, 2))

    # ======================================================
    # 6️⃣ FITTED MODEL
    # ======================================================

    st.header("4️⃣ Fitted Model")

    def build_rate_equation(model_result):
        params = model_result.params

        pieces = []

        for name in params.index:

            coef = round(params[name], 4)

            if name == "Intercept":
                pieces.append(f"{coef}")
                continue

            if name.startswith("C(") and "T." in name:
                var_name = name.split("[")[0].replace("C(", "").split(",")[0]
                level = name.split("T.")[-1].replace("]", "")
                term_label = f"D_{{{var_name}={level}}}"
            else:
                term_label = name

            if coef >= 0:
                pieces.append(f"+ {abs(coef)}\\cdot {term_label}")
            else:
                pieces.append(f"- {abs(coef)}\\cdot {term_label}")

        inside = " ".join(pieces)

        return f"\\widehat{{\\lambda}} = \\exp\\left({inside}\\right)"

    st.markdown("**In the fitted model, the estimated rate is:**")
    st.latex(build_rate_equation(res))

    # ======================================================
    # 7️⃣ INTERPRETATION
    # ======================================================

    st.header("5️⃣ Interpretation")

    st.markdown("Interpretation uses $e^{\\beta}$:")
    st.markdown("New expected count = old expected count multiplied by $e^{\\beta}$.")

    for term in res.params.index:

        coef = res.params[term]
        pval = res.pvalues[term]
        exp_coef = np.exp(coef)

        if term.startswith("C("):
            var_name = term.split("[")[0].replace("C(", "").split(",")[0]
            level = term.split("T.")[-1].replace("]", "")
            label = f"{var_name}[{level}]"
        else:
            label = term

        st.subheader(label)
        st.latex(f"e^{{{coef:.4f}}} = {exp_coef:.4f}")

        if term == "Intercept":

            st.write(
                f"When all predictors are at zero and categorical predictors are at their reference levels, "
                f"the expected count is {exp_coef:.4f}."
            )

        elif term.startswith("C("):

            var_name = term.split("[")[0].replace("C(", "").split(",")[0]
            level = term.split("T.")[-1].replace("]", "")
            ref = reference_dict.get(var_name, "reference")

            st.write(
                f"For {var_name} = {level} relative to {ref}, the expected count is multiplied by {exp_coef:.4f}."
            )

        else:

            st.write(
                f"If {label} increases by 1, the expected count is multiplied by {exp_coef:.4f}."
            )

        st.write(f"Coefficient = {coef:.4f}")
        st.write(f"p-value = {pval:.4f}")

        if pval <= 0.05:
            st.success("Significant")
        else:
            st.warning("Not significant")

    # ======================================================
    # 8️⃣ PREDICTION
    # ======================================================

    st.header("6️⃣ Prediction")

    input_dict = {}

    for var in predictors:

        if var in categorical_vars:

            input_dict[var] = st.selectbox(
                var,
                list(df[var].cat.categories)
            )

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

        for var in predictors:
            if var not in categorical_vars:
                new_df[var] = pd.to_numeric(new_df[var], errors="coerce")

        try:
            prediction = res.predict(new_df)[0]

            st.subheader("Prediction Results")
            st.success(f"Predicted expected count for {response_original}: {prediction:.4f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ======================================================
    # 9️⃣ PREDICTED VS ACTUAL
    # ======================================================

    st.header("7️⃣ Predicted vs Actual")

    try:
        predicted_vals = res.predict(df_model)

        plot_df = pd.DataFrame({
            "Predicted": predicted_vals,
            "Actual": df_model[response_original]
        })

        fig = px.scatter(
            plot_df,
            x="Predicted",
            y="Actual",
            title="Predicted Count vs Actual Count",
            labels={"Predicted": "Predicted Count", "Actual": "Actual Count"},
            trendline="ols"
        )

        min_val = min(plot_df["Predicted"].min(), plot_df["Actual"].min())
        max_val = max(plot_df["Predicted"].max(), plot_df["Actual"].max())

        fig.add_shape(
            type="line",
            x0=min_val,
            y0=min_val,
            x1=max_val,
            y1=max_val,
            line=dict(dash="dash")
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not create predicted vs actual plot: {e}")


if __name__ == "__main__":
    run()
