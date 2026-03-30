import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2
import statsmodels.formula.api as smf


def run():

    st.title("📘 Generalized Logit Model for Nominal Response")

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
        "Select Nominal Response Variable (Y)",
        df.columns
    )

    df[response_original] = df[response_original].astype(str).astype("category")

    response_levels = list(df[response_original].cat.categories)

    reference_level = st.selectbox(
        "Select reference level for response",
        response_levels
    )

    ordered_response_levels = [reference_level] + [
        level for level in response_levels if level != reference_level
    ]

    df[response_original] = pd.Categorical(
        df[response_original],
        categories=ordered_response_levels
    )

    choice_mapping = {i: cat for i, cat in enumerate(ordered_response_levels)}

    st.info(
        f"The model uses **{reference_level}** as the baseline response category."
    )

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
        model = smf.mnlogit(formula, data=df_model)
        res = model.fit(disp=False)

    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        return

    st.subheader("Response Category Mapping")
    mapping_df = pd.DataFrame({
        "Code": list(choice_mapping.keys()),
        "Category": list(choice_mapping.values())
    })
    st.dataframe(mapping_df, use_container_width=True)

    st.subheader("Model Summary")
    st.text(res.summary())

    # ======================================================
    # 4️⃣ LIKELIHOOD RATIO TEST
    # ======================================================

    st.subheader("Likelihood Ratio Test")

    try:
        null_model = smf.mnlogit(response_original + " ~ 1", data=df_model)
        res_null = null_model.fit(disp=False)

        ll_null = res_null.llf
        ll_model = res.llf

        dev_null = -2 * ll_null
        dev_model = -2 * ll_model
        lr_stat = dev_null - dev_model

        df_diff = int(res.df_model - res_null.df_model)
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
    bic = res.bic

    p = res.params.size
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
    # 6️⃣ EQUATION BUILDER
    # ======================================================

    def build_equations(result, baseline_label):

        params = result.params
        equations = []

        for col in params.columns:

            category_label = choice_mapping.get(col, f"Category {col}")

            equation = (
                f"\\log\\left(\\frac{{P(Y={category_label})}}{{P(Y={baseline_label})}}\\right)"
            )

            intercept_val = params.loc["Intercept", col] if "Intercept" in params.index else 0
            equation += f" = {round(intercept_val, 4)}"

            for name in params.index:

                if name == "Intercept":
                    continue

                coef = round(params.loc[name, col], 4)
                sign = "+" if coef >= 0 else "-"

                if name.startswith("C(") and "T." in name:
                    var_name = name.split("[")[0]
                    var_name = var_name.replace("C(", "").split(",")[0]
                    level = name.split("T.")[-1].replace("]", "")
                    equation += f" {sign} {abs(coef)} D_{{{var_name}={level}}}"
                else:
                    equation += f" {sign} {abs(coef)} \\cdot {name}"

            equations.append((category_label, equation))

        return equations

    st.subheader("Fitted Regression Equations (Generalized Logits)")

    equations = build_equations(res, reference_level)

    for category_label, eq in equations:
        st.markdown(f"**Baseline comparison: {category_label} vs {reference_level}**")
        st.latex(eq)

    # ======================================================
    # 7️⃣ INTERPRETATION
    # ======================================================

    st.header("4️⃣ Interpretation of Coefficients")

    params = res.params
    pvalues = res.pvalues

    for col in params.columns:

        category_label = choice_mapping.get(col, f"Category {col}")

        st.subheader(f"Logit for {category_label} vs {reference_level}")

        for term in params.index:

            coef = params.loc[term, col]
            pval = pvalues.loc[term, col]
            odds_ratio = np.exp(coef)
            percent_change = (odds_ratio - 1) * 100

            if term == "Intercept":

                interpretation = (
                    f"When all predictors are at their reference levels or zero values, "
                    f"the log-odds of choosing **{category_label}** rather than **{reference_level}** "
                    f"is **{coef:.4f}**."
                )

            elif term.startswith("C("):

                var_name = term.split("[")[0]
                var_name = var_name.replace("C(", "").split(",")[0]

                level = term.split("T.")[-1].replace("]", "")
                reference = reference_dict.get(var_name, "reference")

                direction = "increase" if percent_change >= 0 else "decrease"

                interpretation = (
                    f"For **{var_name} = {level}** relative to **{reference}**, "
                    f"the estimated odds of choosing **{category_label}** rather than **{reference_level}** "
                    f"change by **{abs(percent_change):.2f}%** "
                    f"({direction})."
                )

            else:

                direction = "increase" if percent_change >= 0 else "decrease"

                interpretation = (
                    f"For every one-unit increase in **{term}**, "
                    f"the estimated odds of choosing **{category_label}** rather than **{reference_level}** "
                    f"change by **{abs(percent_change):.2f}%** "
                    f"({direction})."
                )

            significance = (
                "Statistically significant at the 5% level."
                if pval <= 0.05
                else "Not statistically significant at the 5% level."
            )

            st.markdown(
                f"""
### {term}

- **Coefficient:** {coef:.4f}  
- **p-value:** {pval:.4f}  
- **Odds Ratio:** {odds_ratio:.4f}  

**Interpretation**

{interpretation}

**Statistical significance:** {significance}
"""
            )

    # ======================================================
    # 8️⃣ PREDICTION
    # ======================================================

    st.header("5️⃣ Prediction")

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
            prediction = res.predict(new_df)
            prediction.columns = ordered_response_levels

            st.subheader("Prediction Results")
            st.dataframe(prediction, use_container_width=True)

            predicted_class = prediction.idxmax(axis=1).iloc[0]
            predicted_prob = prediction.max(axis=1).iloc[0]

            st.success(
                f"Predicted most likely category: {predicted_class} "
                f"(probability = {predicted_prob:.4f})"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ======================================================
    # 9️⃣ PREDICTED VS ACTUAL
    # ======================================================

    st.header("6️⃣ Predicted vs Actual")

    try:
        predicted_probs = res.predict(df_model)
        predicted_probs.columns = ordered_response_levels

        predicted_class = predicted_probs.idxmax(axis=1)

        plot_df = pd.DataFrame({
            "Actual": df_model[response_original].astype(str),
            "Predicted": predicted_class.astype(str)
        })

        confusion_df = pd.crosstab(
            plot_df["Actual"],
            plot_df["Predicted"]
        ).reset_index()

        confusion_long = confusion_df.melt(
            id_vars="Actual",
            var_name="Predicted",
            value_name="Count"
        )

        fig = px.density_heatmap(
            confusion_long,
            x="Predicted",
            y="Actual",
            z="Count",
            text_auto=True,
            title="Predicted Category vs Actual Category"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not create predicted vs actual plot: {e}")


if __name__ == "__main__":
    run()
