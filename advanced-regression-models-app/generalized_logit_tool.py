import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2
import statsmodels.api as sm


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

    if len(response_levels) < 3:
        st.error("The nominal response variable must have at least 3 categories.")
        return

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

    df["response_code"] = df[response_original].cat.codes

    choice_mapping = {i: cat for i, cat in enumerate(ordered_response_levels)}

    st.info(
        f"The model uses **{reference_level}** as the baseline response category."
    )

    predictors = st.multiselect(
        "Select Predictor Variables (X)",
        [c for c in df.columns if c not in [response_original, "response_code"]]
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

    displayed_formula = response_original + " ~ " + " + ".join(terms)
    st.code(displayed_formula)

    # ======================================================
    # 3️⃣ MODEL FITTING
    # ======================================================

    st.header("2️⃣ Model Fitting")

    df_model = df[[response_original, "response_code"] + predictors].copy()

    for var in predictors:
        if var not in categorical_vars:
            df_model[var] = pd.to_numeric(df_model[var], errors="coerce")

    df_model = df_model.dropna()

    if df_model.empty:
        st.error("No valid rows remain after removing missing values.")
        return

    try:
        y = df_model["response_code"]

        X = pd.DataFrame(index=df_model.index)

        for var in predictors:
            if var in categorical_vars:
                dummies = pd.get_dummies(df_model[var], prefix=var, drop_first=False)

                ref = reference_dict[var]
                ref_col = f"{var}_{ref}"

                if ref_col in dummies.columns:
                    dummies = dummies.drop(columns=[ref_col])

                X = pd.concat([X, dummies], axis=1)
            else:
                X[var] = df_model[var]

        X = sm.add_constant(X, has_constant="add")
        X = X.astype(float)

        model = sm.MNLogit(y, X)
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

    st.subheader("Likelihood Ratio (Deviance) Test")

    try:
        X_null = pd.DataFrame(
            {"const": np.ones(len(df_model))},
            index=df_model.index
        )

        null_model = sm.MNLogit(y, X_null)
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
    col3.metric("AICC", round(aicc, 2) if pd.notna(aicc) else "N/A")
    col4.metric("BIC", round(bic, 2))
    col5.metric("Model Deviance", round(dev_model, 2))

    # ======================================================
    # 6️⃣ EQUATION BUILDER
    # ======================================================

    def get_nonbaseline_labels(pred_col_count, ordered_levels):
        if pred_col_count == len(ordered_levels):
            return ordered_levels
        if pred_col_count == len(ordered_levels) - 1:
            return ordered_levels[1:]
        return [f"Category {i}" for i in range(pred_col_count)]

    def build_equations(result, baseline_label, ordered_levels):

        params = result.params.copy()
        equations = []

        category_labels = get_nonbaseline_labels(params.shape[1], ordered_levels)

        for j, col in enumerate(params.columns):

            if j < len(category_labels):
                category_label = category_labels[j]
            else:
                category_label = f"Category {col}"

            if category_label == baseline_label:
                continue

            intercept_val = params.loc["const", col] if "const" in params.index else 0
            linear_part = f"{intercept_val:.4f}"

            for name in params.index:

                if name == "const":
                    continue

                coef = params.loc[name, col]
                sign = "+" if coef >= 0 else "-"

                if "_" in name and any(name.startswith(f"{v}_") for v in categorical_vars):
                    label = name.split("_", 1)[1]
                    linear_part += f" {sign} {abs(coef):.4f}\\cdot {label}"
                else:
                    linear_part += f" {sign} {abs(coef):.4f}\\cdot {name}"

            eq = (
                rf"\frac{{\widehat{{\mathbb{{P}}}}({category_label})}}"
                rf"{{\widehat{{\mathbb{{P}}}}({baseline_label})}}"
                rf"=\exp\big\{{{linear_part}\big\}}"
            )

            equations.append(eq)

        return equations

    st.subheader("From the output, the estimated generalized logit model for nominal response is:")

    equations = build_equations(res, reference_level, ordered_response_levels)

    for eq in equations:
        st.latex(eq)
    
   
    # ======================================================
    # 7️⃣ INTERPRETATION
    # ======================================================

    # ======================================================
    # 7️⃣ INTERPRETATION
    # ======================================================

    st.header("4️⃣ Interpretation of Coefficients")

    params = res.params
    pvalues = res.pvalues

    def nice_label(var, level):
        if var.lower() in ["gender", "sex"]:
            return level
        return f"{var} = {level}"

    category_labels = get_nonbaseline_labels(params.shape[1], ordered_response_levels)

    for j, col in enumerate(params.columns):

        if j < len(category_labels):
            category_label = category_labels[j]
        else:
            category_label = f"Category {col}"

        if category_label == reference_level:
            continue

        st.subheader(f"{category_label} vs {reference_level}")

        # -----------------------------
        # Significant predictors summary
        # -----------------------------
        significant_terms = []

        for term in params.index:
            if term == "const":
                continue
            if pvalues.loc[term, col] <= 0.05:
                significant_terms.append(term)

        if significant_terms:

            readable = []

            for term in significant_terms:
                if "_" in term and any(term.startswith(f"{v}_") for v in categorical_vars):
                    readable.append(term.split("_")[0])
                else:
                    readable.append(term)

            readable = list(dict.fromkeys(readable))

            if len(readable) == 1:
                sentence = f"{readable[0].capitalize()} is a significant predictor"
            elif len(readable) == 2:
                sentence = f"{readable[0].capitalize()} and {readable[1]} are significant predictors"
            else:
                sentence = ", ".join(readable[:-1]) + f", and {readable[-1]} are significant predictors"

            st.markdown(
                f"**{sentence} of odds in favor of {category_label} versus {reference_level}, "
                f"since their $p$-values are less than 0.05.**"
            )

        # -----------------------------
        # Individual interpretations
        # -----------------------------
        for term in params.index:

            coef = params.loc[term, col]
            pval = pvalues.loc[term, col]
            odds_ratio = np.exp(coef)
            percent_change = (odds_ratio - 1) * 100

            st.markdown(f"### {term}")

            if term == "const":

                st.markdown(
                    f"**When all predictors are at their reference levels, "
                    f"the log-odds of choosing {category_label} rather than {reference_level} "
                    f"is {coef:.4f}.**"
                )

            elif "_" in term and any(term.startswith(f"{v}_") for v in categorical_vars):

                var_name = term.split("_")[0]
                level = term.split("_")[1]
                reference = reference_dict.get(var_name, "reference")

                st.markdown(
                    f"**For {nice_label(var_name, level)}, the estimated odds in favor of "
                    f"{category_label} versus {reference_level} are:**"
                )

                st.latex(
                    rf"e^{{{coef:.4f}}}\cdot 100\% = {odds_ratio * 100:.2f}\%"
                )

                st.markdown(
                    f"**of those for {reference}.**"
                )

            else:

                if percent_change >= 0:
                    st.markdown(
                        f"**As {term} increases by one unit, the estimated odds grow by:**"
                    )
                else:
                    st.markdown(
                        f"**As {term} increases by one unit, the estimated odds change by:**"
                    )

                st.latex(
                    rf"(e^{{{coef:.4f}}}-1)\cdot 100\% = {percent_change:.2f}\%"
                )

                if percent_change < 0:
                    st.markdown(
                        f"**that is, decrease by {abs(percent_change):.2f}%.**"
                    )

            st.write(f"Coefficient = {coef:.4f}")
            st.write(f"p-value = {pval:.4f}")
            st.write(f"Odds Ratio = {odds_ratio:.4f}")

            if pval <= 0.05:
                st.success("Statistically significant.")
            else:
                st.warning("Not statistically significant.")

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
            X_new = pd.DataFrame(index=new_df.index)

            for var in predictors:
                if var in categorical_vars:
                    dummies = pd.get_dummies(new_df[var], prefix=var, drop_first=False)

                    ref = reference_dict[var]
                    ref_col = f"{var}_{ref}"

                    model_cols = [col for col in X.columns if col.startswith(f"{var}_")]

                    for col in model_cols:
                        if col not in dummies.columns:
                            dummies[col] = 0

                    if ref_col in dummies.columns:
                        dummies = dummies.drop(columns=[ref_col])

                    keep_cols = [col for col in model_cols if col in dummies.columns]
                    dummies = dummies[keep_cols]

                    X_new = pd.concat([X_new, dummies], axis=1)
                else:
                    X_new[var] = new_df[var]

            X_new = sm.add_constant(X_new, has_constant="add")

            for col in X.columns:
                if col not in X_new.columns:
                    X_new[col] = 0

            X_new = X_new[X.columns]
            X_new = X_new.astype(float)

            prediction = res.predict(X_new)
            prediction = pd.DataFrame(prediction)

            if prediction.shape[1] == len(ordered_response_levels):
                prediction.columns = ordered_response_levels

            elif prediction.shape[1] == len(ordered_response_levels) - 1:
                nonbaseline_levels = ordered_response_levels[1:]
                prediction.columns = nonbaseline_levels
                prediction.insert(0, reference_level, 1 - prediction.sum(axis=1))
                prediction = prediction[ordered_response_levels]

            else:
                st.error(
                    f"Unexpected number of prediction columns: {prediction.shape[1]}. "
                    f"Expected either {len(ordered_response_levels)} or {len(ordered_response_levels) - 1}."
                )
                return

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
        predicted_probs = res.predict(X)
        predicted_probs = pd.DataFrame(predicted_probs)

        if predicted_probs.shape[1] == len(ordered_response_levels):
            predicted_probs.columns = ordered_response_levels

        elif predicted_probs.shape[1] == len(ordered_response_levels) - 1:
            nonbaseline_levels = ordered_response_levels[1:]
            predicted_probs.columns = nonbaseline_levels
            predicted_probs.insert(0, reference_level, 1 - predicted_probs.sum(axis=1))
            predicted_probs = predicted_probs[ordered_response_levels]

        else:
            st.warning(
                f"Could not create predicted vs actual plot: unexpected number of prediction columns "
                f"({predicted_probs.shape[1]})."
            )
            return

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
