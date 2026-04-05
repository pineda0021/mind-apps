import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2
from statsmodels.miscmodels.ordinal_model import OrderedModel


def run():

    st.title("📘 Cumulative Probit Model (Ordinal Response)")

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
        "Select Ordinal Response Variable (Y)",
        df.columns
    )

    df[response_original] = df[response_original].astype(str)

    raw_levels = list(pd.Series(df[response_original].dropna().unique()).astype(str))

    st.markdown(
        """
Enter the response categories in the correct order from lowest to highest,  
separated by commas.
"""
    )

    default_order_text = ",".join(raw_levels)

    order_text = st.text_input(
        "Ordered response levels",
        value=default_order_text
    )

    response_order = [x.strip() for x in order_text.split(",") if x.strip() != ""]

    if len(response_order) < 3:
        st.warning("Please provide at least 3 ordered categories for the ordinal response.")
        return

    invalid_levels = sorted(set(df[response_original].dropna()) - set(response_order))

    if invalid_levels:
        st.error(
            f"These response values appear in the data but are not included in your ordered levels: {invalid_levels}"
        )
        return

    df[response_original] = pd.Categorical(
        df[response_original],
        categories=response_order,
        ordered=True
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
        model = OrderedModel.from_formula(
            formula,
            data=df_model,
            distr="probit"
        )

        res = model.fit(method="newton", disp=False)

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
        null_model = OrderedModel(
            df_model[response_original],
            exog=None,
            distr="probit"
        )

        res_null = null_model.fit(method="newton", disp=False)

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
    bic = res.bic

    p = len(res.params)
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
    
    def clean_term_label(name):
        if name.startswith("C(") and "T." in name:
            return name.split("T.")[-1].replace("]", "")
        return name
    
    def join_categories(cats):
        if len(cats) == 1:
            return cats[0]
        elif len(cats) == 2:
            return cats[0] + r",\mathrm{or}," + cats[1]
        else:
            return r",".join(cats[:-1]) + r",\mathrm{or}," + cats[-1]
    
    def build_equations(result, response_levels):
    
        params = result.params
    
        slope_terms = []
        threshold_terms = []
    
        for name in params.index:
            if "/" in name:
                threshold_terms.append((name, params[name]))
            else:
                slope_terms.append((name, params[name]))
    
        # ensure correct threshold order
        threshold_terms = sorted(
            threshold_terms,
            key=lambda x: list(response_levels).index(x[0].split("/")[0])
        )
    
        linear_part = ""
    
        for name, coef in slope_terms:
            coef_r = round(coef, 4)
            sign = "+" if coef_r >= 0 else "-"
            label = clean_term_label(name)
    
            linear_part += f" {sign} {abs(coef_r):.4f}\\cdot {label}"
    
        # reconstruct actual thresholds
        actual_thresholds = []
    
        for i, (thresh_name, thresh_val) in enumerate(threshold_terms):
            if i == 0:
                actual_val = float(thresh_val)
            else:
                actual_val = actual_thresholds[-1][1] + np.exp(float(thresh_val))
    
            actual_thresholds.append((thresh_name, actual_val, float(thresh_val)))
    
        equations = []
    
        for thresh_name, actual_val, raw_val in actual_thresholds:
    
            thresh_r = round(actual_val, 4)
            boundary = thresh_name.split("/")[0]
    
            try:
                idx = list(response_levels).index(boundary)
                cumulative_levels = list(response_levels)[:idx + 1]
            except ValueError:
                cumulative_levels = [boundary]
    
            left_side = join_categories(cumulative_levels)
    
            eq = (
                rf"\widehat{{P}}({left_side})"
                rf"=\Phi\left({thresh_r:.4f}{linear_part}\right)"
            )
    
            equations.append(eq)
    
        return equations, actual_thresholds
    
    st.subheader("Fitted Regression Equations (Cumulative Probits)")
    
    response_levels = list(df[response_original].cat.categories)
    equations, actual_thresholds = build_equations(res, response_levels)
    
    for eq in equations:
        st.latex(eq)
    
    # -----------------------------------
    # Threshold Reconstruction
    # -----------------------------------
    st.subheader("Threshold Reconstruction")

    if len(actual_thresholds) > 0:
        first_name, first_actual, _ = actual_thresholds[0]
    
        st.latex(
            rf"\text{{{first_name}}} = {first_actual:.4f}"
        )
    
    for i in range(1, len(actual_thresholds)):
        current_name, current_actual, current_raw = actual_thresholds[i]
    
        pieces = [f"{actual_thresholds[0][1]:.4f}"]
        for k in range(1, i + 1):
            pieces.append(f"{actual_thresholds[k][2]:.4f}")
    
        sum_string = " + ".join(pieces)
    
        st.latex(
            rf"\text{{{current_name}}} = {sum_string} = {current_actual:.4f}"
        )
    
    st.markdown(
        r"**Note:** R and Python outputs may differ in appearance, but they represent the same threshold values."
    )

    # ======================================================
    # 7️⃣ INTERPRETATION
    # ======================================================

    st.header("4️⃣ Interpretation of Coefficients")

    for term in res.params.index:

        coef = res.params[term]
        pval = res.pvalues[term]

        st.markdown(f"### {term}")

        if "/" in term:

            interpretation = (
                f"**This is a threshold (cutpoint) parameter for the ordinal response. "
                f"It separates adjacent cumulative categories on the latent probit scale.**"
            )

        elif term.startswith("C("):

            var_name = term.split("[")[0]
            var_name = var_name.replace("C(", "").split(",")[0]

            level = term.split("T.")[-1].replace("]", "")
            reference = reference_dict.get(var_name, "reference")

            # nicer phrasing depending on variable
            if var_name.lower() in ["gender", "sex"]:
                interpretation = (
                    f"**The z-score of the estimated probability of worse health for "
                    f"{level} is larger than that for {reference} by {coef:.4f}.**"
                )
            else:
                interpretation = (
                    f"**People with {var_name} = {level} have a z-score that is "
                    f"{'larger' if coef >= 0 else 'smaller'} than that for "
                    f"{var_name} = {reference} by {abs(coef):.4f}.**"
                )

        else:

            if coef >= 0:
                interpretation = (
                    f"**As {term} increases by one unit, the z-score of the estimated "
                    f"probability of worse health increases by {coef:.4f}.**"
                )
            else:
                interpretation = (
                    f"**As {term} increases by one unit, the z-score of the estimated "
                    f"probability of worse health decreases by {abs(coef):.4f}.**"
                )

        st.markdown(interpretation)

        st.write(f"Coefficient = {coef:.4f}")
        st.write(f"p-value = {pval:.4f}")

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
            prediction = res.predict(new_df)
            prediction.columns = response_order

            st.subheader("Prediction Results")
            st.dataframe(prediction)

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
        predicted_probs.columns = response_order

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
