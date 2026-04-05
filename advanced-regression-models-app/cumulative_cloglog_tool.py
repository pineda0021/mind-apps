import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel


# ======================================================
# Custom Complementary Log-Log Distribution
# ======================================================

class r_cloglog_gen(stats.rv_continuous):
    def _cdf(self, x):
        return 1.0 - np.exp(-np.exp(x))

    def _pdf(self, x):
        return np.exp(x - np.exp(x))

    def _ppf(self, q):
        return np.log(-np.log(1.0 - q))


r_cloglog = r_cloglog_gen(name="r_cloglog")


def run():

    st.title("📘 Cumulative Complementary Log-Log Model (Ordinal Response)")

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
            distr=r_cloglog
        )

        res = model.fit(method="bfgs", disp=False)

    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        return

    st.subheader("Model Summary")
    st.text(res.summary())

    # ======================================================
    # 4️⃣ LIKELIHOOD RATIO TEST
    # ======================================================

    st.subheader("Likelihood Ratio (Deviance) Test")

    try:
        null_model = OrderedModel(
            df_model[response_original],
            exog=None,
            distr=r_cloglog
        )

        res_null = null_model.fit(method="newton", tol=1e-10, disp=False)

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
        ll_null = np.nan
        dev_model = -2 * res.llf
        lr_stat = np.nan
        df_diff = np.nan
        p_value = np.nan

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
    col3.metric("AICc", round(aicc, 2) if pd.notna(aicc) else "N/A")
    col4.metric("BIC", round(bic, 2))
    col5.metric("Model Deviance", round(dev_model, 2) if pd.notna(dev_model) else "N/A")


    # ======================================================
    # 7️⃣ INTERPRETATION (Aligned LaTeX Block)
    # ======================================================

    st.header("4️⃣ Interpretation of Coefficients")

    lines = []

    for term in res.params.index:

        coef = res.params[term]
        pval = res.pvalues[term]
        exp_beta = np.exp(coef)

        if "/" in term:
            continue  # skip thresholds in this block

        elif term.startswith("C("):

            var_name = term.split("[")[0]
            var_name = var_name.replace("C(", "").split(",")[0]

            level = term.split("T.")[-1].replace("]", "")
            reference = reference_dict.get(var_name, "reference")

            line = (
                rf"\text{{For }} {var_name} = {level},\ "
                rf"\text{{the estimated probability is that for }} {reference}\ "
                rf"\text{{raised to the power }} e^{{{coef:.4f}}} = {exp_beta:.4f}."
            )

        else:

            line = (
                rf"\text{{For a one-unit increase in }} {term},\ "
                rf"\text{{the estimated probability is multiplied by }} "
                rf"e^{{{coef:.4f}}} = {exp_beta:.4f}."
            )

        lines.append(line)

    # Build aligned LaTeX block
    latex_block = r"\begin{aligned}" + " \\\\ ".join(lines) + r"\end{aligned}"

    st.latex(latex_block)


    # ======================================================
    # 7️⃣ INTERPRETATION (Aligned LaTeX Block)
    # ======================================================

    st.header("4️⃣ Interpretation of Coefficients")

    lines = []

    for term in res.params.index:

        coef = res.params[term]
        pval = res.pvalues[term]
        exp_beta = np.exp(coef)

        if "/" in term:
            continue  # skip thresholds in this block

        elif term.startswith("C("):

            var_name = term.split("[")[0]
            var_name = var_name.replace("C(", "").split(",")[0]

            level = term.split("T.")[-1].replace("]", "")
            reference = reference_dict.get(var_name, "reference")

            line = (
                rf"\text{{For }} {var_name} = {level},\ "
                rf"\text{{the estimated probability is that for }} {reference}\ "
                rf"\text{{raised to the power }} e^{{{coef:.4f}}} = {exp_beta:.4f}."
            )

        else:

            line = (
                rf"\text{{For a one-unit increase in }} {term},\ "
                rf"\text{{the estimated probability is multiplied by }} "
                rf"e^{{{coef:.4f}}} = {exp_beta:.4f}."
            )

        lines.append(line)

    # Build aligned LaTeX block
    latex_block = r"\begin{aligned}" + " \\\\ ".join(lines) + r"\end{aligned}"

    st.latex(latex_block)
   
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

    # ======================================================
    # 🔟 BEST-FITTED MODEL COMPARISON
    # ======================================================

    st.header("7️⃣ Best-Fitted Model Comparison")

    try:
        model_logit = OrderedModel.from_formula(
            formula,
            data=df_model,
            distr="logit"
        )
        res_logit = model_logit.fit(method="newton", disp=False)

        model_probit = OrderedModel.from_formula(
            formula,
            data=df_model,
            distr="probit"
        )
        res_probit = model_probit.fit(method="newton", disp=False)

        model_cloglog = OrderedModel.from_formula(
            formula,
            data=df_model,
            distr=r_cloglog
        )
        res_cloglog = model_cloglog.fit(method="bfgs", disp=False)

        def calc_aicc(result):
            k = len(result.params)
            n_obs = result.nobs
            if n_obs - k - 1 > 0:
                return result.aic + ((2 * k * (k + 1)) / (n_obs - k - 1))
            return np.nan

        comparison_df = pd.DataFrame({
            "Model": [
                "Cumulative Logit",
                "Cumulative Probit",
                "Cumulative Complementary Log-Log"
            ],
            "AIC": [
                res_logit.aic,
                res_probit.aic,
                res_cloglog.aic
            ],
            "AICc": [
                calc_aicc(res_logit),
                calc_aicc(res_probit),
                calc_aicc(res_cloglog)
            ],
            "BIC": [
                res_logit.bic,
                res_probit.bic,
                res_cloglog.bic
            ]
        })

        comparison_df["AIC"] = comparison_df["AIC"].round(4)
        comparison_df["AICc"] = comparison_df["AICC"].round(4)
        comparison_df["BIC"] = comparison_df["BIC"].round(4)

        st.subheader("Model Selection Criteria")
        st.dataframe(comparison_df, use_container_width=True)

        best_aic_model = comparison_df.loc[comparison_df["AIC"].idxmin(), "Model"]
        best_aicc_model = comparison_df.loc[comparison_df["AICC"].idxmin(), "Model"]
        best_bic_model = comparison_df.loc[comparison_df["BIC"].idxmin(), "Model"]

        st.markdown(
            f"""
**Best model by AIC:** {best_aic_model}  

**Best model by AICC:** {best_aicc_model}  

**Best model by BIC:** {best_bic_model}
"""
        )

        if (
            best_aic_model == best_aicc_model
            and best_aic_model == best_bic_model
        ):
            st.success(
                f"The **{best_aic_model}** has the smallest AIC, AICc, and BIC values, "
                f"so it fits the data the best among the three ordinal models and would typically be preferred."
            )
        else:
            st.info(
                "The information criteria do not all select the same model. "
                "In that case, you may compare interpretability, theoretical suitability, "
                "and predictive performance before choosing a final model."
            )

        st.markdown(
            """
For these criteria, **smaller values indicate better fit relative to model complexity**.  
So the preferred model is the one with the **smallest AIC, AICc, and BIC** values.
"""
        )

    except Exception as e:
        st.warning(f"Could not compare the cumulative logit, probit, and cloglog models: {e}")


if __name__ == "__main__":
    run()
