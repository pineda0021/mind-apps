import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import patsy
from scipy.stats import chi2
from scipy.special import expit
from statsmodels.discrete.count_model import ZeroInflatedPoisson


def run():

    st.title("📘 Zero-inflated Poisson Regression Model (Count Response)")

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

    if df[response_original].isna().all():
        st.error("The response variable must contain numeric values.")
        return

    if (df[response_original].dropna() < 0).any():
        st.error("The zero-inflated Poisson response variable must be nonnegative.")
        return

    all_predictors = [c for c in df.columns if c != response_original]

    count_predictors = st.multiselect(
        "Select Count-Model Predictor Variables (X)",
        all_predictors
    )

    inflation_predictors = st.multiselect(
        "Select Zero-Inflation Predictor Variables (Z)",
        all_predictors
    )

    if not count_predictors:
        st.warning("Please select at least one predictor for the count model.")
        return

    if not inflation_predictors:
        st.warning("Please select at least one predictor for the zero-inflation model.")
        return

    selected_predictors = sorted(set(count_predictors + inflation_predictors))

    categorical_vars = st.multiselect(
        "Select Categorical Variables (Factors)",
        selected_predictors
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

    def build_terms(var_list):
        terms_local = []

        for var in var_list:
            if var in categorical_vars:
                ref = reference_dict[var]
                terms_local.append(f'C({var}, Treatment(reference="{ref}"))')
            else:
                terms_local.append(var)

        return terms_local

    count_terms = build_terms(count_predictors)
    infl_terms = build_terms(inflation_predictors)

    count_formula_rhs = " + ".join(count_terms) if count_terms else "1"
    infl_formula_rhs = " + ".join(infl_terms) if infl_terms else "1"

    count_formula = response_original + " ~ " + count_formula_rhs

    st.subheader("Count Model Formula")
    st.code(count_formula)

    st.subheader("Zero-Inflation Model Formula")
    st.code(f"logit(π) ~ {infl_formula_rhs}")

    st.info(
        "This model has two parts: a **Poisson count model** and a **logistic zero-inflation model** "
        "for structural zeros."
    )

    # ======================================================
    # 3️⃣ MODEL FITTING
    # ======================================================

    st.header("2️⃣ Model Fitting")

    df_model = df[[response_original] + selected_predictors].copy()

    for var in selected_predictors:
        if var not in categorical_vars:
            df_model[var] = pd.to_numeric(df_model[var], errors="coerce")

    df_model = df_model.dropna()

    if df_model.empty:
        st.error("No valid rows remain after removing missing values.")
        return

    try:
        y, X_count = patsy.dmatrices(
            count_formula,
            df_model,
            return_type="dataframe"
        )

        X_infl = patsy.dmatrix(
            infl_formula_rhs,
            df_model,
            return_type="dataframe"
        )

        y_series = y.iloc[:, 0]

        model = ZeroInflatedPoisson(
            endog=y_series,
            exog=X_count,
            exog_infl=X_infl,
            inflation="logit"
        )

        res = model.fit(method="bfgs", maxiter=200, disp=False)

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
        y_null, X_count_null = patsy.dmatrices(
            response_original + " ~ 1",
            df_model,
            return_type="dataframe"
        )

        X_infl_null = patsy.dmatrix(
            "1",
            df_model,
            return_type="dataframe"
        )

        null_model = ZeroInflatedPoisson(
            endog=y_null.iloc[:, 0],
            exog=X_count_null,
            exog_infl=X_infl_null,
            inflation="logit"
        )

        res_null = null_model.fit(method="bfgs", maxiter=200, disp=False)

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

    observed_zero_rate = (df_model[response_original] == 0).mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Log-Likelihood", round(loglik, 2))
    col2.metric("AIC", round(aic, 2))
    col3.metric("AICc", round(aicc, 2) if pd.notna(aicc) else "N/A")
    col4.metric("BIC", round(bic, 2))
    col5.metric("Observed Zero Rate", round(observed_zero_rate, 3))

    # ======================================================
    # 6️⃣ FITTED REGRESSION MODEL
    # ======================================================

    st.header("4️⃣ Fitted Regression Model")

    def split_params(result_obj):
        inflate_params = result_obj.params[result_obj.params.index.str.startswith("inflate_")]
        count_params = result_obj.params[~result_obj.params.index.str.startswith("inflate_")]
        return inflate_params, count_params

    def clean_name(name, prefix=""):
        return name.replace(prefix, "", 1) if name.startswith(prefix) else name

    def build_linear_part(params, prefix=""):
        intercept_name = None

        for name in params.index:
            if name in [f"{prefix}Intercept", f"{prefix}const", "Intercept", "const"]:
                intercept_name = name
                break

        intercept_val = round(params[intercept_name], 4) if intercept_name else 0
        pieces = [f"{intercept_val}"]

        for name in params.index:
            if name == intercept_name:
                continue

            coef = round(params[name], 4)
            display = clean_name(name, prefix)

            if display.startswith("C(") and "T." in display:
                var = display.split("[")[0].replace("C(", "").split(",")[0]
                level = display.split("T.")[-1].replace("]", "")
                term = f"D_{{{var}={level}}}"
            else:
                term = display

            sign = "+" if coef >= 0 else "-"
            pieces.append(f" {sign} {abs(coef)}\\cdot {term}")

        return "".join(pieces)

    inflate_params, count_params = split_params(res)

    pi_eq = build_linear_part(inflate_params, prefix="inflate_")
    lambda_eq = build_linear_part(count_params)

    st.markdown("**From this output, the fitted regression model has estimated parameters:**")

    st.latex(
        f"\\widehat{{\\pi}}=\\frac{{\\exp\\left\\{{{pi_eq}\\right\\}}}}{{1+\\exp\\left\\{{{pi_eq}\\right\\}}}},"
    )

    st.markdown("and")

    st.latex(
        f"\\widehat{{\\lambda}}=\\exp\\left\\{{{lambda_eq}\\right\\}}."
    )

    # ======================================================
    # 7️⃣ INTERPRETATION
    # ======================================================

    st.header("5️⃣ Interpretation")

    # ---------------- COUNT COMPONENT ----------------
    st.subheader("Count Component")

    st.markdown("Interpretation uses $e^{\\beta}$.")

    for term in count_params.index:

        coef = count_params[term]
        pval = res.pvalues[term]
        exp_coef = np.exp(coef)

        label = term

        st.subheader(label)
        st.latex(f"e^{{{coef:.4f}}} = {exp_coef:.4f}")

        if term in ["Intercept", "const"]:
            st.write(
                f"When all predictors are held at zero and all indicator variables are at their reference levels, "
                f"the estimated count rate is {exp_coef:.4f}."
            )

        elif term.startswith("C("):

            var = term.split("[")[0].replace("C(", "").split(",")[0]
            level = term.split("T.")[-1].replace("]", "")
            ref = reference_dict.get(var, "reference")

            st.write(
                f"If {var} is an indicator variable, then $e^{{\\hat{{\\beta}}}} = {exp_coef:.4f}$ "
                f"represents the ratio of the estimated rates for {var} = {level} and {var} = {ref}."
            )

            st.write(
                f"Equivalently, $e^{{\\hat{{\\beta}}}}\\cdot 100\\% = {exp_coef*100:.2f}\\%$ "
                f"represents the estimated percent ratio of rates."
            )

        else:

            pct = (exp_coef - 1) * 100

            st.write(
                f"If {label} is numeric, then $e^{{\\hat{{\\beta}}}} = {exp_coef:.4f}$ represents "
                f"the estimated rate ratio for a one-unit increase in {label}."
            )

            st.write(
                f"Equivalently, $(e^{{\\hat{{\\beta}}}} - 1)\\cdot 100\\% = {pct:.2f}\\% "
                f"is the estimated percent change in rate."
            )

        st.write(f"Coefficient = {coef:.4f}")
        st.write(f"p-value = {pval:.4f}")

        if pval <= 0.05:
            st.success("Significant")
        else:
            st.warning("Not significant")

    # ---------------- ZERO-INFLATION COMPONENT ----------------
    st.subheader("Zero-Inflation Component")

    st.markdown("Interpretation uses $e^{\\beta}$.")

    for term in inflate_params.index:

        coef = inflate_params[term]
        pval = res.pvalues[term]
        exp_coef = np.exp(coef)

        label = term.replace("inflate_", "")

        st.subheader(label)
        st.latex(f"e^{{{coef:.4f}}} = {exp_coef:.4f}")

        if label in ["Intercept", "const"]:
            st.write(
                f"When predictors are held at zero and reference levels, "
                f"the baseline odds multiplier for structural zeros is {exp_coef:.4f}."
            )

        elif label.startswith("C("):

            var = label.split("[")[0].replace("C(", "").split(",")[0]
            level = label.split("T.")[-1].replace("]", "")
            ref = reference_dict.get(var, "reference")

            st.write(
                f"If {var} is an indicator variable, then $e^{{\\hat{{\\beta}}}} = {exp_coef:.4f}$ "
                f"represents the ratio of the odds of being a structural zero for "
                f"{var} = {level} and {var} = {ref}."
            )

            st.write(
                f"Equivalently, $e^{{\\hat{{\\beta}}}}\\cdot 100\\% = {exp_coef*100:.2f}\\% "
                f"represents the estimated percent ratio of odds."
            )

        else:

            pct = (exp_coef - 1) * 100

            st.write(
                f"If {label} is numeric, then $e^{{\\hat{{\\beta}}}} = {exp_coef:.4f}$ represents "
                f"the odds ratio for a one-unit increase in {label}."
            )

            st.write(
                f"Equivalently, $(e^{{\\hat{{\\beta}}}} - 1)\\cdot 100\\% = {pct:.2f}\\% "
                f"is the estimated percent change in odds."
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

    for var in selected_predictors:

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

        for var in selected_predictors:
            if var not in categorical_vars:
                new_df[var] = pd.to_numeric(new_df[var], errors="coerce")

        try:
            X_count_new = patsy.build_design_matrices(
                [X_count.design_info],
                new_df,
                return_type="dataframe"
            )[0]

            X_infl_new = patsy.build_design_matrices(
                [X_infl.design_info],
                new_df,
                return_type="dataframe"
            )[0]

            overall_mean = res.predict(
                exog=X_count_new,
                exog_infl=X_infl_new,
                which="mean"
            )[0]

            gamma = inflate_params.values
            pi_hat = expit(np.dot(X_infl_new.iloc[0].values, gamma))

            mu_hat = np.exp(np.dot(X_count_new.iloc[0].values, count_params.values))
            prob_zero = pi_hat + (1 - pi_hat) * np.exp(-mu_hat)

            st.subheader("Prediction Results")
            st.write(f"Predicted structural-zero probability: **{pi_hat:.4f}**")
            st.write(f"Predicted overall probability of zero count: **{prob_zero:.4f}**")
            st.success(f"Predicted expected count for {response_original}: {overall_mean:.4f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ======================================================
    # 9️⃣ PREDICTED VS ACTUAL
    # ======================================================

    st.header("6️⃣ Predicted vs Actual")

    try:
        predicted_mean = res.predict(
            exog=X_count,
            exog_infl=X_infl,
            which="mean"
        )

        plot_df = pd.DataFrame({
            "Predicted": predicted_mean,
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
