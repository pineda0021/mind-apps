import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import patsy
import statsmodels.formula.api as smf
from statsmodels.discrete.truncated_model import TruncatedLFPoisson
from scipy.stats import chi2


def run():

    st.title("📘 Hurdle Model (Logistic + Zero-truncated Poisson)")

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
        st.error("The response variable must be nonnegative.")
        return

    all_predictors = [c for c in df.columns if c != response_original]

    count_predictors = st.multiselect(
        "Select Positive-Count Model Predictor Variables (X)",
        all_predictors
    )

    hurdle_predictors = st.multiselect(
        "Select Hurdle Model Predictor Variables (Z)",
        all_predictors
    )

    if not count_predictors:
        st.warning("Please select at least one predictor for the positive-count model.")
        return

    if not hurdle_predictors:
        st.warning("Please select at least one predictor for the hurdle model.")
        return

    selected_predictors = sorted(set(count_predictors + hurdle_predictors))

    categorical_vars = st.multiselect(
        "Select Categorical Variables (Factors)",
        selected_predictors
    )

    reference_dict = {}

    for col in categorical_vars:
        df[col] = df[col].astype(str).astype("category")

        ref = st.selectbox(
            f"Select reference level for {col}",
            list(df[col].cat.categories),
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
    hurdle_terms = build_terms(hurdle_predictors)

    count_formula_rhs = " + ".join(count_terms) if count_terms else "1"
    hurdle_formula_rhs = " + ".join(hurdle_terms) if hurdle_terms else "1"

    count_formula = f"{response_original} ~ {count_formula_rhs}"
    hurdle_response = "bought_any"
    hurdle_formula = f"{hurdle_response} ~ {hurdle_formula_rhs}"

    st.subheader("Positive-Count Model Formula")
    st.code(count_formula)

    st.subheader("Hurdle Model Formula")
    st.code(hurdle_formula)

    st.info(
        "This model has two parts: a logistic model for whether the count is positive, "
        "and a zero-truncated Poisson model for the positive counts."
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

    df_model["bought_any"] = (df_model[response_original] > 0).astype(int)

    nonzero_data = df_model[df_model[response_original] > 0].copy()

    if nonzero_data.empty:
        st.error("No positive counts are available for the zero-truncated Poisson model.")
        return

    try:
        y_count, X_count = patsy.dmatrices(
            count_formula,
            data=nonzero_data,
            return_type="dataframe"
        )

        count_model = TruncatedLFPoisson(
            y_count,
            X_count,
            truncation=0
        )
        count_res = count_model.fit(disp=False)

        hurdle_res = smf.logit(
            hurdle_formula,
            data=df_model
        ).fit(disp=False)

    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        return

    st.subheader("Zero-truncated Poisson Summary")
    st.text(count_res.summary())

    st.subheader("Hurdle Logistic Summary")
    st.text(hurdle_res.summary())

    # ======================================================
    # 4️⃣ MODEL FIT EVALUATION
    # ======================================================

    st.header("3️⃣ Model Fit Evaluation")

    try:
        count_loglik = count_res.llf
        count_aic = count_res.aic
        count_bic = count_res.bic
    except Exception:
        count_loglik = np.nan
        count_aic = np.nan
        count_bic = np.nan

    try:
        hurdle_loglik = hurdle_res.llf
        hurdle_aic = hurdle_res.aic
        hurdle_bic = hurdle_res.bic
    except Exception:
        hurdle_loglik = np.nan
        hurdle_aic = np.nan
        hurdle_bic = np.nan

    st.subheader("Positive-Count Model")
    c1, c2, c3 = st.columns(3)
    c1.metric("Log-Likelihood", round(count_loglik, 2) if pd.notna(count_loglik) else "N/A")
    c2.metric("AIC", round(count_aic, 2) if pd.notna(count_aic) else "N/A")
    c3.metric("BIC", round(count_bic, 2) if pd.notna(count_bic) else "N/A")

    st.subheader("Hurdle Model")
    h1, h2, h3 = st.columns(3)
    h1.metric("Log-Likelihood", round(hurdle_loglik, 2) if pd.notna(hurdle_loglik) else "N/A")
    h2.metric("AIC", round(hurdle_aic, 2) if pd.notna(hurdle_aic) else "N/A")
    h3.metric("BIC", round(hurdle_bic, 2) if pd.notna(hurdle_bic) else "N/A")

    # ======================================================
    # 5️⃣ FITTED REGRESSION MODEL
    # ======================================================

    st.header("4️⃣ Fitted Regression Model")

    def build_linear_part(params):
        intercept_name = None

        for name in params.index:
            if name in ["Intercept", "const"]:
                intercept_name = name
                break

        intercept_val = round(params[intercept_name], 4) if intercept_name else 0
        pieces = [f"{intercept_val}"]

        for name in params.index:
            if name == intercept_name:
                continue

            coef = round(params[name], 4)

            if name.startswith("C(") and "T." in name:
                var = name.split("[")[0].replace("C(", "").split(",")[0]
                level = name.split("T.")[-1].replace("]", "")
                term = f"D_{{{var}={level}}}"
            else:
                term = name

            sign = "+" if coef >= 0 else "-"
            pieces.append(f" {sign} {abs(coef)}\\cdot {term}")

        return "".join(pieces)

    lambda_eq = build_linear_part(count_res.params)

    # Reverse the logistic linear predictor for pi_hat = P(Y = 0)
    reversed_hurdle_params = -hurdle_res.params
    pi_eq = build_linear_part(reversed_hurdle_params)

    st.markdown("**From this output, the fitted regression model has estimated parameters:**")

    st.latex(
        f"\\widehat{{\\pi}}=\\frac{{\\exp\\left\\{{{pi_eq}\\right\\}}}}{{1+\\exp\\left\\{{{pi_eq}\\right\\}}}},"
    )

    st.markdown("and")

    st.latex(
        f"\\widehat{{\\lambda}}=\\exp\\left\\{{{lambda_eq}\\right\\}}."
    )

    # ======================================================
    # 6️⃣ INTERPRETATION
    # ======================================================

    st.header("5️⃣ Interpretation")

    st.subheader("Positive-Count Component")

    st.markdown("Interpretation uses $e^{\\beta}$.")

    for term in count_res.params.index:

        coef = count_res.params[term]
        pval = count_res.pvalues[term]
        exp_coef = np.exp(coef)

        label = term

        st.subheader(label)
        st.latex(f"e^{{{coef:.4f}}} = {exp_coef:.4f}")

        if term in ["Intercept", "const"]:
            st.write(
                f"When all predictors are held at zero and all indicator variables are at their reference levels, "
                f"the estimated positive-count rate is {exp_coef:.4f}."
            )

        elif term.startswith("C("):
            var = term.split("[")[0].replace("C(", "").split(",")[0]
            level = term.split("T.")[-1].replace("]", "")
            ref = reference_dict.get(var, "reference")

            st.write(
                f"If {var} is an indicator variable, then $e^{{\\hat{{\\beta}}}} = {exp_coef:.4f}$ "
                f"represents the ratio of the estimated positive-count rates for {var} = {level} and {var} = {ref}."
            )

            st.write(
                f"Equivalently, $e^{{\\hat{{\\beta}}}}\\cdot 100\\% = {exp_coef * 100:.2f}\\%$ "
                f"represents the estimated percent ratio of positive-count rates."
            )

        else:
            pct = (exp_coef - 1) * 100

            st.write(
                f"If {label} is numeric, then $e^{{\\hat{{\\beta}}}} = {exp_coef:.4f}$ represents "
                f"the estimated rate ratio for a one-unit increase in {label}."
            )

            st.write(
                f"Equivalently, $(e^{{\\hat{{\\beta}}}} - 1)\\cdot 100\\% = {pct:.2f}\\%$"
                f" is the estimated percent change in positive-count rate."
            )

        st.write(f"Coefficient = {coef:.4f}")
        st.write(f"p-value = {pval:.4f}")

        if pval <= 0.05:
            st.success("Significant")
        else:
            st.warning("Not significant")

    # ---------------- HURDLE COMPONENT (π = P(Y = 0)) ----------------

    st.subheader("Hurdle Component")

    st.markdown("Interpretation uses $e^{\\beta}$.")

    for term in hurdle_res.params.index:

        # 🔥 Reverse coefficient (since π = P(Y=0))
        coef = -hurdle_res.params[term]
        pval = hurdle_res.pvalues[term]
        exp_coef = np.exp(coef)

        label = term

        st.subheader(label)
        st.latex(f"e^{{{coef:.4f}}} = {exp_coef:.4f}")

        # ---------------- INTERCEPT ----------------
        if term in ["Intercept", "const"]:

            st.write(
                f"When predictors are held at zero and reference levels, "
                f"the baseline odds of not purchasing textbooks are multiplied by {exp_coef:.4f}."
            )

        # ---------------- CATEGORICAL ----------------
        elif term.startswith("C("):

            var = term.split("[")[0].replace("C(", "").split(",")[0]
            level = term.split("T.")[-1].replace("]", "")
            ref = reference_dict.get(var, "reference")

            percent_change = abs((exp_coef - 1) * 100)

            st.write(
                f"If {var} is an indicator variable, then $e^{{\\hat{{\\beta}}}} = {exp_coef:.4f}$ "
                f"represents the ratio of the odds of not purchasing textbooks for "
                f"{var} = {level} and {var} = {ref}."
            )

            if coef >= 0:
                st.write(
                    f"Equivalently, the estimated odds increase by "
                    f"$(e^{{\\hat{{\\beta}}}} - 1)\\cdot 100\\% = {percent_change:.2f}\\%.$"
                )
            else:
                st.write(
                    f"Equivalently, the estimated odds decrease by "
                    f"$\\left(1 - e^{{\\hat{{\\beta}}}}\\right)\\cdot 100\\% = {percent_change:.2f}\\%.$"
                )

        # ---------------- NUMERIC ----------------
        else:

            percent_change = abs((exp_coef - 1) * 100)

            st.write(
                f"If {label} is numeric, then $e^{{\\hat{{\\beta}}}} = {exp_coef:.4f}$ represents "
                f"the odds ratio for a one-unit increase in {label}."
            )

            if coef >= 0:
                st.write(
                    f"Equivalently, the estimated odds increase by "
                    f"$(e^{{\\hat{{\\beta}}}} - 1)\\cdot 100\\% = {percent_change:.2f}\\%.$"
                )
            else:
                st.write(
                    f"Equivalently, the estimated odds decrease by "
                    f"$\\left(1 - e^{{\\hat{{\\beta}}}}\\right)\\cdot 100\\% = {percent_change:.2f}\\%.$"
                )

        st.write(f"Coefficient = {coef:.4f}")
        st.write(f"p-value = {pval:.4f}")

        if pval <= 0.05:
            st.success("Significant")
        else:
            st.warning("Not significant")

    # ======================================================
    # 7️⃣ PREDICTION
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
            prob_buy = hurdle_res.predict(new_df).iloc[0]
            prob_not_buy = 1 - prob_buy

            new_X_count = patsy.build_design_matrices(
                [X_count.design_info],
                new_df,
                return_type="dataframe"
            )[0]

            truncated_mean = count_res.predict(new_X_count).iloc[0]
            expected_count = prob_buy * truncated_mean

            st.subheader("Prediction Results")
            st.write(f"Predicted probability of not buying any item: **{prob_not_buy:.4f}**")
            st.write(f"Predicted probability of buying at least 1 item: **{prob_buy:.4f}**")
            st.write(f"Predicted expected count if positive: **{truncated_mean:.4f}**")
            st.success(f"Final expected count for {response_original}: {expected_count:.4f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ======================================================
    # 8️⃣ PREDICTED VS ACTUAL
    # ======================================================

    st.header("7️⃣ Predicted vs Actual")

    try:
        prob_buy_all = hurdle_res.predict(df_model)

        X_count_all = patsy.build_design_matrices(
            [X_count.design_info],
            df_model,
            return_type="dataframe"
        )[0]

        truncated_mean_all = count_res.predict(X_count_all)
        overall_pred = prob_buy_all.values * truncated_mean_all.values

        plot_df = pd.DataFrame({
            "Predicted": overall_pred,
            "Actual": df_model[response_original].values
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
