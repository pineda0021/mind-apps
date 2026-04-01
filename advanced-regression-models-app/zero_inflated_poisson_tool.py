import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import patsy
from scipy.stats import chi2
from scipy.special import expit
from statsmodels.discrete.count_model import ZeroInflatedPoisson
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree


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
    infl_terms = build_terms(inflation_predictors)

    count_formula_rhs = " + ".join(count_terms) if count_terms else "1"
    infl_formula_rhs = " + ".join(infl_terms) if infl_terms else "1"

    count_formula = response_original + " ~ " + count_formula_rhs

    st.subheader("Count Model Formula")
    st.code(count_formula)

    st.subheader("Zero-Inflation Model Formula")
    st.code(f"logit(π) ~ {infl_formula_rhs}")

    st.info(
        "This model has two components: a Poisson count model for expected counts and "
        "a logistic model for the probability of structural zeros."
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

    def format_term(display_name, coef):
        coef_abs = abs(round(coef, 4))
        sign = "+" if coef >= 0 else "-"

        if display_name.startswith("C(") and "T." in display_name:
            var_name = display_name.split("[")[0].replace("C(", "").split(",")[0]
            level = display_name.split("T.")[-1].replace("]", "")
            term_label = f"D_{{{var_name}={level}}}"
        else:
            term_label = display_name

        return f" {sign} {coef_abs}\\cdot {term_label}"

    def build_pi_equation(params):
        intercept_name = None
        for name in params.index:
            if name in ["inflate_Intercept", "inflate_const", "Intercept", "const"]:
                intercept_name = name
                break

        pieces = []

        intercept_val = round(params[intercept_name], 4) if intercept_name is not None else 0
        pieces.append(f"{intercept_val}")

        for name in params.index:
            if name == intercept_name:
                continue

            display_name = name.replace("inflate_", "", 1) if name.startswith("inflate_") else name
            pieces.append(format_term(display_name, params[name]))

        inside = "".join(pieces)
        return (
            f"\\widehat{{\\pi}}="
            f"\\frac{{\\exp\\left\\{{{inside}\\right\\}}}}"
            f"{{1+\\exp\\left\\{{{inside}\\right\\}}}}"
        )

    def build_lambda_equation(params):
        intercept_name = "Intercept" if "Intercept" in params.index else "const"

        pieces = []
        intercept_val = round(params[intercept_name], 4) if intercept_name in params.index else 0
        pieces.append(f"{intercept_val}")

        for name in params.index:
            if name == intercept_name:
                continue
            pieces.append(format_term(name, params[name]))

        inside = "".join(pieces)
        return f"\\widehat{{\\lambda}}=\\exp\\left\\{{{inside}\\right\\}}"

    inflate_params, count_params = split_params(res)

    st.markdown("**From this output, the fitted regression model has estimated parameters:**")
    st.latex(build_pi_equation(inflate_params))
    st.markdown("and")
    st.latex(build_lambda_equation(count_params))

    # ======================================================
    # 7️⃣ INTERPRETATION
    # ======================================================

    st.header("5️⃣ Interpretation")

    st.subheader("Count Component")

    st.markdown("Interpretation uses $e^{\\beta}$.")

    for term in count_params.index:

        coef = count_params[term]
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

        if term in ["Intercept", "const"]:

            st.write(
                f"When all predictors are held at zero and all indicator variables are at their reference levels, "
                f"the estimated count rate is {exp_coef:.4f}."
            )

        elif term.startswith("C("):

            var_name = term.split("[")[0].replace("C(", "").split(",")[0]
            level = term.split("T.")[-1].replace("]", "")
            ref = reference_dict.get(var_name, "reference")

            st.write(
                f"If {var_name} is an indicator variable, then $e^{{\\hat{{\\beta}}}} = {exp_coef:.4f}$ "
                f"represents the ratio of the estimated rates for {var_name} = {level} and {var_name} = {ref}."
            )

            st.write(
                f"Equivalently, $e^{{\\hat{{\\beta}}}}\\cdot 100\\% = {exp_coef * 100:.2f}\\%$ "
                f"represents the estimated percent ratio of rates."
            )

        else:

            percent_change = (exp_coef - 1) * 100

            st.write(
                f"If {label} is numeric, then $e^{{\\hat{{\\beta}}}} = {exp_coef:.4f}$ represents "
                f"the estimated rate ratio for a one-unit increase in {label}."
            )

            st.write(
                f"Equivalently, $(e^{{\\hat{{\\beta}}}} - 1)\\cdot 100\\% = {percent_change:.2f}\\%$ "
                f"is the estimated percent change in rate."
            )

        st.write(f"Coefficient = {coef:.4f}")
        st.write(f"p-value = {pval:.4f}")

        if pval <= 0.05:
            st.success("Significant")
        else:
            st.warning("Not significant")

    st.subheader("Zero-Inflation Component")

    st.markdown("Interpretation uses $e^{\\beta}$.")

    for term in inflate_params.index:

        coef = inflate_params[term]
        pval = res.pvalues[term]
        exp_coef = np.exp(coef)

        display_term = term.replace("inflate_", "", 1)

        if display_term.startswith("C("):
            var_name = display_term.split("[")[0].replace("C(", "").split(",")[0]
            level = display_term.split("T.")[-1].replace("]", "")
            label = f"{var_name}[{level}]"
        else:
            label = display_term

        st.subheader(label)
        st.latex(f"e^{{{coef:.4f}}} = {exp_coef:.4f}")

        if display_term in ["Intercept", "const"]:

            st.write(
                f"When all predictors are held at zero and all indicator variables are at their reference levels, "
                f"the odds ratio baseline for structural zero membership is {exp_coef:.4f}."
            )

        elif display_term.startswith("C("):

            var_name = display_term.split("[")[0].replace("C(", "").split(",")[0]
            level = display_term.split("T.")[-1].replace("]", "")
            ref = reference_dict.get(var_name, "reference")

            st.write(
                f"If {var_name} is an indicator variable, then $e^{{\\hat{{\\beta}}}} = {exp_coef:.4f}$ "
                f"represents the ratio of the odds of being a structural zero for "
                f"{var_name} = {level} and {var_name} = {ref}."
            )

            st.write(
                f"Equivalently, $e^{{\\hat{{\\beta}}}}\\cdot 100\\% = {exp_coef * 100:.2f}\\%$ "
                f"represents the estimated percent ratio of odds of being a structural zero."
            )

        else:

            percent_change = (exp_coef - 1) * 100

            st.write(
                f"If {label} is numeric, then $e^{{\\hat{{\\beta}}}} = {exp_coef:.4f}$ represents "
                f"the odds ratio for a one-unit increase in {label} for structural zero membership."
            )

            st.write(
                f"Equivalently, $(e^{{\\hat{{\\beta}}}} - 1)\\cdot 100\\% = {percent_change:.2f}\\%$ "
                f"is the estimated percent change in the odds of being a structural zero."
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

    st.header("7️⃣ Predicted vs Actual")

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

    # ======================================================
    # 🔟 STRUCTURAL ZERO PROBABILITIES
    # ======================================================

    st.header("8️⃣ Structural Zero Probabilities")

    try:
        df_probs = df_model.copy()

        gamma = inflate_params.values
        df_probs["prob_structural_zero"] = expit(np.dot(X_infl.values, gamma))

        display_cols = [col for col in selected_predictors if col in df_probs.columns]
        display_cols = display_cols + [response_original, "prob_structural_zero"]

        st.dataframe(
            df_probs[display_cols].head(10),
            use_container_width=True
        )

        fig_prob = px.scatter(
            df_probs,
            x=response_original,
            y="prob_structural_zero",
            title="Structural-Zero Probability vs Observed Count",
            labels={
                response_original: "Observed Count",
                "prob_structural_zero": "Predicted Structural-Zero Probability"
            }
        )

        st.plotly_chart(fig_prob, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not compute structural-zero probabilities: {e}")

    # ======================================================
    # 1️⃣1️⃣ OPTIONAL DECISION TREES
    # ======================================================

    st.header("9️⃣ Optional Tree Views")

    try:
        tree_df = df_model.copy()
        tree_df["structural_zero"] = (tree_df[response_original] == 0).astype(int)

        dummy_df = pd.get_dummies(
            tree_df[[response_original, "structural_zero"] + selected_predictors],
            columns=[v for v in selected_predictors if v in categorical_vars],
            drop_first=True
        )

        zero_tree_features = [
            c for c in dummy_df.columns
            if c not in [response_original, "structural_zero"]
            and any(c == v or c.startswith(f"{v}_") for v in inflation_predictors)
        ]

        count_tree_features = [
            c for c in dummy_df.columns
            if c not in [response_original, "structural_zero"]
            and any(c == v or c.startswith(f"{v}_") for v in count_predictors)
        ]

        tree_zero = None
        tree_count = None

        if zero_tree_features:
            X_zero = dummy_df[zero_tree_features]
            y_zero = dummy_df["structural_zero"]

            tree_zero = DecisionTreeClassifier(
                ccp_alpha=0.01,
                random_state=123
            )
            tree_zero.fit(X_zero, y_zero)

        smokers_df = dummy_df[dummy_df[response_original] > 0].copy()

        if count_tree_features and not smokers_df.empty:
            X_count_tree = smokers_df[count_tree_features]
            y_count_tree = smokers_df[response_original]

            tree_count = DecisionTreeRegressor(
                ccp_alpha=0.05,
                min_samples_split=20,
                min_samples_leaf=7,
                random_state=123
            )
            tree_count.fit(X_count_tree, y_count_tree)

        if tree_zero is None and tree_count is None:
            st.info("Not enough valid predictors were available to build the optional trees.")
        else:
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))

            if tree_zero is not None:
                plot_tree(
                    tree_zero,
                    feature_names=list(X_zero.columns),
                    filled=True,
                    ax=ax[0],
                    proportion=True,
                    impurity=False
                )
                ax[0].set_title("Zero Model Tree (Structural Zeros)")
            else:
                ax[0].axis("off")
                ax[0].set_title("Zero Model Tree Not Available")

            if tree_count is not None:
                plot_tree(
                    tree_count,
                    feature_names=list(X_count_tree.columns),
                    filled=True,
                    ax=ax[1],
                    proportion=True,
                    impurity=False
                )
                ax[1].set_title("Count Model Tree (Positive Counts)")
            else:
                ax[1].axis("off")
                ax[1].set_title("Count Model Tree Not Available")

            st.pyplot(fig)
            plt.close(fig)

    except Exception as e:
        st.warning(f"Could not create the optional tree views: {e}")


if __name__ == "__main__":
    run()
