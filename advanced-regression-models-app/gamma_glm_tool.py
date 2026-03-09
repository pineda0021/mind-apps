import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, shapiro


def run():

    st.title("Gamma Generalized Linear Model (Log Link)")

    # ======================================================
    # 1. DATA UPLOAD
    # ======================================================

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="gamma_upload"
    )

    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ======================================================
    # 2. VARIABLE SELECTION
    # ======================================================

    st.header("1️⃣ Select Variables")

    response = st.selectbox("Select Response Variable (Y)", df.columns)

    if not pd.api.types.is_numeric_dtype(df[response]):
        st.error("Response must be numeric.")
        return

    if (df[response] <= 0).any():
        st.error("Gamma regression requires strictly positive response values.")
        return

    # ======================================================
    # ✅ Normality Check (Y vs log(Y))
    # ======================================================

    st.subheader("Distribution Check")

    y_original = df[response].dropna()

    if len(y_original) >= 3:
        stat_orig, p_orig = shapiro(y_original)
        st.write(f"Shapiro-Wilk p-value (Original Y): {p_orig:.4f}")
    else:
        st.warning("Not enough data for Shapiro-Wilk test (original Y).")

    y_log = np.log(y_original)

    if len(y_log) >= 3:
        stat_log, p_log = shapiro(y_log)
        st.write(f"Shapiro-Wilk p-value (Log(Y)): {p_log:.4f}")
    else:
        st.warning("Not enough data for Shapiro-Wilk test (log Y).")

    # Side-by-side histograms
    col1, col2 = st.columns(2)

    with col1:
        fig_y = px.histogram(
            df,
            x=response,
            title="Original Y Distribution"
        )
        st.plotly_chart(fig_y)

    with col2:
        fig_log = px.histogram(
            x=y_log,
            title="Log(Y) Distribution"
        )
        st.plotly_chart(fig_log)

    predictors = st.multiselect(
        "Select Predictor Variables (X)",
        [col for col in df.columns if col != response]
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
            key=f"ref_gamma_{col}"
        )
        reference_dict[col] = ref

    # ======================================================
    # 3. BUILD FORMULA
    # ======================================================

    terms = []

    for var in predictors:
        if var in categorical_vars:
            ref = reference_dict[var]
            terms.append(f'C({var}, Treatment(reference="{ref}"))')
        else:
            terms.append(var)

    formula = response + " ~ " + " + ".join(terms)

    # ======================================================
    # 4. FIT GAMMA GLM
    # ======================================================

    st.header("2️⃣ Fit Gamma GLM (Log Link)")

    model_gamma = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Gamma(link=sm.families.links.log())
    ).fit()

    st.subheader("Model Summary")
    st.text(model_gamma.summary())

    # ======================================================
    # 5. MODEL FIT STATISTICS
    # ======================================================

    st.header("3️⃣ Model Fit Evaluation")

    n = df.shape[0]
    k = int(model_gamma.df_model) + 1

    loglik = model_gamma.llf
    aic = model_gamma.aic
    bic = model_gamma.bic
    deviance = model_gamma.deviance

    if (n - k - 1) > 0:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = float("nan")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Log-Likelihood", round(loglik, 2))
    col2.metric("AIC", round(aic, 2))
    col3.metric("AICc", round(aicc, 2))
    col4.metric("BIC", round(bic, 2))

    st.write(f"Model Deviance: {deviance:.4f}")

    st.markdown("""
Lower AIC/AICc/BIC indicate better balance between model fit and complexity.

Gamma models are appropriate for positively skewed continuous outcomes.
""")

    # ======================================================
    # 6. Likelihood Ratio Test
    # ======================================================

    st.subheader("Likelihood Ratio (Deviance) Test")

    null_formula = response + " ~ 1"

    null_model = smf.glm(
        formula=null_formula,
        data=df,
        family=sm.families.Gamma(link=sm.families.links.log())
    ).fit()

    lr_stat = -2 * (null_model.llf - model_gamma.llf)
    df_diff = int(model_gamma.df_model)
    p_value_lr = chi2.sf(lr_stat, df_diff)

    st.write(f"LR Statistic: {lr_stat:.4f}")
    st.write(f"Degrees of Freedom: {df_diff}")
    st.write(f"p-value: {p_value_lr:.6f}")

    if p_value_lr < 0.05:
        st.success("The Gamma model significantly improves over the intercept-only model.")
    else:
        st.warning("The Gamma model does not significantly improve over the intercept-only model.")

    # ======================================================
    # 7. INTERPRETATION
    # ======================================================

    st.header("4️⃣ Interpretation of Coefficients")

    for name, coef in model_gamma.params.items():

        if name == "Intercept":
            continue

        coef = round(coef, 4)
        multiplicative_effect = round(np.exp(coef), 4)

        if "C(" in name:
            var_name = name.split("[")[0]
            var_name = var_name.replace("C(", "").split(",")[0]
            level = name.split("T.")[1].replace("]", "")
            ref = reference_dict[var_name]

            st.write(
                f"For **{var_name} = {level}**, the expected **{response}** "
                f"is multiplied by **{multiplicative_effect}** "
                f"relative to the reference group (**{ref}**)."
            )
        else:
            st.write(
                f"For each one-unit increase in **{name}**, the expected "
                f"**{response}** is multiplied by **{multiplicative_effect}**, "
                "holding other variables constant."
            )

    # ======================================================
    # 8. PREDICTION
    # ======================================================

    st.header("5️⃣ Prediction")

    input_dict = {}

    for var in predictors:

        if not pd.api.types.is_numeric_dtype(df[var]):

            if not pd.api.types.is_categorical_dtype(df[var]):
                df[var] = df[var].astype("category")

            input_dict[var] = st.selectbox(
                var,
                df[var].cat.categories
            )

        else:
            input_dict[var] = st.number_input(
                var,
                value=float(df[var].mean())
            )

    if st.button("Predict (Gamma)"):

        new_df = pd.DataFrame([input_dict])
        prediction = model_gamma.predict(new_df)[0]

        st.success(f"Predicted {response}: {prediction:.4f}")
