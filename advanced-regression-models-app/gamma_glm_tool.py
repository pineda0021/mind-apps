import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, shapiro
import plotly.graph_objects as go


def run():

    st.title("Gamma Generalized Linear Model (Log Link)")

    # ======================================================
    # 1. DATA UPLOAD
    # ======================================================

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)

    initial_rows = df.shape[0]
    df = df.dropna()
    dropped = initial_rows - df.shape[0]

    if dropped > 0:
        st.warning(f"{dropped} rows removed due to missing values.")

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
    # Distribution Check
    # ======================================================

    st.subheader("Distribution Check")

    y_original = df[response]

    if len(y_original) >= 3:
        _, p_orig = shapiro(y_original)
        st.write(f"Shapiro-Wilk p-value (Original Y): {p_orig:.4f}")

    y_log = np.log(y_original)

    if len(y_log) >= 3:
        _, p_log = shapiro(y_log)
        st.write(f"Shapiro-Wilk p-value (Log(Y)): {p_log:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(px.histogram(df, x=response, title="Original Y"))

    with col2:
        st.plotly_chart(px.histogram(x=y_log, title="Log(Y)"))

    predictors = st.multiselect(
        "Select Predictor Variables (X)",
        [col for col in df.columns if col != response]
    )

    if not predictors:
        return

    categorical_vars = st.multiselect(
        "Select Categorical Variables",
        predictors
    )

    reference_dict = {}

    for col in categorical_vars:
        df[col] = df[col].astype("category")
        ref = st.selectbox(
            f"Reference level for {col}",
            df[col].cat.categories,
            key=f"ref_{col}"
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

    st.code(formula)

    # ======================================================
    # 4. FIT MODEL
    # ======================================================

    st.header("2️⃣ Fit Gamma GLM (Log Link)")

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Gamma(link=sm.families.links.log())
    ).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # 5. MODEL FIT STATISTICS
    # ======================================================

    st.header("3️⃣ Model Fit Evaluation")

    null_model = smf.glm(
        formula=response + " ~ 1",
        data=df,
        family=sm.families.Gamma(link=sm.families.links.log())
    ).fit()

    loglik = model.llf
    aic = model.aic
    bic = model.bic
    deviance = model.deviance

    mcfadden_r2 = 1 - (model.llf / null_model.llf)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Log-Likelihood", round(loglik, 2))
    col2.metric("AIC", round(aic, 2))
    col3.metric("BIC", round(bic, 2))
    col4.metric("McFadden R²", round(mcfadden_r2, 4))

    st.write(f"Model Deviance: {deviance:.4f}")

    # ======================================================
    # Likelihood Ratio Test
    # ======================================================

    lr_stat = -2 * (null_model.llf - model.llf)
    df_diff = int(model.df_model)
    p_value_lr = chi2.sf(lr_stat, df_diff)

    st.write(f"LR Test p-value: {p_value_lr:.6f}")

    # ======================================================
    # 6. RESIDUAL DIAGNOSTICS
    # ======================================================

    st.header("4️⃣ Residual Diagnostics")

    fitted = model.fittedvalues
    resid_dev = model.resid_deviance

    fig = px.scatter(x=fitted, y=resid_dev,
                     labels={"x": "Fitted", "y": "Deviance Residuals"},
                     title="Residuals vs Fitted")
    st.plotly_chart(fig)

    # ======================================================
    # 7. INTERPRETATION
    # ======================================================

    st.header("5️⃣ Interpretation")

    conf = model.conf_int()
    conf.columns = ["2.5%", "97.5%"]

    for name, coef in model.params.items():

        if name == "Intercept":
            continue

        exp_coef = np.exp(coef)
        lower = np.exp(conf.loc[name, "2.5%"])
        upper = np.exp(conf.loc[name, "97.5%"])

        st.write(
            f"**{name}** → Multiplicative Effect: "
            f"{exp_coef:.4f} "
            f"(95% CI: {lower:.4f}, {upper:.4f})"
        )

    # ======================================================
    # 8. PREDICTION
    # ======================================================

    st.header("6️⃣ Prediction")

    input_dict = {}

    for var in predictors:
        if var in categorical_vars:
            input_dict[var] = st.selectbox(
                var,
                df[var].cat.categories
            )
        else:
            input_dict[var] = st.number_input(
                var,
                value=float(df[var].mean())
            )

    if st.button("Predict"):

        new_df = pd.DataFrame([input_dict])

        pred = model.get_prediction(new_df)
        pred_summary = pred.summary_frame()

        mean_pred = pred_summary["mean"].values[0]
        lower = pred_summary["mean_ci_lower"].values[0]
        upper = pred_summary["mean_ci_upper"].values[0]

        st.success(
            f"Predicted {response}: {mean_pred:.4f} "
            f"(95% CI: {lower:.4f}, {upper:.4f})"
        )
