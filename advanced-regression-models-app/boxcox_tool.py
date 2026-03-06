import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import boxcox, shapiro
from scipy.special import inv_boxcox
from scipy.stats import chi2


def run():

    st.title("Box–Cox Transformation Model Lab")

    # ======================================================
    # 1. DATA UPLOAD
    # ======================================================

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="boxcox_upload"
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
        st.error("Box–Cox requires strictly positive response values.")
        return

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
            key=f"ref_bc_{col}"
        )
        reference_dict[col] = ref

    # ======================================================
    # 3. ESTIMATE BOX–COX TRANSFORMATION
    # ======================================================

    st.header("2️⃣ Estimate Box–Cox Transformation")

    y_transformed, lambda_bc = boxcox(df[response])
    df["y_bc"] = y_transformed

    st.write(f"Estimated λ (lambda): **{lambda_bc:.4f}**")

    st.markdown("""
If λ = 1 → No transformation  
If λ = 0 → Log transformation  
Other values indicate power transformation.
""")

    # ======================================================
    # 4. BUILD FORMULA
    # ======================================================

    terms = []

    for var in predictors:
        if var in categorical_vars:
            ref = reference_dict[var]
            terms.append(f'C({var}, Treatment(reference="{ref}"))')
        else:
            terms.append(var)

    formula_bc = "y_bc ~ " + " + ".join(terms)

    # ======================================================
    # 5. FIT TRANSFORMED MODEL
    # ======================================================

    st.header("3️⃣ Fit Transformed Model")

    model_bc = smf.ols(formula=formula_bc, data=df).fit()

    st.subheader("Model Summary (Transformed)")
    st.text(model_bc.summary())

    # ======================================================
    # 6. MODEL FIT STATISTICS
    # ======================================================

    st.header("4️⃣ Model Fit Evaluation")

    n = df.shape[0]
    k = int(model_bc.df_model) + 1

    loglik = model_bc.llf
    aic = model_bc.aic
    bic = model_bc.bic

    if (n - k - 1) > 0:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = float("nan")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Log-Likelihood", round(loglik, 2))
    col2.metric("AIC", round(aic, 2))
    col3.metric("AICc", round(aicc, 2))
    col4.metric("BIC", round(bic, 2))

    # ======================================================
    # 7. PREDICTION (BACK-TRANSFORMED)
    # ======================================================

    st.header("5️⃣ Prediction (Back-Transformed)")

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

    if st.button("Predict (Box–Cox)"):

        new_df = pd.DataFrame([input_dict])

        pred_transformed = model_bc.predict(new_df)[0]
        pred_original = inv_boxcox(pred_transformed, lambda_bc)

        st.success(f"Predicted {response} (original scale): {pred_original:.4f}")
