import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from scipy import stats
from scipy.stats import chi2


def run():

    st.title("📘 Gaussian Linear Model with Ladder-of-Powers Transformation")

    # ======================================================
    # 1. DATA UPLOAD
    # ======================================================

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ======================================================
    # 2. VARIABLE SELECTION (WITH REFERENCE LEVELS)
    # ======================================================

    st.header("1️⃣ Select Variables")

    response = st.selectbox("Select Response Variable (Y)", df.columns)

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

    # ======================================================
    # 3. LADDER-OF-POWERS TRANSFORMATION
    # ======================================================

    st.header("2️⃣ Ladder-of-Powers Transformation")

    if not pd.api.types.is_numeric_dtype(df[response]):
        st.error("Response must be numeric.")
        return

    if (df[response] <= 0).any():
        st.error("Transformation requires strictly positive response.")
        return

    ladder_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    lam = st.selectbox("Select λ", ladder_values, index=1)

    # Apply transformation safely

    df_model = df.copy()
    y = pd.to_numeric(df_model[response], errors="coerce")
    transformed_response = response + "_tr"

    if np.isclose(lam, -1.0):
        df_model[transformed_response] = 1 - (1 / y)
    elif np.isclose(lam, 0.0):
        df_model[transformed_response] = np.log(y)
    elif np.isclose(lam, 0.5):
        df_model[transformed_response] = 2 * (np.sqrt(y) - 1)
    elif np.isclose(lam, 1.0):
        df_model[transformed_response] = y - 1
    elif np.isclose(lam, 2.0):
        df_model[transformed_response] = 0.5 * (y**2 - 1)
    elif np.isclose(lam, -2.0):
        df_model[transformed_response] = 0.5 * (1 - 1/(y**2))
    elif np.isclose(lam, -0.5):
        df_model[transformed_response] = 2 * (1 - 1/np.sqrt(y))

    y_trans = df_model[transformed_response].dropna()

    # ======================================================
    # 4. HISTOGRAM + NORMAL OVERLAY
    # ======================================================

    st.header("3️⃣ Histogram with Normal Overlay")

    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist(y_trans, bins=9, density=True)

    mean_val = np.mean(y_trans)
    sd_val = np.std(y_trans)

    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean_val, sd_val)

    ax.plot(x, p, 'r')
    st.pyplot(fig)

    # ======================================================
    # 5. SHAPIRO-WILK TEST
    # ======================================================

    st.header("4️⃣ Shapiro-Wilk Test")

    stat, p_value = stats.shapiro(y_trans)

    st.write(f"Shapiro-Wilk Test Statistic: {stat:.4f}")
    st.write(f"P-Value: {p_value:.4f}")

    if p_value > 0.05:
        st.success("Fail to reject H₀ → Data appears normally distributed.")
    else:
        st.warning("Reject H₀ → Data is NOT normally distributed.")

    # ======================================================
    # 6. FIT OLS MODEL (MATCH COLAB EXACTLY)
    # ======================================================

    st.header("5️⃣ Fit General Linear Model (OLS)")

    formula_transformed = transformed_response + " ~ " + " + ".join(terms)

    model = sm.ols(
        formula=formula_transformed,
        data=df_model
    ).fit()

    st.text(model.summary())

    # ======================================================
    # ERROR ESTIMATES (MATCH YOUR OUTPUT)
    # ======================================================

    sigma_unbiased = np.sqrt(model.mse_resid)
    sigma_mle = np.sqrt(model.ssr / model.nobs)

    st.write(f"MSE: {sigma_unbiased}")
    st.write(f"MLE: {sigma_mle}")

    # ======================================================
    # LIKELIHOOD RATIO TEST (R-STYLE)
    # ======================================================

    null_model = sm.ols(transformed_response + " ~ 1", data=df_model).fit()

    lr_stat = 2 * (model.llf - null_model.llf)
    df_diff = int(model.df_model)
    p_lr = chi2.sf(lr_stat, df_diff)

    st.subheader("Likelihood Ratio Test")
    st.write(f"LR Statistic: {lr_stat}")
    st.write(f"P-Value: {p_lr}")


if __name__ == "__main__":
    run()
