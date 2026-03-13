import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from scipy import stats
from scipy.stats import chi2


def run():

    st.title("📘 Gaussian Linear Model (λ = -1 Transformation)")

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
    # 2. SELECT COLUMN TO TRANSFORM (CRITICAL FIX)
    # ======================================================

    st.header("1️⃣ Select Column to Transform")

    transform_col = st.selectbox(
        "Select the SAME column used in Colab",
        df.columns
    )

    if not pd.api.types.is_numeric_dtype(df[transform_col]):
        st.error("Selected column must be numeric.")
        return

    if (df[transform_col] <= 0).any():
        st.error("Values must be strictly positive for λ = -1.")
        return

    # Apply exact λ = -1 transformation
    df["tr_score"] = 1 - (1 / df[transform_col])

    # ======================================================
    # DISPLAY ORIGINAL VS TRANSFORMED
    # ======================================================

    st.subheader("Verification: Original vs Transformed")

    comparison_df = df[[transform_col]].copy()
    comparison_df["tr_score"] = df["tr_score"]

    st.dataframe(comparison_df)

    st.write("Mean original:", df[transform_col].mean())
    st.write("Mean transformed:", df["tr_score"].mean())

    # ======================================================
    # 3. HISTOGRAM + NORMAL OVERLAY
    # ======================================================

    st.header("2️⃣ Histogram with Normal Overlay")

    y_trans = df["tr_score"].dropna()

    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist(y_trans, bins=9, density=True)

    mean_val = np.mean(y_trans)
    sd_val = np.std(y_trans)

    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean_val, sd_val)

    ax.plot(x, p, 'r')
    ax.set_title("Histogram of Transformed Score with Normal Distribution")

    st.pyplot(fig)

    # ======================================================
    # 4. SHAPIRO-WILK TEST
    # ======================================================

    st.header("3️⃣ Shapiro-Wilk Test")

    stat, p_value = stats.shapiro(y_trans)

    st.write(f"Shapiro-Wilk Test Statistic: {stat:.4f}")
    st.write(f"P-Value: {p_value:.4f}")

    if p_value > 0.05:
        st.success("Conclusion: Fail to reject H₀ → Data appears normally distributed.")
    else:
        st.warning("Conclusion: Reject H₀ → Data is NOT normally distributed.")

    # ======================================================
    # 5. SELECT PREDICTORS
    # ======================================================

    st.header("4️⃣ Select Predictors")

    predictors = st.multiselect(
        "Select Predictor Variables",
        [col for col in df.columns if col not in ["tr_score"]]
    )

    if not predictors:
        return

    categorical_vars = st.multiselect(
        "Select Categorical Variables",
        predictors
    )

    # Force categorical dtype
    for col in categorical_vars:
        df[col] = df[col].astype("category")

    # Build formula dynamically
    terms = []
    for var in predictors:
        if var in categorical_vars:
            ref = st.selectbox(
                f"Reference level for {var}",
                df[var].cat.categories,
                key=f"ref_{var}"
            )
            terms.append(f'C({var}, Treatment(reference="{ref}"))')
        else:
            terms.append(var)

    formula = "tr_score ~ " + " + ".join(terms)

    st.code(formula)

    # ======================================================
    # 6. FIT OLS (MATCH COLAB EXACTLY)
    # ======================================================

    st.header("5️⃣ Fit General Linear Model (OLS)")

    model = sm.ols(formula=formula, data=df).fit()

    st.text(model.summary())

    # ======================================================
    # 7. ERROR ESTIMATES (MATCH YOUR COLAB PRINT)
    # ======================================================

    st.subheader("Error Estimates")

    sigma_unbiased = np.sqrt(model.mse_resid)
    sigma_mle = np.sqrt(model.ssr / model.nobs)

    st.write("MSE:", sigma_unbiased)
    st.write("MLE:", sigma_mle)

    # ======================================================
    # 8. LIKELIHOOD RATIO TEST (R-STYLE)
    # ======================================================

    null_model = sm.ols("tr_score ~ 1", data=df).fit()

    lr_stat = 2 * (model.llf - null_model.llf)
    df_diff = int(model.df_model)
    p_lr = chi2.sf(lr_stat, df_diff)

    st.subheader("Likelihood Ratio Test")
    st.write(f"LR Statistic: {lr_stat}")
    st.write(f"P-Value: {p_lr}")


if __name__ == "__main__":
    run()
