import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from scipy import stats
from scipy.stats import chi2


def run():

    st.title("📘 Gaussian Linear Model (Ladder-of-Powers λ = -1)")

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
    # 2. VARIABLE SELECTION
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

    # ======================================================
    # 3. FORCE CATEGORY ORDER (CRITICAL FOR MATCHING OUTPUT)
    # ======================================================

    for col in categorical_vars:
        df[col] = df[col].astype("category")

    # ======================================================
    # 4. APPLY λ = -1 TRANSFORMATION EXACTLY
    # ======================================================

    st.header("2️⃣ Transformation (λ = -1)")

    if (df[response] <= 0).any():
        st.error("Response must be strictly positive for λ = -1.")
        return

    df["tr_score"] = 1 - (1 / df[response])

    # ======================================================
    # DISPLAY ORIGINAL VS TRANSFORMED
    # ======================================================

    st.subheader("Original vs Transformed Response")

    comparison_df = df[[response]].copy()
    comparison_df["tr_score"] = df["tr_score"]

    st.dataframe(comparison_df)

    # ======================================================
    # 5. HISTOGRAM + NORMAL OVERLAY (MATCH COLAB)
    # ======================================================

    st.header("3️⃣ Histogram with Normal Overlay")

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
    # 6. SHAPIRO-WILK TEST
    # ======================================================

    st.header("4️⃣ Shapiro-Wilk Test")

    stat, p_value = stats.shapiro(y_trans)

    st.write(f"Shapiro-Wilk Test Statistic: {stat:.4f}")
    st.write(f"P-Value: {p_value:.4f}")

    if p_value > 0.05:
        st.success("Conclusion: Fail to reject H₀ → Data appears normally distributed.")
    else:
        st.warning("Conclusion: Reject H₀ → Data is NOT normally distributed.")

    # ======================================================
    # 7. BUILD FORMULA SAFELY
    # ======================================================

    terms = []

    for var in predictors:
        if var in categorical_vars:
            # User chooses reference
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
    # 8. FIT OLS (MATCH COLAB EXACTLY)
    # ======================================================

    st.header("5️⃣ Fit General Linear Model (OLS)")

    model = sm.ols(
        formula=formula,
        data=df
    ).fit()

    st.text(model.summary())

    # ======================================================
    # 9. ERROR ESTIMATES (MATCH YOUR PRINT)
    # ======================================================

    st.subheader("Error Estimates")

    sigma_unbiased = np.sqrt(model.mse_resid)
    sigma_mle = np.sqrt(model.ssr / model.nobs)

    st.write("MSE:", sigma_unbiased)
    st.write("MLE:", sigma_mle)

    # ======================================================
    # 10. LIKELIHOOD RATIO TEST (R STYLE)
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
