import streamlit as st 
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, chi2
import numpy as np


def run():

    st.title("📘 Gaussian Linear Model (OLS)")

    # ======================================================
    # 1. DATA UPLOAD
    # ======================================================

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="glm_upload"
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

    formula = response + " ~ " + " + ".join(terms)
    st.code(formula)

    # ======================================================
    # 3. RESPONSE NORMALITY CHECK
    # ======================================================

    st.header("2️⃣ Response Normality Check")

    if not pd.api.types.is_numeric_dtype(df[response]):
        st.error("Response must be numeric.")
        return

    fig = px.histogram(
        df,
        x=response,
        title=f"Histogram of {response}",
        marginal="box"
    )
    st.plotly_chart(fig)

    y_clean = df[response].dropna()

    if len(y_clean) < 3:
        st.warning("Not enough data for Shapiro-Wilk test.")
        return

    qq_fig = sm.qqplot(y_clean, line='s')
    st.pyplot(qq_fig.figure)

    stat, p = shapiro(y_clean)

    st.write(f"Shapiro-Wilk Statistic: {stat:.4f}")
    st.write(f"p-value: {p:.4f}")

    if p <= 0.05:

        st.warning("Response does NOT appear normally distributed.")

        st.markdown("### 📌 Box-Cox Transformation (MASS Equivalent Profile Likelihood)")

        if (df[response] <= 0).any():
            st.error("Box-Cox requires strictly positive response values.")
            st.info("Recommended: Fit a Gamma Generalized Linear Model (GLM).")
            st.stop()

        y = df[response]
        n = len(y)

        gm = np.exp(np.mean(np.log(y)))

        lambda_grid = np.arange(-3, 3.25, 0.25)
        log_likelihoods = []

        for lmbda in lambda_grid:

            if abs(lmbda) < 1e-8:
                y_trans = gm * np.log(y)
            else:
                y_trans = (y**lmbda - 1) / (lmbda * gm**(lmbda - 1))

            df_temp = df.copy()
            df_temp["_y_trans_"] = y_trans

            formula_bc = "_y_trans_ ~ " + " + ".join(terms)
            model_bc = smf.ols(formula=formula_bc, data=df_temp).fit()

            sse = np.sum(model_bc.resid ** 2)
            sigma2 = sse / n
            profile_ll = -(n / 2) * np.log(sigma2)

            log_likelihoods.append(profile_ll)

        log_likelihoods = np.array(log_likelihoods)

        best_idx = np.argmax(log_likelihoods)
        best_lambda = lambda_grid[best_idx]
        max_loglik = log_likelihoods[best_idx]

        st.success(f"Recommended λ: {best_lambda:.4f}")

        cutoff = max_loglik - 0.5 * chi2.ppf(0.95, df=1)
        ci_lambdas = lambda_grid[log_likelihoods >= cutoff]

        if len(ci_lambdas) > 0:
            st.write(f"95% CI for λ: ({ci_lambdas.min():.4f}, {ci_lambdas.max():.4f})")

        boxcox_df = pd.DataFrame({
            "lambda": lambda_grid,
            "logLik": log_likelihoods
        })

        fig_lambda = px.line(
            boxcox_df,
            x="lambda",
            y="logLik",
            title="Box-Cox Profile Log-Likelihood"
        )
        fig_lambda.add_hline(y=cutoff, line_dash="dash")
        st.plotly_chart(fig_lambda)

        st.error("⚠ OLS halted due to non-normal response.")

        st.markdown(f"""
### Next Step

Write down the recommended λ = **{best_lambda:.4f}**  
and fit a **Box-Cox Transformation GLM**,  

OR  

Fit a **Gamma GLM** if the response is positive and skewed.
""")

        st.stop()

    st.success("Response appears normally distributed.")

    st.header("3️⃣ Fit General Linear Model")

    model = smf.ols(formula=formula, data=df).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # UPDATED COEFFICIENT INTERPRETATION
    # ======================================================

    st.header("5️⃣ Interpretation")
    st.markdown("Interpretation uses $\\hat{\\beta}$.")

    for term in model.params.index:

        coef = model.params[term]
        pval = model.pvalues[term]

        if term.startswith("C("):
            var_name = term.split("[")[0].replace("C(", "").split(",")[0]
            level = term.split("T.")[-1].replace("]", "")
            label = f"{var_name}[{level}]"
        else:
            label = term

        st.subheader(label)
        st.latex(f"\\hat{{\\beta}} = {coef:.4f}")

        if term in ["Intercept", "const"]:
            st.write(
                f"When all predictors are held at zero and all indicator variables are at their reference levels, "
                f"the estimated mean of {response} is {coef:.4f}."
            )

        elif term.startswith("C("):
            ref = reference_dict.get(var_name, "reference")
            st.write(
                f"If {var_name} is an indicator variable, then "
                f"$\\hat{{\\beta}} = {coef:.4f}$ represents the estimated mean difference in {response} "
                f"for {var_name} = {level} compared with {var_name} = {ref}, holding all other predictors constant."
            )

        else:
            st.write(
                f"If {label} is numeric, then $\\hat{{\\beta}} = {coef:.4f}$ represents "
                f"the estimated change in the mean of {response} for a one-unit increase in {label}, "
                f"holding all other predictors constant."
            )

        st.write(f"Coefficient = {coef:.4f}")
        st.write(f"p-value = {pval:.4f}")

        if pval <= 0.05:
            st.success("Significant")
        else:
            st.warning("Not significant")


if __name__ == "__main__":
    run()
