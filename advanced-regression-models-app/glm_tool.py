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

    # ======================================================
    # IF RESPONSE IS NOT NORMAL → RUN MASS BOXCox AND STOP
    # ======================================================

    if p <= 0.05:

        st.warning("Response does NOT appear normally distributed.")

        st.markdown("### 📌 Box-Cox Transformation (MASS Equivalent Profile Likelihood)")

        if (df[response] <= 0).any():
            st.error("Box-Cox requires strictly positive response values.")
            st.info("Recommended: Fit a Gamma Generalized Linear Model (GLM).")
            st.stop()

        y = df[response]
        n = len(y)

        # Geometric mean scaling (CRITICAL)
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

        # 95% CI
        cutoff = max_loglik - 0.5 * chi2.ppf(0.95, df=1)
        ci_lambdas = lambda_grid[log_likelihoods >= cutoff]

        if len(ci_lambdas) > 0:
            st.write(f"95% CI for λ: ({ci_lambdas.min():.4f}, {ci_lambdas.max():.4f})")

        # Plot profile likelihood
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

    # ======================================================
    # 5. FIT MODEL IF NORMAL → PROCEED WITH OLS
    # ======================================================

    st.success("Response appears normally distributed.")

    st.header("3️⃣ Fit General Linear Model")

    model = smf.ols(formula=formula, data=df).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # 6. MODEL FIT STATISTICS
    # ======================================================

    st.header("4️⃣ Model Fit Evaluation")

    n = int(model.nobs)
    k = int(model.df_model) + 1

    loglik = model.llf
    aic = model.aic
    bic = model.bic

    # FIXED σ̂ and RMSE
    sigma_hat = np.sqrt(model.mse_resid)              # Correct residual SD
    rmse = np.sqrt(np.mean(model.resid ** 2))         # RMSE

    if (n - k - 1) > 0:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = float("nan")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Log-Likelihood", round(loglik, 2))
    col2.metric("AIC", round(aic, 2))
    col3.metric("AICc", round(aicc, 2))
    col4.metric("BIC", round(bic, 2))
    col5.metric("σ̂", round(sigma_hat, 4))

    st.metric("RMSE", round(rmse, 4))

    # ======================================================
    # Interpretation Panel
    # ======================================================

    st.markdown("**Log-Likelihood (ℓ)**")
    st.latex(r"\ell(\hat{\beta})")
    st.markdown("Measures how probable the observed data are under the fitted model.")

    st.markdown("**AIC**")
    st.latex(r"AIC = -2\ell + 2k")
    st.markdown("Balances model fit and complexity. Lower values are preferred.")

    st.markdown("**AICC**")
    st.latex(r"AICC = AIC + \frac{2k(k+1)}{n-k-1}")
    st.markdown("Small-sample corrected AIC. Lower values are preferred.")

    st.markdown("**BIC**")
    st.latex(r"BIC = -2\ell + k\ln(n)")
    st.markdown("Penalizes model complexity more strongly than AIC.")

    st.markdown("**Residual Standard Deviation (σ̂)**")
    st.latex(r"\hat{\sigma} = \sqrt{\frac{SSE}{n-k}}")
    st.markdown("Estimated standard deviation of the regression errors.")

    st.markdown("**RMSE**")
    st.latex(r"RMSE = \sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}")
    st.markdown("Average magnitude of prediction error.")

    # ======================================================
    # 7. LIKELIHOOD RATIO TEST
    # ======================================================

    st.subheader("Likelihood Ratio (Deviance) Test")

    null_model = smf.ols(response + " ~ 1", data=df).fit()

    lr_stat = 2 * (model.llf - null_model.llf)
    df_diff = int(model.df_model)
    p_value_lr = chi2.sf(lr_stat, df_diff)

    st.write(f"LR Statistic: {lr_stat:.4f}")
    st.write(f"Degrees of Freedom: {df_diff}")
    st.write(f"p-value: {p_value_lr:.6f}")

    # ======================================================
    # 8. EQUATION BUILDER
    # ======================================================

    def build_equation(model, response):

        params = model.params
        equation = f"\\widehat{{\\mathbb{{E}}}}({response}) = {round(params['Intercept'],4)}"

        for name in params.index:

            if name == "Intercept":
                continue

            coef = round(params[name], 4)
            sign = "+" if coef >= 0 else "-"

            if name.startswith("C(") and "T." in name:
                var_name = name.split("[")[0]
                var_name = var_name.replace("C(", "").split(",")[0]
                level = name.split("T.")[1].rstrip("]")
                equation += f" {sign} {abs(coef)} D_{{{var_name}={level}}}"
            else:
                equation += f" {sign} {abs(coef)} \\cdot {name}"

        return equation

    st.subheader("Fitted Regression Equation (Full Model)")
    st.latex(build_equation(model, response))


    # ======================================================
    # Coefficient Interpretation
    # ======================================================

    st.subheader("Coefficient Interpretation")

    for name in model.params.index:

        coef = round(model.params[name], 4)
        pval = model.pvalues[name]

        significance = (
            "Statistically significant."
            if pval <= 0.05
            else "Not statistically significant."
        )

        # Intercept
        if name == "Intercept":
            interpretation = (
                f"Estimated mean of {response} when all predictors "
                f"are at their reference levels or equal to zero."
            )
            term_label = f"Intercept"

        # Categorical predictors (dummy variables)
        elif name.startswith("C(") and "T." in name:
            var_name = name.split("[")[0]
            var_name = var_name.replace("C(", "").split(",")[0]
            level = name.split("T.")[1].rstrip("]")

            interpretation = (
                f"The estimated mean difference in {response} between "
                f"{level} and the reference level, holding other variables constant."
            )
            term_label = f"{var_name} = {level}"

        # Continuous predictors
        else:
            interpretation = (
                f"For each one-unit increase in {name}, {response} changes "
                f"by {coef} units, holding other predictors constant."
            )
            term_label = name
       
        # Display
        st.markdown(
            f"**{term_label}**  \n"
            f"- Coefficient (β): {coef:.4f}  \n"
            f"- p-value: {pval:.4f}  \n"
            f"- Interpretation: {interpretation}  \n"
            f"- {significance}"
        )


    # ======================================================
    # 9. PREDICTION
    # ======================================================

    st.header("6️⃣ Prediction")

    input_dict = {}

    for var in predictors:

        if var in categorical_vars:
            input_dict[var] = st.selectbox(var, df[var].cat.categories)
        else:
            numeric_series = pd.to_numeric(df[var], errors="coerce")
            input_dict[var] = st.number_input(var, value=float(numeric_series.mean()))

    if st.button("Predict"):

        new_df = pd.DataFrame([input_dict])

        for var in categorical_vars:
            new_df[var] = pd.Categorical(
                new_df[var],
                categories=df[var].cat.categories
            )

        prediction = model.predict(new_df)[0]
        st.success(f"Predicted {response}: {prediction:.4f}")

    # ======================================================
    # 10. PREDICTED VS ACTUAL
    # ======================================================

    st.header("7️⃣ Predicted vs Actual")

    predicted_vals = model.predict(df)

    fig2 = px.scatter(
        x=predicted_vals,
        y=df[response],
        labels={'x': 'Predicted', 'y': 'Actual'},
        title="Predicted vs Actual Values"
    )

    # perfect prediction line
    min_val = min(predicted_vals.min(), df[response].min())
    max_val = max(predicted_vals.max(), df[response].max())

    fig2.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
    line=dict(color="red", dash="dash")
    )
    st.plotly_chart(fig2)


if __name__ == "__main__":
    run()
