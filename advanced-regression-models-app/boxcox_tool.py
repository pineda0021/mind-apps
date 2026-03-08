import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import plotly.express as px
from scipy.stats import shapiro, chi2


def boxcox_transform(y, lmbda):
    if lmbda == 0:
        return np.log(y)
    return (y**lmbda - 1) / lmbda


def boxcox_loglik(y, lmbda):
    n = len(y)
    log_y_sum = np.sum(np.log(y))
    y_trans = boxcox_transform(y, lmbda)
    s2 = np.var(y_trans, ddof=1)
    return -(n/2)*np.log(s2) + (lmbda - 1)*log_y_sum


def run():

    st.title("General Linear Regression with Box–Cox Transformation")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # =============================
    # VARIABLE SELECTION
    # =============================

    response_original = st.selectbox("Select Response Variable (Y)", df.columns)
    predictors = st.multiselect(
        "Select Predictor Variables",
        [c for c in df.columns if c != response_original]
    )

    if not predictors:
        return

    # =============================
    # NORMALITY CHECK
    # =============================

    y = df[response_original]

    st.header("Response Normality Check")

    stat, p = shapiro(y)
    st.write(f"Shapiro-Wilk p-value: {p:.4f}")

    st.plotly_chart(px.histogram(df, x=response_original, marginal="box"))

    # =============================
    # BOX-COX TRANSFORMATION
    # =============================

    lambda_rec = None
    response = response_original

    if p <= 0.05:

        st.header("Box–Cox Transformation")

        if (y <= 0).any():
            st.error("Response must be strictly positive for Box–Cox.")
            return

        lambdas = np.arange(-3, 3.25, 0.25)
        ll_vals = [boxcox_loglik(y, l) for l in lambdas]

        df_lambda = pd.DataFrame({"lambda": lambdas, "logLik": ll_vals})
        lambda_hat = df_lambda.loc[df_lambda.logLik.idxmax(), "lambda"]

        max_ll = max(ll_vals)
        cutoff = max_ll - 0.5 * chi2.ppf(0.95, 1)

        ci_vals = df_lambda[df_lambda.logLik >= cutoff]["lambda"]
        ci_lower, ci_upper = ci_vals.min(), ci_vals.max()

        st.write(f"λ̂ = {lambda_hat:.3f}")
        st.write(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

        # Practical recommendation
        candidates = [-2, -1, -0.5, 0, 0.5, 1, 2]
        valid = [l for l in candidates if ci_lower <= l <= ci_upper]

        lambda_rec = min(valid, key=lambda x: abs(x - 1)) if valid else lambda_hat

        st.success(f"Recommended λ = {lambda_rec}")

        # Display transformation definition
        st.subheader("Box–Cox Transformation Definition")

        st.latex(r"""
        \tilde{y} =
        \begin{cases}
        \frac{y^\lambda - 1}{\lambda}, & \lambda \ne 0 \\
        \ln(y), & \lambda = 0
        \end{cases}
        """)

        # Apply transformation
        df["y_trans"] = boxcox_transform(y, lambda_rec)
        response = "y_trans"

    # =============================
    # FIT MODEL
    # =============================

    formula = response + " ~ " + " + ".join(predictors)

    model = smf.ols(formula, data=df).fit()

    st.header("Fitted Model")
    st.text(model.summary())

    # =============================
    # MODEL FIT EVALUATION
    # =============================

    st.header("Model Fit Evaluation")

    sigma_hat = np.sqrt(model.mse_resid)
    rmse = np.sqrt(np.mean(model.resid**2))

    st.write(f"σ̂ (Residual SD): {sigma_hat:.4f}")
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"Log-Likelihood: {model.llf:.4f}")
    st.write(f"AIC: {model.aic:.4f}")
    st.write(f"BIC: {model.bic:.4f}")

    # =============================
    # LIKELIHOOD RATIO TEST
    # =============================

    st.header("Likelihood Ratio (Deviance) Test")

    null_model = smf.ols(response + " ~ 1", data=df).fit()

    lr_stat = 2 * (model.llf - null_model.llf)
    df_diff = model.df_model
    p_lr = chi2.sf(lr_stat, df_diff)

    st.write(f"LR Statistic: {lr_stat:.4f}")
    st.write(f"Degrees of Freedom: {int(df_diff)}")
    st.write(f"p-value: {p_lr:.6f}")

    # =============================
    # FITTED REGRESSION EQUATION
    # =============================

    st.header("Fitted Regression Equation (Transformed Scale)")

    equation = r"\widehat{E(\tilde{y})} = "

    for i, (name, coef) in enumerate(model.params.items()):
        if i == 0:
            equation += f"{coef:.3f}"
        else:
            sign = "+" if coef >= 0 else "-"
            equation += f" {sign} {abs(coef):.3f}({name})"

    st.latex(equation)

    # =============================
    # INTERPRETATION
    # =============================

    st.header("Interpretation of Estimated Coefficients")

    if lambda_rec is not None:
        st.write("""
The regression coefficients represent changes in the expected value
of the Box–Cox transformed response.
""")
    else:
        st.write("""
The regression coefficients represent changes in the expected value
of the original response variable.
""")

    for name, coef in model.params.items():
        if name == "Intercept":
            continue
        st.write(
            f"A one-unit increase in {name} changes the expected response by {coef:.4f}, "
            "holding other predictors constant."
        )


if __name__ == "__main__":
    run()
