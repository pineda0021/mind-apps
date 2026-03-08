import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

from scipy.stats import shapiro, chi2, boxcox, boxcox_normmax


def run():

    st.title("General Linear Regression Model Lab (College-Level)")

    # ======================================================
    # 1️⃣ DATA UPLOAD
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
    # 2️⃣ VARIABLE SELECTION
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

    # ======================================================
    # 3️⃣ RESPONSE NORMALITY CHECK
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

    qq_fig = sm.qqplot(df[response].dropna(), line='s')
    st.pyplot(qq_fig.figure)

    stat, p = shapiro(df[response].dropna())

    st.write(f"Shapiro-Wilk Statistic: {stat:.4f}")
    st.write(f"p-value: {p:.4f}")

    if p > 0.05:
        st.success("Response appears normally distributed.")
    else:
        st.warning("Response does NOT appear normally distributed.")

    # ======================================================
    # 4️⃣ BOX-COX GUIDED TRANSFORMATION
    # ======================================================

    st.header("3️⃣ Box-Cox Transformation (If Needed)")

    lambda_hat = None
    transformed = False

    if p <= 0.05:

        st.warning("Response not normal. Box-Cox will guide transformation.")

        if (df[response] <= 0).any():
            st.error("Box-Cox requires strictly positive response values.")
        else:
            y_original = df[response]

            lambda_mle = boxcox_normmax(y_original, method="mle")

            lambdas = np.linspace(-2, 2, 200)
            llf_vals = [boxcox(y_original, lmbda=l)[1] for l in lambdas]

            st.subheader("Lambda Optimization (Profile Log-Likelihood)")

            fig_lambda = px.line(
                x=lambdas,
                y=llf_vals,
                labels={"x": "Lambda (λ)", "y": "Log-Likelihood"},
                title="Box-Cox Lambda Optimization Curve"
            )

            fig_lambda.add_vline(
                x=lambda_mle,
                line_dash="dash",
                annotation_text=f"λ̂ = {lambda_mle:.3f}"
            )

            st.plotly_chart(fig_lambda)

            st.write(f"Estimated λ (MLE): **{lambda_mle:.4f}**")

            # ======================================================
            # Lambda Interpretation Table
            # ======================================================

            lambda_table = pd.DataFrame({
                "Recommended λ": [-2, -1, -0.5, 0, 0.5, 1, 2],
                "Transformation": [
                    "1 / y²",
                    "1 / y",
                    "1 / √y",
                    "ln(y)",
                    "√y",
                    "y",
                    "y²"
                ]
            })

            st.dataframe(lambda_table)

            lambda_hat = lambda_table.iloc[
                (lambda_table["Recommended λ"] - lambda_mle).abs().argsort()[0]
            ]["Recommended λ"]

            st.info(f"Using Recommended λ = {lambda_hat}")

            transformed = True

            # Exact named transformations
            if lambda_hat == -2:
                df[response] = 1 / (y_original ** 2)
                st.latex(r"y^* = \frac{1}{y^2}")

            elif lambda_hat == -1:
                df[response] = 1 / y_original
                st.latex(r"y^* = \frac{1}{y}")

            elif lambda_hat == -0.5:
                df[response] = 1 / np.sqrt(y_original)
                st.latex(r"y^* = \frac{1}{\sqrt{y}}")

            elif lambda_hat == 0:
                df[response] = np.log(y_original)
                st.latex(r"y^* = \ln(y)")

            elif lambda_hat == 0.5:
                df[response] = np.sqrt(y_original)
                st.latex(r"y^* = \sqrt{y}")

            elif lambda_hat == 1:
                st.latex(r"y^* = y")

            elif lambda_hat == 2:
                df[response] = y_original ** 2
                st.latex(r"y^* = y^2")

    # ======================================================
    # 5️⃣ BUILD FORMULA
    # ======================================================

    terms = []

    for var in predictors:
        if var in categorical_vars:
            ref = reference_dict[var]
            terms.append(f'C({var}, Treatment(reference=\"{ref}\"))')
        else:
            terms.append(var)

    formula = response + " ~ " + " + ".join(terms)

    # ======================================================
    # 6️⃣ FIT MODEL
    # ======================================================

    st.header("4️⃣ Fit General Linear Model")

    model = smf.ols(formula=formula, data=df).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # 7️⃣ FITTED REGRESSION EQUATION
    # ======================================================

    if transformed:
        st.subheader("Fitted Regression Equation (on Transformed Scale)")
    else:
        st.subheader("Fitted Regression Equation")

    coefs = model.params
    equation_terms = []

    for name, coef in coefs.items():
        if name == "Intercept":
            equation_terms.append(f"{coef:.4f}")
        else:
            equation_terms.append(f"{coef:.4f}({name})")

    equation = response + " = " + " + ".join(equation_terms)
    st.code(equation)

    if transformed:
        st.info("Interpret coefficients on the transformed scale.")
