import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, chi2


# ======================================================
# PROFILE LIKELIHOOD BOX–COX (MASS-STYLE)
# ======================================================

def boxcox_profile_mle(df, response, formula, lambda_grid=None):

    if lambda_grid is None:
        lambda_grid = np.arange(-3, 3.25, 0.25)

    y = df[response]

    if (y <= 0).any():
        shift = abs(y.min()) + 1
        y = y + shift
    else:
        shift = 0

    log_likelihoods = []

    for lam in lambda_grid:

        if lam == 0:
            y_trans = np.log(y)
        else:
            y_trans = (y**lam - 1) / lam

        df_temp = df.copy()
        df_temp[response] = y_trans

        model = smf.ols(formula=formula, data=df_temp).fit()
        log_likelihoods.append(model.llf)

    results = pd.DataFrame({
        "lambda": lambda_grid,
        "log_likelihood": log_likelihoods
    })

    best_row = results.loc[results["log_likelihood"].idxmax()]

    return best_row["lambda"], results, shift


# ======================================================
# MAIN APP
# ======================================================

def run():

    st.title("General Linear Regression Model Lab")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)

    df.columns = df.columns.str.replace(r"[^\w]", "_", regex=True)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ======================================================
    # VARIABLE SELECTION
    # ======================================================

    st.header("1️⃣ Select Variables")

    response = st.selectbox("Select Response Variable (Y)", df.columns)

    predictors = st.multiselect(
        "Select Predictor Variables (X)",
        [col for col in df.columns if col != response]
    )

    if not predictors:
        return

    formula = response + " ~ " + " + ".join(predictors)

    # ======================================================
    # ORIGINAL MODEL
    # ======================================================

    st.header("2️⃣ Original Model")

    original_model = smf.ols(formula=formula, data=df).fit()

    st.text(original_model.summary())

    # ======================================================
    # BOX–COX PROFILE LIKELIHOOD
    # ======================================================

    st.header("3️⃣ Box–Cox Profile Likelihood")

    if st.checkbox("Estimate λ via Profile Likelihood (MASS-style)"):

        lambda_opt, profile_table, shift = boxcox_profile_mle(
            df, response, formula
        )

        st.write(f"Optimal λ = {round(lambda_opt,4)}")

        fig = px.line(
            profile_table,
            x="lambda",
            y="log_likelihood",
            title="Box–Cox Profile Log-Likelihood"
        )

        st.plotly_chart(fig)

        # ======================================================
        # REFIT TRANSFORMED MODEL
        # ======================================================

        y = df[response]

        if shift != 0:
            y = y + shift

        if lambda_opt == 0:
            y_trans = np.log(y)
        else:
            y_trans = (y**lambda_opt - 1) / lambda_opt

        df_trans = df.copy()
        df_trans[response] = y_trans

        transformed_model = smf.ols(formula=formula, data=df_trans).fit()

        st.header("4️⃣ Transformed Model")

        st.text(transformed_model.summary())

        # AIC Comparison
        st.subheader("AIC Comparison")
        col1, col2 = st.columns(2)
        col1.metric("Original AIC", round(original_model.aic,2))
        col2.metric("Transformed AIC", round(transformed_model.aic,2))

        # ======================================================
        # TRANSFORMED EQUATION
        # ======================================================

        st.subheader("Estimated Regression Equation (Transformed Scale)")

        params = transformed_model.params
        eq = r"\widehat{\tilde{y}} = " + str(round(params["Intercept"],4))

        for name in params.index:
            if name == "Intercept":
                continue
            coef = round(params[name],4)
            sign = "+" if coef >= 0 else "-"
            eq += f" {sign} {abs(coef)} \\cdot {name}"

        st.latex(eq)

        # ======================================================
        # BACK-TRANSFORMATION
        # ======================================================

        st.subheader("Back-Transformation to Original Scale")

        st.latex(r"""
        \hat{y}
        =
        \begin{cases}
        (\lambda \widehat{\tilde{y}} + 1)^{1/\lambda}, & \lambda \neq 0 \\
        \exp(\widehat{\tilde{y}}), & \lambda = 0
        \end{cases}
        """)

        # ======================================================
        # PREDICTION
        # ======================================================

        st.header("5️⃣ Prediction")

        input_dict = {}

        for var in predictors:
            input_dict[var] = st.number_input(
                var,
                value=float(df[var].mean())
            )

        if st.button("Predict"):

            new_df = pd.DataFrame([input_dict])

            y_pred_trans = transformed_model.predict(new_df)[0]

            if lambda_opt == 0:
                y_pred = np.exp(y_pred_trans)
            else:
                y_pred = (lambda_opt * y_pred_trans + 1)**(1/lambda_opt)

            if shift != 0:
                y_pred -= shift

            st.success(f"Predicted {response} (back-transformed): {round(y_pred,4)}")


if __name__ == "__main__":
    run()
