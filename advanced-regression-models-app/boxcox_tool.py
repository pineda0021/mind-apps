import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

from scipy.stats import shapiro, chi2, skew


def run():

    st.title("General Linear Regression Model")

    # ======================================================
    # 1️⃣ DATA UPLOAD
    # ======================================================

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ======================================================
    # 2️⃣ VARIABLE SELECTION
    # ======================================================

    st.header("1️⃣ Select Variables")

    response_original = st.selectbox("Select Response Variable (Y)", df.columns)
    response = response_original

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
    # 3️⃣ BUILD FORMULA FIRST (CRITICAL FIX)
    # ======================================================

    terms = []
    for var in predictors:
        if var in categorical_vars:
            ref = reference_dict[var]
            terms.append(f'C({var}, Treatment(reference="{ref}"))')
        else:
            terms.append(var)

    formula_base = " + ".join(terms)

    # ======================================================
    # 4️⃣ NORMALITY CHECK
    # ======================================================

    st.header("2️⃣ Response Normality Check")

    y = df[response].dropna()

    fig = px.histogram(df, x=response, marginal="box")
    st.plotly_chart(fig)

    qq_fig = sm.qqplot(y, line='s')
    st.pyplot(qq_fig.figure)

    stat, p = shapiro(y)
    st.write(f"Shapiro-Wilk p-value: {p:.4f}")

    # ======================================================
    # 5️⃣ BOX-COX (PROFILE LIKELIHOOD)
    # ======================================================

    lambda_rec = None
    original_model = None

    if p <= 0.05 and (y > 0).all():

        st.header("3️⃣ Box-Cox Profile Likelihood")

        lambdas = np.arange(-3, 3.01, 0.25)
        loglik_values = []

        for lam in lambdas:

            if lam == 0:
                y_temp = np.log(y)
            else:
                y_temp = (y**lam - 1) / lam

            df_temp = df.copy()
            df_temp["_temp_y"] = y_temp

            formula_temp = "_temp_y ~ " + formula_base
            temp_model = smf.ols(formula_temp, data=df_temp).fit()

            loglik_values.append(temp_model.llf)

        profile_df = pd.DataFrame({
            "lambda": lambdas,
            "loglik": loglik_values
        })

        lambda_hat = profile_df.loc[
            profile_df["loglik"].idxmax(),
            "lambda"
        ]

        st.write(f"Estimated λ (Profile MLE): {lambda_hat:.4f}")

        fig_profile = px.line(profile_df, x="lambda", y="loglik")
        st.plotly_chart(fig_profile)

        lambda_rec = lambda_hat

        # Apply transformation
        if lambda_rec == 0:
            df[response + "_boxcox"] = np.log(df[response])
        else:
            df[response + "_boxcox"] = (
                df[response]**lambda_rec - 1
            ) / lambda_rec

        original_model = smf.ols(
            response_original + " ~ " + formula_base,
            data=df
        ).fit()

        response = response + "_boxcox"

    # ======================================================
    # 6️⃣ FIT MODEL
    # ======================================================

    st.header("4️⃣ Fit General Linear Model")

    formula = response + " ~ " + formula_base
    model = smf.ols(formula=formula, data=df).fit()

    st.text(model.summary())

    # ======================================================
    # 7️⃣ DEVIANCE (LIKELIHOOD RATIO TEST)
    # ======================================================

    st.header("5️⃣ Likelihood Ratio (Deviance) Test")

    null_model = smf.ols(response + " ~ 1", data=df).fit()

    lr_stat = -2 * (null_model.llf - model.llf)
    df_diff = int(model.df_model)
    p_lr = chi2.sf(lr_stat, df_diff)

    st.write(f"LR Statistic: {lr_stat:.4f}")
    st.write(f"p-value: {p_lr:.6f}")

    # ======================================================
    # 8️⃣ INTERPRETATION
    # ======================================================

    st.header("6️⃣ Interpretation of Coefficients")

    for name, coef in model.params.items():
        if name != "Intercept":
            if lambda_rec is not None:
                st.write(
                    f"For every-unit increase in {name}, "
                    f"the expected value of the transformed response "
                    f"changes by {coef:.4f}, holding others constant."
                )
            else:
                st.write(
                    f"For every-unit increase in {name}, "
                    f"the expected value of {response_original} "
                    f"changes by {coef:.4f}, holding others constant."
                )

    # ======================================================
    # 9️⃣ PREDICTION
    # ======================================================

    st.header("7️⃣ Prediction")

    input_dict = {}

    for var in predictors:
        if var in categorical_vars:
            input_dict[var] = st.selectbox(var, df[var].cat.categories)
        else:
            input_dict[var] = st.number_input(
                var,
                value=float(df[var].mean())
            )

    if st.button("Predict"):

        new_df = pd.DataFrame([input_dict])
        pred = model.predict(new_df)[0]

        if lambda_rec is not None:
            if lambda_rec == 0:
                pred_original = np.exp(pred)
            else:
                pred_original = (
                    lambda_rec * pred + 1
                )**(1 / lambda_rec)

            st.success(f"Predicted (Original Scale): {pred_original:.4f}")
        else:
            st.success(f"Predicted: {pred:.4f}")


if __name__ == "__main__":
    run()
