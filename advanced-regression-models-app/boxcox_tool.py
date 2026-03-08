import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

from scipy.stats import shapiro, chi2, boxcox, boxcox_normmax, skew


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
    # 3️⃣ RESPONSE NORMALITY CHECK
    # ======================================================

    st.header("2️⃣ Response Normality Check")

    if not pd.api.types.is_numeric_dtype(df[response]):
        st.error("Response must be numeric.")
        return

    y_clean = df[response].dropna()

    fig = px.histogram(df, x=response, marginal="box",
                       title=f"Histogram of {response}")
    st.plotly_chart(fig)

    qq_fig = sm.qqplot(y_clean, line='s')
    st.pyplot(qq_fig.figure)

    stat, p = shapiro(y_clean)

    st.write(f"Shapiro-Wilk Statistic: {stat:.4f}")
    st.write(f"p-value: {p:.4f}")

    if p > 0.05:
        st.success("Response appears normally distributed.")
    else:
        st.warning("Response does NOT appear normally distributed.")

    # ======================================================
    # 4️⃣ BOX-COX TRANSFORMATION (AUTO IF NEEDED)
    # ======================================================

    st.header("3️⃣ Box-Cox Transformation (If Needed)")

    transformed_response = None
    lambda_rec = None
    original_model = None
    note = ""

    if p <= 0.05:

        if (df[response] <= 0).any():
            st.error("Box-Cox requires strictly positive response values.")
        else:

            y_original = y_clean

            # -------------------------------
            # MASS-STYLE PROFILE LIKELIHOOD
            # -------------------------------

            st.subheader("Profile Log-Likelihood for λ (MASS-Style)")

            lambda_grid = np.arange(-3, 3.01, 0.1)
            loglik_values = []

            for lam in lambda_grid:

                if lam == 0:
                    y_temp = np.log(y_original)
                else:
                    y_temp = (y_original**lam - 1) / lam

                df_temp = df.copy()
                df_temp["_temp_response"] = y_temp

                temp_formula = "_temp_response ~ " + " + ".join(predictors)
                temp_model = smf.ols(temp_formula, data=df_temp).fit()

                loglik_values.append(temp_model.llf)

            profile_df = pd.DataFrame({
                "lambda": lambda_grid,
                "loglik": loglik_values
            })

            idx_max = profile_df["loglik"].idxmax()
            lambda_hat = profile_df.loc[idx_max, "lambda"]
            max_loglik = profile_df.loc[idx_max, "loglik"]

            cutoff = max_loglik - 0.5 * 3.84

            fig_profile = px.line(profile_df,
                                  x="lambda",
                                  y="loglik",
                                  title="Box–Cox Profile Log-Likelihood")

            fig_profile.add_vline(x=lambda_hat,
                                  line_dash="dash",
                                  annotation_text="MLE λ")

            fig_profile.add_hline(y=cutoff,
                                  line_dash="dash",
                                  annotation_text="95% CI cutoff")

            st.plotly_chart(fig_profile)

            ci_lambdas = profile_df.loc[
                profile_df["loglik"] >= cutoff, "lambda"
            ]

            if not ci_lambdas.empty:
                st.write(
                    f"Approximate 95% CI for λ: "
                    f"({ci_lambdas.min():.2f}, {ci_lambdas.max():.2f})"
                )

            st.write(f"Estimated λ (Profile MLE): **{lambda_hat:.4f}**")

            # -------------------------------
            # Snap to recommended λ
            # -------------------------------

            if -0.25 <= lambda_hat < 0.25:
                lambda_rec = 0
                trans_name = "Log Transformation"
                formula_tex = r"\tilde{y} = \ln(y)"
            elif 0.25 <= lambda_hat < 0.75:
                lambda_rec = 0.5
                trans_name = "Square Root"
                formula_tex = r"\tilde{y} = 2(\sqrt{y}-1)"
            elif -0.75 <= lambda_hat < -0.25:
                lambda_rec = -0.5
                trans_name = "Inverse Square Root"
                formula_tex = r"\tilde{y} = 2\left(1-\frac{1}{\sqrt{y}}\right)"
            elif -1.5 <= lambda_hat < -0.75:
                lambda_rec = -1
                trans_name = "Reciprocal"
                formula_tex = r"\tilde{y} = 1 - \frac{1}{y}"
            else:
                lambda_rec = 1
                trans_name = "Linear"

            st.write(f"Recommended λ: {lambda_rec}")
            st.write(f"Transformation: {trans_name}")
            st.latex(formula_tex)

            y_transformed = boxcox(y_original, lmbda=lambda_rec)
            transformed_response = f"{response}_boxcox"
            df.loc[y_original.index, transformed_response] = y_transformed

            original_formula = response_original + " ~ " + " + ".join(predictors)
            original_model = smf.ols(original_formula, data=df).fit()

            response = transformed_response

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

    st.text(model.summary())

    # ======================================================
    # 7️⃣ MODEL COMPARISON
    # ======================================================

    if original_model is not None:
        st.subheader("Model Comparison")
        comp = pd.DataFrame({
            "Model": ["Original", "Box-Cox"],
            "AIC": [original_model.aic, model.aic],
            "BIC": [original_model.bic, model.bic]
        })
        st.dataframe(comp)

    # ======================================================
    # 8️⃣ PREDICTION
    # ======================================================

    st.header("6️⃣ Prediction")

    input_dict = {}
    for var in predictors:
        if var in categorical_vars:
            input_dict[var] = st.selectbox(var, df[var].cat.categories)
        else:
            input_dict[var] = st.number_input(var, value=float(df[var].mean()))

    if st.button("Predict"):

        new_df = pd.DataFrame([input_dict])
        pred = model.predict(new_df)[0]

        if lambda_rec is not None:
            if lambda_rec == 0:
                pred_original = np.exp(pred)
            elif lambda_rec == 1:
                pred_original = pred + 1
            else:
                pred_original = (lambda_rec * pred + 1) ** (1 / lambda_rec)

            st.success(f"Predicted (Transformed): {pred:.4f}")
            st.success(f"Predicted (Original Scale): {pred_original:.4f}")
        else:
            st.success(f"Predicted {response_original}: {pred:.4f}")

    # ======================================================
    # 9️⃣ PREDICTED VS ACTUAL
    # ======================================================

    st.header("7️⃣ Predicted vs Actual")

    preds = model.predict(df)

    if lambda_rec is not None:
        if lambda_rec == 0:
            preds = np.exp(preds)
        elif lambda_rec != 1:
            preds = (lambda_rec * preds + 1) ** (1 / lambda_rec)

        y_actual = df[response_original]
    else:
        y_actual = df[response_original]

    fig2 = px.scatter(x=preds, y=y_actual,
                      labels={'x': 'Predicted', 'y': 'Actual'})
    st.plotly_chart(fig2)
    st.write("Points closer to the diagonal indicate better predictions.")


if __name__ == "__main__":
    run()
