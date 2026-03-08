import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

from scipy.stats import shapiro, chi2, boxcox, boxcox_normmax, skew


def run():

    st.title("General Linear Regression Model")

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
# 4️⃣ BOX-COX TRANSFORMATION (AUTO IF NEEDED)
# ======================================================

st.header("3️⃣ Box-Cox Transformation (If Needed)")

transformed_response = None
lambda_rec = None
original_model = None
note = ""

if p <= 0.05:

    st.warning(
        "The response is not normally distributed. "
        "A Box-Cox transformation will be applied."
    )

    if (df[response] <= 0).any():
        st.error("Box-Cox requires strictly positive response values.")
    else:
        y_original = df[response].dropna()

        # R-style grid: seq(-3, 3, 1/4)
        lambdas = np.arange(-3, 3.25, 0.25)
        llf_vals = [boxcox(y_original, lmbda=l)[1] for l in lambdas]

        lambda_hat = lambdas[np.argmax(llf_vals)]

        st.write(f"Estimated λ (Grid Search): **{lambda_hat:.4f}**")

        # Snap to recommended λ
        if -2.5 <= lambda_hat < -1.5:
            lambda_rec = -2
            trans_name = "Inverse Square"
            formula_tex = r"\tilde{y} = \frac{1}{2}\left(1 - \frac{1}{y^2}\right)"
            note = "Strong compression of large values for extreme right skew."

        elif -1.5 <= lambda_hat < -0.75:
            lambda_rec = -1
            trans_name = "Reciprocal"
            formula_tex = r"\tilde{y} = 1 - \frac{1}{y}"
            note = "Common for rates; reduces influence of large values."

        elif -0.75 <= lambda_hat < -0.25:
            lambda_rec = -0.5
            trans_name = "Inverse Square Root"
            formula_tex = r"\tilde{y} = 2\left(1 - \frac{1}{\sqrt{y}}\right)"
            note = "Moderately reduces right skewness."

        elif -0.25 <= lambda_hat < 0.25:
            lambda_rec = 0
            trans_name = "Log Transformation"
            formula_tex = r"\tilde{y} = \ln(y)"
            note = "Most common transformation; stabilizes variance."

        elif 0.25 <= lambda_hat < 0.75:
            lambda_rec = 0.5
            trans_name = "Square Root"
            formula_tex = r"\tilde{y} = 2(\sqrt{y} - 1)"
            note = "Useful for count data and moderate skew."

        elif 0.75 <= lambda_hat < 1.5:
            lambda_rec = 1
            trans_name = "Linear"
            formula_tex = r"\tilde{y} = y - 1"
            note = "No transformation needed."

        else:
            lambda_rec = 2
            trans_name = "Square"
            formula_tex = r"\tilde{y} = \frac{1}{2}(y^2 - 1)"
            note = "Used for left-skewed data."

        st.write(f"**Recommended λ:** {lambda_rec}")
        st.write(f"**Transformation:** {trans_name}")
        st.latex(formula_tex)
        st.info(f"Teaching Note: {note}")

        # Apply transformation
        y_transformed = boxcox(y_original, lmbda=lambda_rec)
        transformed_response = f"{response}_boxcox"
        df[transformed_response] = y_transformed

        # Skewness comparison
        st.subheader("Skewness Comparison")

        skew_before = skew(y_original)
        skew_after = skew(y_transformed)

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(px.histogram(
                x=y_original,
                title=f"Original (Skew={skew_before:.3f})"
            ))

        with col2:
            st.plotly_chart(px.histogram(
                x=y_transformed,
                title=f"Transformed (Skew={skew_after:.3f})"
            ))

        if abs(skew_after) < abs(skew_before):
            st.success("Skewness reduced after transformation.")

        # Re-check normality
        stat_bc, p_bc = shapiro(y_transformed)
        st.write(f"Post-Transformation p-value: {p_bc:.4f}")

        # Fit original model for comparison
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

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # 7️⃣ MODEL FIT STATISTICS
    # ======================================================

    st.header("5️⃣ Model Fit Evaluation")

    sigma_hat = model.mse_resid ** 0.5
    rmse = (model.resid ** 2).mean() ** 0.5

    col1, col2 = st.columns(2)
    col1.metric("σ̂ (Residual SD)", round(sigma_hat, 4))
    col2.metric("RMSE", round(rmse, 4))

    if original_model is not None:
        st.subheader("Model Comparison")
        comp = pd.DataFrame({
            "Model": ["Original", "Box-Cox"],
            "AIC": [original_model.aic, model.aic],
            "BIC": [original_model.bic, model.bic]
        })
        st.dataframe(comp)

    # ======================================================
    # 8️⃣ FITTED EQUATION
    # ======================================================

    def build_equation(model, response):
        params = model.params
        eq = f"\\hat{{E}}({response}) = {round(params['Intercept'],4)}"
        for name in params.index:
            if name != "Intercept":
                eq += f" + {round(params[name],4)}\\cdot {name}"
        return eq

    st.subheader("Fitted Regression Equation")
    st.latex(build_equation(model, response))

    # ======================================================
    # 9️⃣ INTERPRETATION OF COEFFICIENTS
    # ======================================================

    st.header("5️⃣ Interpretation of Coefficients")

    for name, coef in model.params.items():

        if name == "Intercept":
            continue

        coef = round(coef, 4)

        st.write(
            f"For every-unit increase in **{name}**, "
            f"the expected value of **{response_original}** changes by "
            f"{coef} units, holding other variables constant."
        )

    # ======================================================
    # 🔟 PREDICTION
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

        if lambda_rec is not None and lambda_rec != 1:
            if lambda_rec == 0:
                pred_original = np.exp(pred)
            else:
                pred_original = (lambda_rec * pred + 1) ** (1 / lambda_rec)

            st.success(f"Predicted (Transformed): {pred:.4f}")
            st.success(f"Predicted (Original Scale): {pred_original:.4f}")
        else:
            st.success(f"Predicted {response_original}: {pred:.4f}")

    # ======================================================
    # 1️⃣1️⃣ PREDICTED VS ACTUAL
    # ======================================================

    st.header("7️⃣ Predicted vs Actual")

    preds = model.predict(df)

    fig2 = px.scatter(
        x=preds,
        y=df[response],
        labels={'x': 'Predicted', 'y': 'Actual'}
    )

    st.plotly_chart(fig2)
    st.write("Points closer to the diagonal indicate better predictions.")
