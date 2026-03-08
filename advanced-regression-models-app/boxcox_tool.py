import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

from scipy.stats import shapiro, chi2, boxcox, skew


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

    y = df[response].dropna()

    if not pd.api.types.is_numeric_dtype(y):
        st.error("Response must be numeric.")
        return

    st.plotly_chart(px.histogram(df, x=response, marginal="box"))
    qq_fig = sm.qqplot(y, line='s')
    st.pyplot(qq_fig.figure)

    stat, p = shapiro(y)
    st.write(f"Shapiro-Wilk p-value: {p:.4f}")

    if p > 0.05:
        st.success("Response appears normally distributed.")
    else:
        st.warning("Response does NOT appear normally distributed.")

    # ======================================================
    # 4️⃣ BOX-COX TRANSFORMATION (R-STYLE)
    # ======================================================

    st.header("3️⃣ Box-Cox Transformation (If Needed)")

    transformed_response = None
    lambda_rec = None
    original_model = None

    if p <= 0.05:

        if (y <= 0).any():
            st.error("Box-Cox requires strictly positive response values.")
            return

        # R-style lambda grid
        lambdas = np.arange(-3, 3.25, 0.25)
        llf_vals = [boxcox(y, lmbda=l)[1] for l in lambdas]

        boxcox_df = pd.DataFrame({
            "lambda": lambdas,
            "logLik": llf_vals
        })

        ordered_df = boxcox_df.sort_values(by="logLik", ascending=False)
        lambda_hat = ordered_df.iloc[0]["lambda"]

        st.write(f"Estimated λ (Profile Likelihood): **{lambda_hat:.4f}**")
        st.write("Top λ values by log-likelihood:")
        st.dataframe(ordered_df.head())

        # 95% CI
        max_llf = ordered_df.iloc[0]["logLik"]
        cutoff = max_llf - 0.5 * chi2.ppf(0.95, df=1)

        ci_vals = boxcox_df[boxcox_df["logLik"] >= cutoff]["lambda"]
        ci_lower = ci_vals.min()
        ci_upper = ci_vals.max()

        st.write(f"95% CI for λ: [{ci_lower:.3f}, {ci_upper:.3f}]")

        # Automatic snapping
        recommended = [-2, -1, -0.5, 0, 0.5, 1, 2]
        valid = [l for l in recommended if ci_lower <= l <= ci_upper]

        if valid:
            lambda_rec = min(valid, key=lambda x: abs(x - 1))
            st.success(f"Using recommended λ = {lambda_rec}")
        else:
            lambda_rec = lambda_hat
            st.warning("Using MLE λ (no recommended value inside CI).")

        # Plot profile likelihood
        fig_lambda = px.line(
            boxcox_df,
            x="lambda",
            y="logLik",
            title="Box-Cox Profile Log-Likelihood",
        )

        fig_lambda.add_vline(x=lambda_hat, line_dash="dash", line_color="red",
                             annotation_text=f"λ̂={lambda_hat:.2f}")
        fig_lambda.add_vline(x=0, line_dash="dot", line_color="blue",
                             annotation_text="λ=0")
        fig_lambda.add_vline(x=1, line_dash="dot", line_color="green",
                             annotation_text="λ=1")

        fig_lambda.add_vrect(x0=ci_lower, x1=ci_upper,
                             fillcolor="gray", opacity=0.2, line_width=0)

        st.plotly_chart(fig_lambda)

        # Apply transformation
        y_transformed = boxcox(y, lmbda=lambda_rec)
        transformed_response = f"{response}_boxcox"
        df[transformed_response] = y_transformed

        # Skewness comparison
        st.subheader("Skewness Comparison")

        skew_before = skew(y)
        skew_after = skew(y_transformed)

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(px.histogram(x=y,
                                         title=f"Original (Skew={skew_before:.3f})"))

        with col2:
            st.plotly_chart(px.histogram(x=y_transformed,
                                         title=f"Transformed (Skew={skew_after:.3f})"))

        response = transformed_response

        # Fit original model for comparison
        original_formula = response_original + " ~ " + " + ".join(predictors)
        original_model = smf.ols(original_formula, data=df).fit()

    # ======================================================
    # 5️⃣ BUILD FORMULA
    # ======================================================

    terms = []
    for var in predictors:
        if var in categorical_vars:
            ref = reference_dict[var]
            terms.append(f'C({var}, Treatment(reference="{ref}"))')
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
    # 8️⃣ INTERPRETATION OF COEFFICIENTS
    # ======================================================

    st.header("5️⃣ Interpretation of Coefficients")

    for name, coef in model.params.items():
        if name == "Intercept":
            continue
        st.write(
            f"For every-unit increase in **{name}**, "
            f"the expected value of **{response_original}** changes by "
            f"{round(coef,4)} units (holding others constant)."
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

        if lambda_rec is not None:
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
    st.write("Points closer to diagonal indicate better predictions.")
