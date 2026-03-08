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

    y = df[response].dropna()

    fig = px.histogram(df, x=response, marginal="box",
                       title=f"Histogram of {response}")
    st.plotly_chart(fig)

    qq_fig = sm.qqplot(y, line='s')
    st.pyplot(qq_fig.figure)

    stat, p = shapiro(y)

    st.write(f"Shapiro-Wilk p-value: {p:.4f}")

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

    if p <= 0.05:

        if (df[response] <= 0).any():
            st.error("Box-Cox requires strictly positive response values.")
            return

        y_original = y

        # MLE lambda
        lambda_hat = boxcox_normmax(y_original, method="mle")
        st.write(f"Estimated λ (MLE): {lambda_hat:.4f}")

        # Wider, smoother grid
        lambdas = np.arange(-3, 3.25, 0.025)
        llf_vals = [boxcox(y_original, lmbda=l)[1] for l in lambdas]

        max_llf = max(llf_vals)

        # 95% CI cutoff
        cutoff = max_llf - 0.5 * chi2.ppf(0.95, df=1)

        # Determine CI region
        ci_lambdas = [l for l, llf in zip(lambdas, llf_vals) if llf >= cutoff]
        ci_lower = min(ci_lambdas)
        ci_upper = max(ci_lambdas)

        fig_lambda = px.line(
            x=lambdas,
            y=llf_vals,
            labels={"x": "Lambda (λ)", "y": "Log-Likelihood"},
            title="Box-Cox Lambda Optimization Curve"
        )

        # MLE line
        fig_lambda.add_vline(
            x=lambda_hat,
            line_dash="dash",
            line_color="red",
            annotation_text=f"λ̂ = {lambda_hat:.3f}"
        )

        # Reference lines
        fig_lambda.add_vline(
            x=0,
            line_dash="dot",
            line_color="blue",
            annotation_text="λ = 0 (log)"
        )

        fig_lambda.add_vline(
            x=1,
            line_dash="dot",
            line_color="green",
            annotation_text="λ = 1 (linear)"
        )

        # Shade CI
        fig_lambda.add_vrect(
            x0=ci_lower,
            x1=ci_upper,
            fillcolor="gray",
            opacity=0.2,
            line_width=0
        )

        st.plotly_chart(fig_lambda)

        st.write(f"95% CI for λ: [{ci_lower:.3f}, {ci_upper:.3f}]")

        # Snap to recommended λ
        if -0.25 <= lambda_hat <= 0.25:
            lambda_rec = 0
            st.latex(r"\tilde{y} = \ln(y)")
        elif 0.25 < lambda_hat <= 0.75:
            lambda_rec = 0.5
            st.latex(r"\tilde{y} = 2(\sqrt{y}-1)")
        elif -0.75 <= lambda_hat < -0.25:
            lambda_rec = -0.5
            st.latex(r"\tilde{y} = 2(1-\frac{1}{\sqrt{y}})")
        else:
            lambda_rec = lambda_hat  # keep MLE if outside typical range

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

    # ======================================================
    # 8️⃣ INTERPRETATION OF COEFFICIENTS
    # ======================================================

    st.header("5️⃣ Interpretation of Coefficients")

    for name, coef in model.params.items():
        if name == "Intercept":
            continue
        st.write(
            f"For every unit increase in **{name}**, "
            f"the expected value of **{response_original}** changes by "
            f"{round(coef,4)} units (holding others constant)."
        )

    # ======================================================
    # 9️⃣ PREDICTION
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
    # 🔟 PREDICTED VS ACTUAL
    # ======================================================

    st.header("7️⃣ Predicted vs Actual")

    preds = model.predict(df)

    fig2 = px.scatter(
        x=preds,
        y=df[response],
        labels={'x': 'Predicted', 'y': 'Actual'}
    )

    st.plotly_chart(fig2)
    st.write("Points closer to diagonal indicate stronger predictive accuracy.")
