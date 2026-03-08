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
    # 4️⃣ BOX-COX TRANSFORMATION
    # ======================================================

    st.header("3️⃣ Box-Cox Transformation (If Needed)")

    lambda_rec = None
    original_model = None
    transformed_response = None

    if p <= 0.05:

        if (y <= 0).any():
            st.error("Box-Cox requires strictly positive response values.")
            return

        lambdas = np.arange(-3, 3.25, 0.25)
        llf_vals = [boxcox(y, lmbda=l)[1] for l in lambdas]

        boxcox_df = pd.DataFrame({
            "lambda": lambdas,
            "logLik": llf_vals
        })

        ordered_df = boxcox_df.sort_values(by="logLik", ascending=False)
        lambda_hat = ordered_df.iloc[0]["lambda"]

        st.write(f"Estimated λ (Profile Likelihood): {lambda_hat:.4f}")

        # 95% CI
        max_llf = ordered_df.iloc[0]["logLik"]
        cutoff = max_llf - 0.5 * chi2.ppf(0.95, df=1)

        ci_vals = boxcox_df[boxcox_df["logLik"] >= cutoff]["lambda"]
        ci_lower = ci_vals.min()
        ci_upper = ci_vals.max()

        st.write(f"95% CI for λ: [{ci_lower:.3f}, {ci_upper:.3f}]")

        # Range-based recommendation
        def recommend_lambda(lambda_hat):
            if -2.5 <= lambda_hat < -1.5:
                return -2
            elif -1.5 <= lambda_hat < -0.75:
                return -1
            elif -0.75 <= lambda_hat < -0.25:
                return -0.5
            elif -0.25 <= lambda_hat < 0.25:
                return 0
            elif 0.25 <= lambda_hat < 0.75:
                return 0.5
            elif 0.75 <= lambda_hat < 1.5:
                return 1
            elif 1.5 <= lambda_hat <= 2.5:
                return 2
            else:
                return lambda_hat

        lambda_rec = recommend_lambda(lambda_hat)

        st.success(f"Recommended λ (based on standard ranges): {lambda_rec}")

        # Transformation display
        st.subheader("Selected Transformation")

        if lambda_rec == -2:
            st.latex(r"\tilde{y} = \frac{1}{2}\left(1 - \frac{1}{y^2}\right)")
        elif lambda_rec == -1:
            st.latex(r"\tilde{y} = 1 - \frac{1}{y}")
        elif lambda_rec == -0.5:
            st.latex(r"\tilde{y} = 2\left(1 - \frac{1}{\sqrt{y}}\right)")
        elif lambda_rec == 0:
            st.latex(r"\tilde{y} = \ln(y)")
        elif lambda_rec == 0.5:
            st.latex(r"\tilde{y} = 2(\sqrt{y} - 1)")
        elif lambda_rec == 1:
            st.latex(r"\tilde{y} = y - 1")
        elif lambda_rec == 2:
            st.latex(r"\tilde{y} = \frac{1}{2}(y^2 - 1)")
        else:
            st.latex(r"\tilde{y} = \frac{y^\lambda - 1}{\lambda}")

        # Interpretation
        st.subheader("Interpretation of λ")
        st.write(f"""
λ̂ = {lambda_hat:.3f}

The selected λ = {lambda_rec} is chosen to improve normality and stabilize variance.
If λ = 1, the linear transformation is adequate and no nonlinear transformation is required.
""")

        # Apply transformation
        y_transformed = boxcox(y, lmbda=lambda_rec)
        transformed_response = f"{response}_boxcox"
        df[transformed_response] = y_transformed
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
    # 7️⃣ MODEL INTERPRETATION
    # ======================================================

    st.subheader("Model Interpretation")

    st.write(f"""
R² = {model.rsquared:.4f}  
Adjusted R² = {model.rsquared_adj:.4f}  
F-test p-value = {model.f_pvalue:.6f}

If the F-test p-value < 0.05, the overall regression model is statistically significant.
""")

    # ======================================================
    # 8️⃣ COEFFICIENT INTERPRETATION
    # ======================================================

    st.subheader("Interpretation of Coefficients")

    for name, coef in model.params.items():
        if name == "Intercept":
            continue

        if lambda_rec is None:
            scale_note = f"in the original units of {response_original}"
        else:
            scale_note = f"in the transformed scale"

        st.write(f"""
Holding other variables constant:

A one-unit increase in {name}
changes the expected response by {coef:.4f} units
({scale_note}).
""")

    # ======================================================
    # 9️⃣ PREDICTION
    # ======================================================

    st.header("5️⃣ Prediction")

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

            st.success(f"Predicted (Transformed Scale): {pred:.4f}")
            st.success(f"Predicted (Original Scale): {pred_original:.4f}")
        else:
            st.success(f"Predicted {response_original}: {pred:.4f}")

    # ======================================================
    # 🔟 PREDICTED VS ACTUAL
    # ======================================================

    st.header("6️⃣ Predicted vs Actual")

    preds = model.predict(df)

    fig2 = px.scatter(
        x=preds,
        y=df[response],
        labels={'x': 'Predicted', 'y': 'Actual'}
    )

    st.plotly_chart(fig2)
    st.write("Points closer to the diagonal indicate better predictions.")


if __name__ == "__main__":
    run()
