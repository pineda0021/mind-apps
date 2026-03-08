import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from scipy.stats import shapiro, chi2


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

    # ======================================================
    # 4️⃣ BOX-COX TRANSFORMATION (Correct Likelihood)
    # ======================================================

    lambda_rec = None
    transformed_response = None
    original_model = None

    if p <= 0.05:

        st.header("3️⃣ Box-Cox Transformation")

        if (y <= 0).any():
            st.error("Box-Cox requires strictly positive response values.")
            return

        lambdas = np.arange(-3, 3.25, 0.25)
        n = len(y)
        log_y_sum = np.sum(np.log(y))

        llf_vals = []

        for l in lambdas:
            if l == 0:
                y_trans = np.log(y)
            else:
                y_trans = (y**l - 1) / l

            s2 = np.var(y_trans, ddof=1)
            llf = -(n/2)*np.log(s2) + (l - 1)*log_y_sum
            llf_vals.append(llf)

        boxcox_df = pd.DataFrame({
            "lambda": lambdas,
            "logLik": llf_vals
        })

        lambda_hat = boxcox_df.loc[boxcox_df["logLik"].idxmax(), "lambda"]

        max_llf = boxcox_df["logLik"].max()
        cutoff = max_llf - 0.5 * chi2.ppf(0.95, df=1)

        ci_vals = boxcox_df[boxcox_df["logLik"] >= cutoff]["lambda"]
        ci_lower = ci_vals.min()
        ci_upper = ci_vals.max()

        st.write(f"Estimated λ̂ = {lambda_hat:.3f}")
        st.write(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

        candidates = [-2, -1, -0.5, 0, 0.5, 1, 2]
        valid = [l for l in candidates if ci_lower <= l <= ci_upper]

        if valid:
            lambda_rec = min(valid, key=lambda x: abs(x - 1))
        else:
            lambda_rec = lambda_hat

        st.success(f"Recommended λ = {lambda_rec}")

        # Apply transformation
        if lambda_rec == 0:
            y_transformed = np.log(y)
        else:
            y_transformed = (y**lambda_rec - 1) / lambda_rec

        transformed_response = f"{response}_boxcox"
        df[transformed_response] = y_transformed
        response = transformed_response

        original_model = smf.ols(
            response_original + " ~ " + " + ".join(predictors),
            data=df
        ).fit()

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
    # 7️⃣ MODEL FIT EVALUATION
    # ======================================================

    st.header("Model Fit Evaluation")

    sigma_hat = np.sqrt(model.mse_resid)
    rmse = np.sqrt(np.mean(model.resid**2))

    col1, col2 = st.columns(2)
    col1.metric("σ̂ (Residual SD)", round(sigma_hat, 4))
    col2.metric("RMSE", round(rmse, 4))

    st.write(f"""
Log-Likelihood: {model.llf:.4f}  
AIC: {model.aic:.4f}  
BIC: {model.bic:.4f}
""")

    # ======================================================
    # 8️⃣ LIKELIHOOD RATIO TEST
    # ======================================================

    st.header("Likelihood Ratio (Deviance) Test")

    null_model = smf.ols(response + " ~ 1", data=df).fit()

    lr_stat = 2 * (model.llf - null_model.llf)
    df_diff = model.df_model
    p_lr = chi2.sf(lr_stat, df_diff)

    st.write(f"""
LR Statistic: {lr_stat:.4f}  
Degrees of Freedom: {int(df_diff)}  
p-value: {p_lr:.6f}
""")

    # ======================================================
    # 9️⃣ FITTED REGRESSION EQUATION
    # ======================================================

    st.header("Fitted Regression Equation (Transformed Scale)")

    equation = f"{response} = "
    params = model.params

    for i, (name, coef) in enumerate(params.items()):
        if i == 0:
            equation += f"{coef:.4f}"
        else:
            sign = "+" if coef >= 0 else "-"
            equation += f" {sign} {abs(coef):.4f}*{name}"

    st.latex(equation)

    # ======================================================
    # 🔟 COEFFICIENT INTERPRETATION
    # ======================================================

    st.header("Interpretation of Coefficients")

    for name, coef in model.params.items():
        if name == "Intercept":
            continue

        scale_note = (
            f"in the original units of {response_original}"
            if lambda_rec is None
            else "in the transformed scale"
        )

        st.write(f"""
Holding all other variables constant:

A one-unit increase in {name}
changes the expected response by {coef:.4f} units
({scale_note}).
""")

    # ======================================================
    # 1️⃣1️⃣ PREDICTION
    # ======================================================

    st.header("Prediction")

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


if __name__ == "__main__":
    run()
    
    

