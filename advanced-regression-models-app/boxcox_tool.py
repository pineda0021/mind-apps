import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import shapiro, chi2


def run():

    st.title("📘 General Linear Regression Model")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ======================================================
    # 1️⃣ Model Specification
    # ======================================================

    st.header("1️⃣ Model Specification")

    response = st.selectbox("Select Response Variable (Y)", df.columns)

    predictors = st.multiselect(
        "Select Predictor Variables (X)",
        [col for col in df.columns if col != response]
    )

    if not predictors:
        return

    categorical_vars = st.multiselect("Select Categorical Predictors", predictors)

    reference_dict = {}

    for col in categorical_vars:
        df[col] = df[col].astype("category")
        ref = st.selectbox(
            f"Reference level for {col}",
            df[col].cat.categories,
            key=f"ref_{col}"
        )
        reference_dict[col] = ref

    terms = []
    for var in predictors:
        if var in categorical_vars:
            ref = reference_dict[var]
            terms.append(f'C({var}, Treatment(reference="{ref}"))')
        else:
            terms.append(var)

    formula_original = response + " ~ " + " + ".join(terms)
    st.code(formula_original)

    # ======================================================
    # 2️⃣ Box-Cox Transformation
    # ======================================================

    st.header("2️⃣ Box-Cox Transformation")

    # ---- Lambda Reference Table ----
    st.markdown("### Recommended Box–Cox Transformations by λ")

    st.latex(r"""
    \begin{array}{|c|c|c|c|}
    \hline
    \textbf{Range for } \lambda & 
    \textbf{Recommended } \lambda & 
    \tilde{y} & 
    \textbf{Transformation Name} \\
    \hline
    [-2.5,-1.5) & -2 & \frac{1}{2}\left(1-\frac{1}{y^2}\right) & \text{Inverse Square} \\
    \hline
    [-1.5,-0.75) & -1 & 1-\frac{1}{y} & \text{Inverse (Reciprocal)} \\
    \hline
    [-0.75,-0.25) & -0.5 & 2\left(1-\frac{1}{\sqrt{y}}\right) & \text{Inverse Square Root} \\
    \hline
    [-0.25,0.25) & 0 & \ln(y) & \text{Natural Logarithm} \\
    \hline
    [0.25,0.75) & 0.5 & 2(\sqrt{y}-1) & \text{Square Root} \\
    \hline
    [0.75,1.5) & 1 & y-1 & \text{Linear} \\
    \hline
    [1.5,2.5] & 2 & \frac{1}{2}(y^2-1) & \text{Square} \\
    \hline
    \end{array}
    """)

    df_model = df.copy()
    df_model[response] = pd.to_numeric(df_model[response], errors="coerce")

    if st.checkbox("Apply Box-Cox Transformation"):

        if (df_model[response] <= 0).any():
            st.warning("Response must be strictly positive for Box-Cox.")
            return

        transformed_response = response + "_tr"

        lambda_value = st.number_input(
            "Enter λ (e.g. -2, -1, -0.5, 0, 0.5, 1, 2)",
            value=-1.0,
            step=0.5
        )

        # General Box-Cox formula
        st.latex(r"""
        y^{(\lambda)} =
        \begin{cases}
        \frac{y^\lambda - 1}{\lambda}, & \lambda \neq 0 \\
        \ln(y), & \lambda = 0
        \end{cases}
        """)

       # ======================================================
       # Apply Table-Based Transformation (Exact Match)
       # ======================================================

        if lambda_value == -2:
            df_model[transformed_response] = 0.5 * (1 - 1 / (df_model[response] ** 2))

        elif lambda_value == -1:
            df_model[transformed_response] = 1 - 1 / df_model[response]

        elif lambda_value == -0.5:
            df_model[transformed_response] = 2 * (1 - 1 / np.sqrt(df_model[response]))

        elif lambda_value == 0:
            df_model[transformed_response] = np.log(df_model[response])

        elif lambda_value == 0.5:
            df_model[transformed_response] = 2 * (np.sqrt(df_model[response]) - 1)

        elif lambda_value == 1:
            df_model[transformed_response] = df_model[response] - 1

        elif lambda_value == 2:
            df_model[transformed_response] = 0.5 * (df_model[response] ** 2 - 1)

        else:
            st.error("λ must be one of: -2, -1, -0.5, 0, 0.5, 1, 2")
            return

        df_model = df_model.dropna(subset=[transformed_response] + predictors)

        # ---- Side-by-side Histograms ----
        col1, col2 = st.columns(2)

        with col1:
            fig_orig = px.histogram(
                df_model,
                x=response,
                title="Original Response",
                marginal="box"
            )
            st.plotly_chart(fig_orig)

        with col2:
            fig_trans = px.histogram(
                df_model,
                x=transformed_response,
                title=f"Transformed Response (λ = {lambda_value})",
                marginal="box"
            )
            st.plotly_chart(fig_trans)

        # ---- Shapiro Tests ----
        if len(df_model[response]) >= 3:
            stat_orig, p_orig = shapiro(df_model[response])
            stat_trans, p_trans = shapiro(df_model[transformed_response])

            st.subheader("Normality Tests")

            st.write(f"Original Y — Shapiro p-value: {p_orig:.4f}")
            st.write(f"Transformed Y — Shapiro p-value: {p_trans:.4f}")

            if p_trans > 0.05:
                st.success("Transformed response appears normally distributed.")
            else:
                st.warning("Transformed response does NOT appear normally distributed.")

    else:
        return

    # ======================================================
    # 3️⃣ Model Fitting
    # ======================================================

    st.header("3️⃣ Model Fitting")

    formula_final = transformed_response + " ~ " + " + ".join(terms)

    model = smf.glm(
        formula=formula_final,
        data=df_model,
        family=sm.families.Gaussian()
    ).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # 4️⃣ Deviance Test
    # ======================================================

    null_model = smf.glm(
        formula=transformed_response + " ~ 1",
        data=df_model,
        family=sm.families.Gaussian()
    ).fit()

    deviance = -2 * (null_model.llf - model.llf)
    df_diff = model.df_model - null_model.df_model
    p_value = 1 - chi2.cdf(deviance, df_diff)

    st.subheader("Likelihood Ratio (Deviance) Test")
    st.write(f"Deviance: {deviance:.4f}")
    st.write(f"Degrees of Freedom: {df_diff}")
    st.write(f"p-value: {p_value:.4f}")

    # ======================================================
    # 5️⃣ Model Fit Evaluation
    # ======================================================

    st.header("4️⃣ Model Fit Evaluation")

    n = int(model.nobs)
    k = int(model.df_model) + 1

    loglik = model.llf
    aic = model.aic
    bic = model.bic

    # ---- Robust Residual Computation (GLM-safe) ----
    y_obs = df_model[transformed_response]
    y_hat = model.predict(df_model)

    residuals = y_obs - y_hat

    sigma_hat = np.sqrt(np.sum(residuals**2) / model.df_resid)
    rmse = np.sqrt(np.mean(residuals**2))

    # ---- AICc ----
    if (n - k - 1) > 0:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = float("nan")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Log-Likelihood", round(loglik, 2))
    col2.metric("AIC", round(aic, 2))
    col3.metric("AICc", round(aicc, 2))
    col4.metric("BIC", round(bic, 2))
    col5.metric("σ̂", round(sigma_hat, 4))

    st.metric("RMSE", round(rmse, 4))

    # ======================================================
    # 8. EQUATION BUILDER
    # ======================================================

    def build_equation(model, response):

        params = model.params
        equation = f"\\widehat{{\\mathbb{{E}}}}({response}) = {round(params['Intercept'],4)}"

        for name in params.index:

            if name == "Intercept":
                continue

            coef = round(params[name], 4)
            sign = "+" if coef >= 0 else "-"

            if name.startswith("C(") and "T." in name:
                var_name = name.split("[")[0]
                var_name = var_name.replace("C(", "").split(",")[0]
                level = name.split("T.")[1].rstrip("]")
                equation += f" {sign} {abs(coef)} D_{{{var_name}={level}}}"
            else:
                equation += f" {sign} {abs(coef)} \\cdot {name}"

        return equation

    st.subheader("Fitted Regression Equation (Full Model)")
    st.latex(build_equation(model, response))


    # ======================================================
    # 6️⃣ Interpretation
    # ======================================================

    st.header("5️⃣ Interpretation of Coefficients")

    for term in model.params.index:

        coef = model.params[term]
        pval = model.pvalues[term]

        if term == "Intercept":
            interpretation = (
                f"When all predictors are at reference levels or zero, "
                f"the expected transformed response is {coef:.4f}."
            )
        else:
            interpretation = (
                f"A one-unit increase in '{term}' changes the expected "
                f"transformed response by {coef:.4f}, holding other variables constant."
            )

        significance = (
            "Statistically significant."
            if pval <= 0.05
            else "Not statistically significant."
        )

        st.markdown(f"**{term}**  \n"
                    f"- Coefficient: {coef:.4f}  \n"
                    f"- p-value: {pval:.4f}  \n"
                    f"- {interpretation}  \n"
                    f"- {significance}")

    # ======================================================
    # 7️⃣ Prediction
    # ======================================================

    st.header("6️⃣ Prediction")

    input_dict = {}

    for var in predictors:

        if var in categorical_vars:
            input_dict[var] = st.selectbox(var, df[var].cat.categories)
        else:
            numeric_series = pd.to_numeric(df[var], errors="coerce")
            input_dict[var] = st.number_input(var, value=float(numeric_series.mean()))

    if st.button("Predict"):

        new_df = pd.DataFrame([input_dict])

        for var in categorical_vars:
            new_df[var] = pd.Categorical(
                new_df[var],
                categories=df_model[var].cat.categories
            )

        # 1️⃣ Prediction on transformed scale
        y_trans_pred = float(model.predict(new_df)[0])

        # 2️⃣ Inverse transformation based on the table
        if lambda_value == -2:
            # ½(1 − 1/y²)
            y_original_pred = 1 / np.sqrt(1 - 2 * y_trans_pred)

        elif lambda_value == -1:
            # 1 − 1/y
            y_original_pred = 1 / (1 - y_trans_pred)

        elif lambda_value == -0.5:
            # 2(1 − 1/√y)
            y_original_pred = 1 / (1 - y_trans_pred / 2) ** 2

        elif lambda_value == 0:
            # ln(y)
            y_original_pred = np.exp(y_trans_pred)

        elif lambda_value == 0.5:
            # 2(√y − 1)
            y_original_pred = (y_trans_pred / 2 + 1) ** 2

        elif lambda_value == 1:
            # y − 1
            y_original_pred = y_trans_pred + 1

        elif lambda_value == 2:
            # ½(y² − 1)
            y_original_pred = np.sqrt(2 * y_trans_pred + 1)

        else:
            st.error("λ must be one of: -2, -1, -0.5, 0, 0.5, 1, 2")
            return

    st.subheader("Prediction Results")
    st.write(f"Predicted transformed value: {y_trans_pred:.4f}")
    st.success(f"Predicted original {response}: {y_original_pred:.4f}")

    # ======================================================
    # 8️⃣ Predicted vs Actual
    # ======================================================

    st.header("7️⃣ Predicted vs Actual")

    predicted_vals = model.predict(df_model)

    fig2 = px.scatter(
        x=predicted_vals,
        y=df_model[transformed_response],
        labels={'x': 'Predicted', 'y': 'Actual'},
        title="Predicted vs Actual Values"
    )

    min_val = min(predicted_vals.min(), df_model[transformed_response].min())
    max_val = max(predicted_vals.max(), df_model[transformed_response].max())

    fig2.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="red", dash="dash")
    )

    st.plotly_chart(fig2)


if __name__ == "__main__":
    run()
    
