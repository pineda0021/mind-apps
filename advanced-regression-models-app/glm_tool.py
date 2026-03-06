import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, chi2


def run():

    st.title("General Linear Regression Model Lab")

    # ======================================================
    # 1. DATA UPLOAD
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
    # 2. VARIABLE SELECTION
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
    # 3. RESPONSE NORMALITY CHECK
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
    # 4. BUILD FORMULA
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
    # 5. FIT MODEL
    # ======================================================

    st.header("3️⃣ Fit General Linear Model")

    model = smf.ols(formula=formula, data=df).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # 6. MODEL FIT STATISTICS
    # ======================================================

    st.header("4️⃣ Model Fit Evaluation")

    n = df.shape[0]
    k = int(model.df_model) + 1

    loglik = model.llf
    aic = model.aic
    bic = model.bic

    if (n - k - 1) > 0:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = float("nan")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Log-Likelihood", round(loglik, 2))
    col2.metric("AIC", round(aic, 2))
    col3.metric("AICc", round(aicc, 2))
    col4.metric("BIC", round(bic, 2))

    st.markdown("""
**Interpretation**

- Log-Likelihood measures how well the model explains the observed data.
- AIC, AICc, and BIC penalize model complexity.
- Lower values indicate better balance between fit and complexity.
- AICc is recommended when the sample size is small relative to the number of parameters.
""")

    # ======================================================
    # 7. LIKELIHOOD RATIO TEST
    # ======================================================

    st.subheader("Likelihood Ratio (Deviance) Test")

    null_formula = response + " ~ 1"
    null_model = smf.ols(formula=null_formula, data=df).fit()

    lr_stat = -2 * (null_model.llf - model.llf)
    df_diff = int(model.df_model)
    p_value_lr = chi2.sf(lr_stat, df_diff)

    st.write(f"LR Statistic: {lr_stat:.4f}")
    st.write(f"Degrees of Freedom: {df_diff}")
    st.write(f"p-value: {p_value_lr:.6f}")

    if p_value_lr < 0.05:
        st.success(
            "At α = 0.05, the full model significantly improves "
            "over the intercept-only model."
        )
    else:
        st.warning(
            "The model does not significantly improve over "
            "the intercept-only model."
        )

    # ======================================================
    # 8. MATHEMATICAL EQUATION
    # ======================================================

    def build_equation(model, response):

        params = model.params
        equation = f"\\widehat{{\\mathbb{{E}}}}({response}) = {round(params['Intercept'],4)}"

        for name in params.index:

            if name == "Intercept":
                continue

            coef = round(params[name], 4)
            sign = "+" if coef >= 0 else "-"

            if name.startswith("C("):
                var_name = name.split("[")[0]
                var_name = var_name.replace("C(", "").split(",")[0]
                level = name.split("T.")[1].replace("]", "")
                equation += f" {sign} {abs(coef)} D_{{{var_name}={level}}}"
            else:
                equation += f" {sign} {abs(coef)} \\cdot {name}"

        return equation

 # ======================================================
    # 8.1 REFIT REDUCED MODEL (FIXED)
    # ======================================================

    def refit_reduced_model(full_model, alpha=0.05):

        pvals = full_model.pvalues.drop("Intercept")
        significant_terms = pvals[pvals < alpha].index.tolist()

        if not significant_terms:
            return None

        keep_predictors = set()

        for term in significant_terms:

            if term.startswith("C("):
                base_var = term.split("(")[1].split(",")[0]
                keep_predictors.add(base_var)

            elif "[" in term:
                base_var = term.split("[")[0]
                keep_predictors.add(base_var)

            else:
                keep_predictors.add(term)

        new_terms = []

        for var in predictors:
            if var in keep_predictors:
                if var in categorical_vars:
                    ref = reference_dict[var]
                    new_terms.append(
                        f'C({var}, Treatment(reference="{ref}"))'
                    )
                else:
                    new_terms.append(var)

        if not new_terms:
            return None

        reduced_formula = response + " ~ " + " + ".join(new_terms)

        reduced_model = smf.ols(
            formula=reduced_formula,
            data=df
        ).fit()

        return reduced_model
    # ======================================================
    # 8.3 DISPLAY EQUATIONS
    # ======================================================

    st.subheader("Fitted Regression Equation (Full Model)")
    st.latex(build_equation(model, response))

    reduced_model = refit_reduced_model(model)

    if reduced_model is not None:
        st.subheader("Reduced Model (Refit Using Significant Predictors)")
        st.latex(build_equation(reduced_model, response))

        st.subheader("Reduced Model Summary")
        st.text(reduced_model.summary())
    else:
        st.warning("No predictors are statistically significant at α = 0.05.")

    # ======================================================
    # 9. INTERPRETATION OF COEFFICIENTS
    # ======================================================

    st.header("5️⃣ Interpretation of Coefficients")

    for name, coef in model.params.items():

        if name == "Intercept":
            continue

        coef = round(coef, 4)

        if name.startswith("C("):
            var_name = name.split("[")[0]
            var_name = var_name.replace("C(", "").split(",")[0]
            level = name.split("T.")[1].replace("]", "")
            ref = reference_dict[var_name]

            direction = "increase" if coef > 0 else "decrease"

            st.write(
                f"For **{var_name} = {level}**, the estimated mean **{response}** "
                f"shows a **{direction} of {abs(coef)} units** compared to "
                f"the reference group (**{ref}**), holding other variables constant."
            )

        else:
            st.write(
                f"the estimated mean in **{name}**, "
                f"the estimated mean **{response}** changes by "
                f"{coef} units, holding other variables constant."
            )

       # ======================================================
    # 10. PREDICTION 
    # ======================================================

    st.header("6️⃣ Prediction")

    input_dict = {}

    for var in predictors:

        # If user explicitly selected categorical
        if var in categorical_vars:

            input_dict[var] = st.selectbox(
                var,
                df[var].astype("category").cat.categories
            )

        else:
            # Attempt numeric conversion safely
            numeric_series = pd.to_numeric(df[var], errors="coerce")

            # If numeric conversion works, treat as numeric
            if numeric_series.notna().sum() > 0:

                input_dict[var] = st.number_input(
                    var,
                    value=float(numeric_series.mean())
                )

            # Otherwise automatically treat as categorical
            else:

                df[var] = df[var].astype("category")

                input_dict[var] = st.selectbox(
                    var,
                    df[var].cat.categories
                )

    if st.button("Predict"):
        new_df = pd.DataFrame([input_dict])
        prediction = model.predict(new_df)[0]
        st.success(f"Predicted {response}: {prediction:.4f}")
    # ======================================================
    # 11. PREDICTED VS ACTUAL
    # ======================================================

    st.header("7️⃣ Predicted vs Actual")

    predicted_vals = model.predict(df)

    fig2 = px.scatter(
        x=predicted_vals,
        y=df[response],
        labels={'x': 'Predicted', 'y': 'Actual'},
        title="Predicted vs Actual Values"
    )

    st.plotly_chart(fig2)
