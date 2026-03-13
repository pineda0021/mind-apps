import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import shapiro, chi2


def run():

    st.title("📘 Gaussian GLM with Ladder-of-Powers Transformation")

    # ======================================================
    # DATA UPLOAD
    # ======================================================

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ======================================================
    # VARIABLE SELECTION WITH REFERENCE LEVELS
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
    # LADDER-OF-POWERS TRANSFORMATION (MATCH R)
    # ======================================================

    st.header("2️⃣ Enter λ for Transformation")

    if not pd.api.types.is_numeric_dtype(df[response]):
        st.error("Response must be numeric.")
        return

    if (df[response] <= 0).any():
        st.error("Transformation requires strictly positive response.")
        return

    lam = st.number_input("Enter λ", value=-1.0, step=0.1)

    df_model = df.copy()
    y = pd.to_numeric(df_model[response], errors="coerce")
    transformed_response = response + "_tr"

    # Exact ladder transformations (match your R code)
    if lam == -2.0:
        df_model[transformed_response] = 0.5 * (1 - 1/(y**2))

    elif lam == -1.0:
        df_model[transformed_response] = 1 - (1 / y)

    elif lam == -0.5:
        df_model[transformed_response] = 2 * (1 - 1/np.sqrt(y))

    elif lam == 0.0:
        df_model[transformed_response] = np.log(y)

    elif lam == 0.5:
        df_model[transformed_response] = 2 * (np.sqrt(y) - 1)

    elif lam == 1.0:
        df_model[transformed_response] = y - 1

    elif lam == 2.0:
        df_model[transformed_response] = 0.5 * (y**2 - 1)

    else:
        df_model[transformed_response] = (y**lam - 1) / lam

    # ======================================================
    # NORMALITY TEST (TRANSFORMED RESPONSE)
    # ======================================================

    st.header("3️⃣ Normality Check (Transformed Response)")

    df_fit = df_model[[transformed_response] + predictors].dropna()
    y_trans = df_fit[transformed_response]

    qq_fig = sm.qqplot(y_trans, line="s")
    st.pyplot(qq_fig.figure)

    stat, p = shapiro(y_trans)

    st.write(f"Shapiro-Wilk Statistic: {stat:.4f}")
    st.write(f"p-value: {p:.4f}")

    # ======================================================
    # FIT GAUSSIAN GLM (IDENTITY LINK)
    # ======================================================

    st.header("4️⃣ Fit Gaussian GLM")

    formula_transformed = transformed_response + " ~ " + " + ".join(terms)

    model = smf.glm(
        formula=formula_transformed,
        data=df_fit,
        family=sm.families.Gaussian(link=sm.families.links.identity())
    ).fit()

    st.subheader("Model Summary")
    st.text(model.summary())

    # ======================================================
    # DEVIANCE (MATCH R EXACTLY)
    # ======================================================

    st.header("5️⃣ Deviance (Likelihood Ratio)")

    null_model = smf.glm(
        transformed_response + " ~ 1",
        data=df_fit,
        family=sm.families.Gaussian(link=sm.families.links.identity())
    ).fit()

    lr_deviance = -2 * (null_model.llf - model.llf)

    st.write(f"Deviance (LR): {lr_deviance:.6f}")

    # ======================================================
    # COEFFICIENT INTERPRETATION
    # ======================================================

    st.subheader("Coefficient Interpretation (Transformed Scale)")

    for name in model.params.index:

        coef = round(model.params[name], 4)
        pval = model.pvalues[name]

        if name == "Intercept":
            st.markdown(
                f"**Intercept ({coef})**: Mean of transformed response "
                f"when predictors are at reference levels."
            )

        elif name.startswith("C(") and "T." in name:
            var_name = name.split("[")[0]
            var_name = var_name.replace("C(", "").split(",")[0]
            level = name.split("T.")[1].rstrip("]")
            st.markdown(
                f"**{var_name} = {level} (β = {coef})**: "
                f"Difference in transformed mean compared to reference level. "
                f"{'Statistically significant.' if pval < 0.05 else 'Not statistically significant.'}"
            )

        else:
            st.markdown(
                f"**{name} (β = {coef})**: "
                f"One-unit increase changes transformed response by {coef}. "
                f"{'Statistically significant.' if pval < 0.05 else 'Not statistically significant.'}"
            )

    # ======================================================
    # PREDICTION (MATCH R INVERSE)
    # ======================================================

    st.header("6️⃣ Prediction")

    input_dict = {}

    for var in predictors:
        if var in categorical_vars:
            input_dict[var] = st.selectbox(var, df[var].cat.categories)
        else:
            input_dict[var] = st.number_input(
                var,
                value=float(pd.to_numeric(df[var], errors="coerce").mean())
            )

    if st.button("Predict"):

        new_df = pd.DataFrame([input_dict])

        for var in categorical_vars:
            new_df[var] = pd.Categorical(
                new_df[var],
                categories=df[var].cat.categories
            )

        prediction_tr = model.predict(new_df)[0]

        # Exact inverse (λ = −1 case matches your R code)
        if lam == -1.0:
            prediction = 1 / (1 - prediction_tr)

        elif lam == 0:
            prediction = np.exp(prediction_tr)

        else:
            prediction = (lam * prediction_tr + 1) ** (1 / lam)

        st.success(f"Predicted {response}: {prediction:.6f}")


if __name__ == "__main__":
    run()
