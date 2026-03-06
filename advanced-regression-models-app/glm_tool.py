import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor

def run():

    st.title("General Linear Regression Model Lab")

    # ======================================================
    # 1. DATA UPLOAD
    # ======================================================

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="glm_upload")

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
    # 3. RESPONSE NORMALITY
    # ======================================================

    st.header("2️⃣ Response Normality Check")

    if not pd.api.types.is_numeric_dtype(df[response]):
        st.error("Response must be numeric.")
        return

    fig = px.histogram(df, x=response,
                       title=f"Histogram of {response}",
                       marginal="box")
    st.plotly_chart(fig)

    qq_fig = sm.qqplot(df[response].dropna(), line='s')
    st.pyplot(qq_fig.figure)

    stat, p = shapiro(df[response].dropna())

    st.write(f"Shapiro-Wilk Statistic: {stat:.4f}")
    st.write(f"p-value: {p:.4f}")

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
    # 6. RESIDUAL NORMALITY
    # ======================================================

    st.header("4️⃣ Residual Normality Check")

    residuals = model.resid

    fig_res = px.histogram(x=residuals,
                           title="Histogram of Residuals",
                           marginal="box")
    st.plotly_chart(fig_res)

    qq_res = sm.qqplot(residuals, line='s')
    st.pyplot(qq_res.figure)

    stat_res, p_res = shapiro(residuals)

    st.write(f"Shapiro-Wilk Statistic: {stat_res:.4f}")
    st.write(f"p-value: {p_res:.4f}")

    # ======================================================
    # 7. VIF
    # ======================================================

    st.header("5️⃣ Multicollinearity Diagnostics (VIF)")

    X = model.model.exog
    vif_data = pd.DataFrame()
    vif_data["Variable"] = model.model.exog_names
    vif_data["VIF"] = [
        variance_inflation_factor(X, i)
        for i in range(X.shape[1])
    ]

    vif_data = vif_data[vif_data["Variable"] != "Intercept"]
    st.dataframe(vif_data.round(3))

    # ======================================================
    # 8. PREDICTION
    # ======================================================

    st.header("6️⃣ Prediction")

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
        prediction = model.predict(new_df)[0]
        st.success(f"Predicted {response}: {prediction:.4f}")
