import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import boxcox, shapiro
from scipy.special import inv_boxcox

st.set_page_config(layout="wide")
st.title("Generalized Linear Model Lab")

# ----------------------------
# 1. Upload Data
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
st.dataframe(df.head())

# ----------------------------
# 2. Variable Selection
# ----------------------------
response = st.selectbox("Response Variable", df.columns)
predictors = st.multiselect(
    "Predictor Variables",
    [c for c in df.columns if c != response]
)

categorical_vars = st.multiselect("Categorical Variables", predictors)

for col in categorical_vars:
    df[col] = df[col].astype("category")

if not predictors:
    st.stop()

formula = response + " ~ " + " + ".join(predictors)

# ----------------------------
# 3. Choose Model
# ----------------------------
model_choice = st.radio(
    "Choose Model Type",
    ["Gaussian Linear Model (OLS)",
     "Box-Cox Transformation + OLS",
     "Gamma GLM (Log Link)"]
)

# ====================================================
# 1️⃣ GAUSSIAN OLS
# ====================================================
if model_choice == "Gaussian Linear Model (OLS)":

    model = smf.ols(formula=formula, data=df).fit()
    st.text(model.summary())

    st.subheader("Diagnostics")
    residuals = model.resid

    fig = px.scatter(x=model.fittedvalues,
                     y=residuals,
                     labels={"x": "Fitted", "y": "Residuals"})
    st.plotly_chart(fig)

    stat, p = shapiro(residuals)
    st.write(f"Shapiro-Wilk p-value: {p:.4f}")

    # Prediction
    st.subheader("Prediction")
    input_dict = {}
    for var in predictors:
        if var in categorical_vars:
            input_dict[var] = st.selectbox(var, df[var].cat.categories)
        else:
            input_dict[var] = st.number_input(var, float(df[var].mean()))

    if st.button("Predict"):
        new_df = pd.DataFrame([input_dict])
        pred = model.predict(new_df)[0]
        st.success(f"Predicted {response}: {pred:.4f}")


# ====================================================
# 2️⃣ BOX-COX
# ====================================================
elif model_choice == "Box-Cox Transformation + OLS":

    if (df[response] <= 0).any():
        st.error("Box-Cox requires strictly positive response.")
        st.stop()

    y_transformed, lambda_bc = boxcox(df[response])
    df["y_bc"] = y_transformed

    formula_bc = "y_bc ~ " + " + ".join(predictors)
    model = smf.ols(formula=formula_bc, data=df).fit()

    st.write(f"Estimated λ: {lambda_bc:.4f}")
    st.text(model.summary())

    # Prediction
    st.subheader("Prediction")
    input_dict = {}
    for var in predictors:
        if var in categorical_vars:
            input_dict[var] = st.selectbox(var, df[var].cat.categories)
        else:
            input_dict[var] = st.number_input(var, float(df[var].mean()))

    if st.button("Predict"):
        new_df = pd.DataFrame([input_dict])
        pred_trans = model.predict(new_df)[0]
        pred_original = inv_boxcox(pred_trans, lambda_bc)
        st.success(f"Predicted {response}: {pred_original:.4f}")


# ====================================================
# 3️⃣ GAMMA GLM
# ====================================================
elif model_choice == "Gamma GLM (Log Link)":

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Gamma(sm.families.links.log())
    ).fit()

    st.text(model.summary())

    st.subheader("Prediction")

    input_dict = {}
    for var in predictors:
        if var in categorical_vars:
            input_dict[var] = st.selectbox(var, df[var].cat.categories)
        else:
            input_dict[var] = st.number_input(var, float(df[var].mean()))

    if st.button("Predict"):
        new_df = pd.DataFrame([input_dict])
        pred = model.predict(new_df)[0]
        st.success(f"Predicted {response}: {pred:.4f}")
