import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(layout="wide")
st.title("General Linear Regression Model Lab")

# ======================================================
# 1. DATA UPLOAD
# ======================================================

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.stop()

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
    st.stop()

categorical_vars = st.multiselect(
    "Select Categorical Variables (Factors)",
    predictors
)

# Convert to category
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
# 3. NORMALITY CHECK – RESPONSE
# ======================================================

st.header("2️⃣ Response Normality Check")

if pd.api.types.is_numeric_dtype(df[response]):

    fig = px.histogram(df, x=response,
                       title=f"Histogram of {response}",
                       marginal="box")
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

else:
    st.error("Response must be numeric.")
    st.stop()

# ======================================================
# 4. BUILD FORMULA WITH REFERENCES
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
# 6. MATHEMATICAL EQUATION
# ======================================================

def build_equation(model, response):

    params = model.params
    equation = f"\\hat{{{response}}} = {round(params['Intercept'],4)}"

    for name in params.index:
        if name == "Intercept":
            continue

        coef = round(params[name], 4)
        sign = "+" if coef >= 0 else "-"

        if "C(" in name:
            var_name = name.split("[")[0]
            var_name = var_name.replace("C(", "").split(",")[0]
            level = name.split("T.")[1].replace("]", "")
            equation += f" {sign} {abs(coef)} D_{{{var_name}={level}}}"
        else:
            equation += f" {sign} {abs(coef)} \\cdot {name}"

    return equation

st.subheader("Fitted Regression Equation")
st.latex(build_equation(model, response))

# ======================================================
# 7. INTERPRETATION
# ======================================================

st.subheader("Interpretation of Coefficients")

for name, coef in model.params.items():

    if name == "Intercept":
        continue

    coef = round(coef, 4)

    if "C(" in name:
        var_name = name.split("[")[0]
        var_name = var_name.replace("C(", "").split(",")[0]
        level = name.split("T.")[1].replace("]", "")
        ref = reference_dict[var_name]

        direction = "increases" if coef > 0 else "decreases"

        st.write(
            f"For **{var_name} = {level}**, expected **{response}** "
            f"{direction} by **{abs(coef)} units** compared to "
            f"reference group (**{ref}**), holding other variables constant."
        )
    else:
        direction = "increases" if coef > 0 else "decreases"

        st.write(
            f"For each one-unit increase in **{name}**, expected "
            f"**{response}** {direction} by **{abs(coef)} units**, "
            "holding other variables constant."
        )

# ======================================================
# 8. VIF DIAGNOSTICS
# ======================================================

st.header("4️⃣ Multicollinearity Diagnostics (VIF)")

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
# 9. RESIDUAL NORMALITY CHECK
# ======================================================

st.header("5️⃣ Residual Normality Check")

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

if p_res > 0.05:
    st.success("Residuals appear normally distributed.")
else:
    st.warning("Residuals may violate normality assumption.")

# ======================================================
# 10. PREDICTION
# ======================================================

st.header("6️⃣ Prediction")

input_dict = {}

for var in predictors:
    if var in categorical_vars:
        input_dict[var] = st.selectbox(var, df[var].cat.categories)
    else:
        input_dict[var] = st.number_input(var,
                                          value=float(df[var].mean()))

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
