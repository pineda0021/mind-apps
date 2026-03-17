import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.metrics import roc_curve, auc

def run():

```
st.title("📘 Complementary Log–Log Regression Model")

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
# 2️⃣ MODEL SPECIFICATION
# ======================================================

st.header("1️⃣ Model Specification")

response_original = st.selectbox(
    "Select Binary Response Variable (Y)", df.columns
)

df[response_original] = df[response_original].astype("category")

ref_level = st.selectbox(
    "Select reference level for response (coded as 0)",
    df[response_original].cat.categories
)

df["response_binary"] = (df[response_original] != ref_level).astype(int)

st.info(
    f"The model estimates the probability that "
    f"{response_original} ≠ '{ref_level}'."
)

response = "response_binary"

predictors = st.multiselect(
    "Select Predictor Variables (X)",
    [c for c in df.columns if c != response_original]
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

formula = response + " ~ " + " + ".join(terms)

st.code(formula)

# ======================================================
# 3️⃣ RESPONSE DIAGNOSTICS
# ======================================================

st.header("2️⃣ Response Diagnostics")

df_model = df.copy()

df_model[response] = pd.to_numeric(df_model[response], errors="coerce")

df_model = df_model[[response] + predictors].dropna()

for col in categorical_vars:
    df_model[col] = df_model[col].astype("category")

fig = px.histogram(
    df_model,
    x=response,
    title="Distribution of Binary Response",
    color=response
)

st.plotly_chart(fig)

# ======================================================
# 4️⃣ MODEL FITTING
# ======================================================

st.header("3️⃣ Model Fitting")

model = smf.glm(
    formula=formula,
    data=df_model,
    family=sm.families.Binomial(link=sm.families.links.cloglog())
).fit()

st.subheader("Cloglog Regression Summary")
st.text(model.summary())

# ======================================================
# 5️⃣ LIKELIHOOD RATIO TEST
# ======================================================

st.subheader("Likelihood Ratio Test")

null_model = smf.glm(
    response + " ~ 1",
    data=df_model,
    family=sm.families.Binomial(link=sm.families.links.cloglog())
).fit()

lr_stat = 2 * (model.llf - null_model.llf)
df_diff = int(model.df_model)
p_value = chi2.sf(lr_stat, df_diff)

st.write(f"LR Statistic: {lr_stat:.4f}")
st.write(f"Degrees of Freedom: {df_diff}")
st.write(f"p-value: {p_value:.6f}")

# ======================================================
# 6️⃣ MODEL FIT EVALUATION
# ======================================================

st.header("4️⃣ Model Fit Evaluation")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Log-Likelihood", round(model.llf,2))
col2.metric("AIC", round(model.aic,2))
col3.metric("BIC", round(model.bic,2))
col4.metric("Deviance", round(model.deviance,2))
col5.metric("Pearson χ²", round(model.pearson_chi2,2))

# ======================================================
# 7️⃣ MODEL EQUATION
# ======================================================

st.subheader("Cloglog Regression Equation")

params = model.params

equation = f"\\log(-\\log(1-p)) = {round(params['Intercept'],4)}"

for name in params.index:

    if name == "Intercept":
        continue

    coef = round(params[name],4)
    sign = "+" if coef >= 0 else "-"

    equation += f" {sign} {abs(coef)} \\cdot {name}"

st.latex(equation)

# ======================================================
# 8️⃣ INTERPRETATION
# ======================================================

st.header("5️⃣ Interpretation of Coefficients")

st.markdown(
"""
For the complementary log–log model

\[
\log(-\log(1-p)) = X\\beta
\]

it is often more convenient to interpret coefficients in terms of **how they change the probability raised to a power**.

If a predictor increases by one unit, the **new estimated probability equals the old probability raised to the power \(e^{\\beta}\)**.
"""
)

st.subheader("Variable Labels (Optional)")

labels = {}

for term in model.params.index:

    if term == "Intercept":
        continue

    labels[term] = st.text_input(
        f"Label for {term}",
        value=term,
        key=f"label_{term}"
    )

st.subheader("Interpretation Statements")

for term in model.params.index:

    coef = model.params[term]
    pval = model.pvalues[term]

    if term == "Intercept":

        st.markdown(
        f"""
        **Intercept**

        Baseline complementary log–log value when predictors are zero or at their reference levels.

        Coefficient = **{coef:.4f}**, p-value = **{pval:.4f}**
        """
        )

    else:

        label = labels[term]

        power_effect = np.exp(coef)

        st.markdown(
        f"""
        **{label}**

        If **{label} increases by one unit**, the new estimated probability of the event equals  
        the **old probability raised to the power**

        \[
        e^{{{coef:.4f}}} = {power_effect:.4f}
        \]

        Coefficient = **{coef:.4f}**, p-value = **{pval:.4f}**
        """
        )

# ======================================================
# 9️⃣ PARAGRAPH INTERPRETATION
# ======================================================

st.header("6️⃣ Model Interpretation Paragraph")

if st.button("Generate Interpretation Paragraph"):

    text = []

    text.append(
    "It is often more convenient to interpret the estimated regression coefficients "
    "in terms of how they change the probability of the event."
    )

    for term in model.params.index:

        if term == "Intercept":
            continue

        coef = model.params[term]
        label = labels[term]
        power = np.exp(coef)

        text.append(
        f"If {label} increases by one unit, the new estimated probability "
        f"is the old probability raised to the power exp({coef:.4f}) = {power:.4f}."
        )

    st.success(" ".join(text))

# ======================================================
# 🔟 ROC CURVE
# ======================================================

st.header("7️⃣ ROC Curve")

y_true = df_model[response]
y_pred = model.predict(df_model)

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

fig = px.line(
    x=fpr,
    y=tpr,
    labels=dict(x="False Positive Rate", y="True Positive Rate"),
    title=f"ROC Curve (AUC = {roc_auc:.3f})"
)

fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))

st.plotly_chart(fig)
```

if **name** == "**main**":
run()
