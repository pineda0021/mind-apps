import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import shapiro, chi2


# ======================================================
# Transformation Helper
# ======================================================

def transformation_info(lam):

    lam_rounded = round(lam, 1)

    transformations = {

        -2.0: {"name": "Inverse Square",
               "formula": r"\tilde{y} = \frac{1}{2}\left(1-\frac{1}{y^2}\right)"},

        -1.0: {"name": "Inverse (Reciprocal)",
               "formula": r"\tilde{y} = 1 - \frac{1}{y}"},

        -0.5: {"name": "Inverse Square Root",
               "formula": r"\tilde{y} = 2\left(1-\frac{1}{\sqrt{y}}\right)"},

        0.0: {"name": "Natural Log",
              "formula": r"\tilde{y} = \ln(y)"},

        0.5: {"name": "Square Root",
              "formula": r"\tilde{y} = 2(\sqrt{y}-1)"},

        1.0: {"name": "Linear",
              "formula": r"\tilde{y} = y-1"},

        2.0: {"name": "Square",
              "formula": r"\tilde{y} = \frac{1}{2}(y^2-1)"}
    }

    return transformations.get(
        lam_rounded,
        {"name": "Custom λ",
         "formula": r"\tilde{y}=\frac{y^{\lambda}-1}{\lambda}"}
    )


# ======================================================
# APP
# ======================================================

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
    # 2️⃣ Box–Cox Transformation
    # ======================================================

    st.header("2️⃣ Box–Cox Transformation")

    st.latex(r"""
    \tilde{y} =
    \begin{cases}
    \dfrac{y^{\lambda}-1}{\lambda}, & \lambda \neq 0 \\
    \ln(y), & \lambda = 0
    \end{cases}
    """)

    df_model = df.copy()
    transformed = False
    transformed_response = None

    y_clean = pd.to_numeric(df[response], errors="coerce").dropna()

    if (y_clean <= 0).any():

        st.warning("Box–Cox requires strictly positive response values.")
        return

    chosen_lambda = st.number_input("Enter λ value", value=0.0, step=0.1)

    info = transformation_info(chosen_lambda)

    st.write(f"Selected transformation: **{info['name']}**")
    st.latex(info["formula"])

    if st.checkbox("Apply Transformation"):

        transformed = True

        y = pd.to_numeric(df[response], errors="coerce")
        transformed_response = response + "_tr"

        # Apply transformation
        if np.isclose(chosen_lambda, 0):
            df_model[transformed_response] = np.log(y)
        else:
            df_model[transformed_response] = (y**chosen_lambda - 1) / chosen_lambda

        y_trans = df_model[transformed_response].dropna()

        if len(y_trans) < 3:
            st.warning("Not enough data for Shapiro-Wilk test.")
            return

        # Q-Q plot
        qq_fig = sm.qqplot(y_trans, line='s')
        st.pyplot(qq_fig.figure)

        # Shapiro test
        stat, p = shapiro(y_trans)

        st.write(f"Shapiro-Wilk Statistic: {stat:.4f}")
        st.write(f"p-value: {p:.4f}")

        # ======================================================
        # 3️⃣ Model Fitting
        # ======================================================

        formula_transformed = transformed_response + " ~ " + " + ".join(terms)

        st.success("Response appears normally distributed.")
        st.header("3️⃣ Model Fitting Transformed")

        model = smf.ols(formula=formula_transformed, data=df_model).fit()
        null_model = smf.ols(f"{transformed_response} ~ 1", data=df_model).fit()

        st.subheader("Box-Cox Model Summary")
        st.text(model.summary())

        # ======================================================
        # Likelihood Ratio Test
        # ======================================================

        st.subheader("Likelihood Ratio (Deviance) Test")

        deviance = -2 * (null_model.llf - model.llf)
        df_diff = model.df_model
        p_value = 1 - chi2.cdf(deviance, df_diff)

        st.write(f"Deviance: {deviance:.4f}")
        st.write(f"Degrees of Freedom: {df_diff}")
        st.write(f"p-value: {p_value:.4f}")

    else:
        st.warning("Please apply a transformation before fitting the model.")
        return

    # ======================================================
    # 6️⃣ EQUATION BUILDER
    # ======================================================

    def build_equation(model, response):

        params = model.params
        equation = f"\\widehat{{\\mathbb{{E}}}}({response}) = {round(params['Intercept'],4)}"

        for name in params.index:

            if name == "Intercept":
                continue

            coef = round(params[name], 4)
            sign = "+" if coef >= 0 else "-"

            equation += f" {sign} {abs(coef)} {name}"

        return equation

    st.subheader("Fitted Regression Equation")
    st.latex(build_equation(model, response))

    # ======================================================
    # 7️⃣ Prediction
    # ======================================================

    st.header("7️⃣ Prediction")

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
            new_df[var] = pd.Categorical(new_df[var], categories=df[var].cat.categories)

        prediction_tr = model.predict(new_df)[0]

        # Back-transform prediction
        if np.isclose(chosen_lambda, 0):
            prediction = np.exp(prediction_tr)
        else:
            prediction = (chosen_lambda * prediction_tr + 1) ** (1 / chosen_lambda)

        st.success(f"Predicted {response}: {prediction:.4f}")

    # ======================================================
    # 8️⃣ Predicted vs Actual
    # ======================================================

    st.header("8️⃣ Predicted vs Actual")

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


# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    run()
