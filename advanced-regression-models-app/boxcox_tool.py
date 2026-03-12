import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import shapiro

# ======================================================
# Transformation Helper
# ======================================================
def transformation_info(lam):
    lam_rounded = round(lam,1)
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
        {"name":"Custom λ",
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
    # MODEL SPECIFICATION
    # ======================================================
    st.header("1️⃣ Model Specification")

    response = st.selectbox("Select Response Variable", df.columns)
    predictors = st.multiselect(
        "Select Predictor Variables",
        [col for col in df.columns if col != response]
    )
    if not predictors:
        return

    categorical_vars = st.multiselect(
        "Select Categorical Predictors",
        predictors
    )

    terms = []
    for var in predictors:
        if var in categorical_vars:
            df[var] = df[var].astype("category")
            terms.append(f"C({var})")
        else:
            terms.append(var)

    formula_original = response + " ~ " + " + ".join(terms)
    st.code(formula_original)

    # ======================================================
    # BOX COX TRANSFORMATION
    # ======================================================
    st.header("2️⃣ Box-Cox / Ladder-of-Powers Transformation")
    st.latex(r"""
    \tilde{y} =
    \begin{cases}
    \frac{y^\lambda -1}{\lambda} & \lambda \neq 0 \\
    \ln(y) & \lambda =0
    \end{cases}
    """)

    y_clean = pd.to_numeric(df[response], errors="coerce").dropna()
    if (y_clean <= 0).any():
        st.warning("Response must be positive for Box-Cox.")
        return

    chosen_lambda = st.number_input("Enter λ", value=0.0, step=0.1)
    info = transformation_info(chosen_lambda)
    st.write(f"Transformation: **{info['name']}**")
    st.latex(info["formula"])

    transformed_response = response + "_tr"
    df_model = df.copy()
    y = pd.to_numeric(df[response], errors="coerce")

    # ======================================================
    # APPLY TRANSFORMATION
    # ======================================================
    lam = round(chosen_lambda,1)
    if lam == -2.0:
        df_model[transformed_response] = 0.5*(1 - 1/(y**2))
    elif lam == -1.0:
        df_model[transformed_response] = 1 - 1/y
    elif lam == -0.5:
        df_model[transformed_response] = 2*(1 - 1/np.sqrt(y))
    elif lam == 0.0:
        df_model[transformed_response] = np.log(y)
    elif lam == 0.5:
        df_model[transformed_response] = 2*(np.sqrt(y)-1)
    elif lam == 1.0:
        df_model[transformed_response] = y-1
    elif lam == 2.0:
        df_model[transformed_response] = 0.5*(y**2 -1)
    else:
        df_model[transformed_response] = (y**lam -1)/lam

    y_trans = df_model[transformed_response].dropna()

    # ======================================================
    # NORMALITY TEST
    # ======================================================
    st.subheader("Normality Test (Shapiro-Wilk)")
    qq = sm.qqplot(y_trans, line="s")
    st.pyplot(qq.figure)
    stat,p = shapiro(y_trans)
    st.write("Shapiro Statistic:", round(stat,4))
    st.write("p-value:", round(p,4))

    # ======================================================
    # MODEL FITTING (GLM) using transformed response
    # ======================================================
    st.header("3️⃣ Model Fit (GLM on transformed response)")
    y_trans = transformed_response + " ~ " + " + ".join(terms)
    model_vars = [transformed_response] + predictors
    df_fit = df_model[model_vars].dropna()  # complete cases like R
    model = smf.glm(
        formula=formula_transformed,
        data=df_fit,
        family=sm.families.Gaussian()
    ).fit()
    st.text(model.summary())

    # ======================================================
    # R-STYLE DEVIANCE
    # ======================================================
    y_bar = df_fit[transformed_response].mean()
    null_deviance = np.sum((df_fit[transformed_response]-y_bar)**2)
    residual_deviance = np.sum(model.resid_response**2)
    st.subheader("Model Diagnostics")
    st.write("Null Deviance:", round(null_deviance,4))
    st.write("Residual Deviance:", round(residual_deviance,4))
    st.write("AIC:", round(model.aic,4))

    # ======================================================
    # PREDICTION
    # ======================================================
    st.header("4️⃣ Prediction")
    input_dict = {}
    for var in predictors:
        if var in categorical_vars:
            input_dict[var] = st.selectbox(var, df[var].unique())
        else:
            input_dict[var] = st.number_input(var, value=float(df[var].mean()))

    if st.button("Predict"):
        new_df = pd.DataFrame([input_dict])
        prediction_tr = model.predict(new_df)[0]

        # back-transform prediction
        if lam == 0:
            prediction = np.exp(prediction_tr)
        else:
            prediction = (lam*prediction_tr +1)**(1/lam)
        st.success(f"Predicted {response}: {prediction:.4f}")

    # ======================================================
    # PREDICTED VS ACTUAL
    # ======================================================
    st.header("5️⃣ Predicted vs Actual")
    predicted = model.predict(df_fit)
    fig = px.scatter(
        x=predicted,
        y=df_fit[transformed_response],
        labels={"x":"Predicted","y":"Actual"},
        title="Predicted vs Actual"
    )
    st.plotly_chart(fig)

if __name__ == "__main__":
    run()
