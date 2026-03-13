import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import shapiro, chi2, norm


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
    # VARIABLE SELECTION (WITH REFERENCES)
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

    # ======================================================
    # LADDER-OF-POWERS λ SELECTION
    # ======================================================

    st.header("2️⃣ Ladder-of-Powers Transformation")

    if not pd.api.types.is_numeric_dtype(df[response]):
        st.error("Response must be numeric.")
        return

    if (df[response] <= 0).any():
        st.error("Transformation requires strictly positive response.")
        return

    ladder_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    lam = st.selectbox(
        "Select λ (Ladder of Powers)",
        ladder_values,
        index=1
    )

    # ======================================================
    # DISPLAY TRANSFORMATION + INVERSE FORMULAS
    # ======================================================

    st.subheader("Transformation Formula")

    if lam == -2.0:
        st.latex(r"\tilde{y} = \frac{1}{2}\left(1 - \frac{1}{y^2}\right)")
        st.latex(r"y = \left(\frac{1}{1 - 2\tilde{y}}\right)^{1/2}")

    elif lam == -1.0:
        st.latex(r"\tilde{y} = 1 - \frac{1}{y}")
        st.latex(r"y = \frac{1}{1 - \tilde{y}}")

    elif lam == -0.5:
        st.latex(r"\tilde{y} = 2\left(1 - \frac{1}{\sqrt{y}}\right)")
        st.latex(r"y = \left(\frac{1}{1 - \tilde{y}/2}\right)^2")

    elif lam == 0.0:
        st.latex(r"\tilde{y} = \ln(y)")
        st.latex(r"y = e^{\tilde{y}}")

    elif lam == 0.5:
        st.latex(r"\tilde{y} = 2(\sqrt{y} - 1)")
        st.latex(r"y = \left(\frac{\tilde{y}}{2} + 1\right)^2")

    elif lam == 1.0:
        st.latex(r"\tilde{y} = y - 1")
        st.latex(r"y = \tilde{y} + 1")

    elif lam == 2.0:
        st.latex(r"\tilde{y} = \frac{1}{2}(y^2 - 1)")
        st.latex(r"y = \sqrt{2\tilde{y} + 1}")

    # ======================================================
    # APPLY TRANSFORMATION
    # ======================================================

    df_model = df.copy()
    y = pd.to_numeric(df_model[response], errors="coerce")
    transformed_response = response + "_tr"

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

    # ======================================================
    # SIDE-BY-SIDE HISTOGRAM COMPARISON
    # ======================================================

    st.header("3️⃣ Distribution Comparison")

    col1, col2 = st.columns(2)

    with col1:
        fig_orig = px.histogram(df, x=response, nbins=15,
                                title="Original Response",
                                marginal="box")
        st.plotly_chart(fig_orig)

    with col2:
        fig_trans = px.histogram(df_model, x=transformed_response,
                                 nbins=15,
                                 title="Transformed Response",
                                 marginal="box")
        st.plotly_chart(fig_trans)

    # ======================================================
    # NORMALITY COMPARISON
    # ======================================================

    st.header("4️⃣ Normality Comparison")

    y_orig = df[response].dropna()
    y_trans = df_model[transformed_response].dropna()

    stat_orig, p_orig = shapiro(y_orig)
    stat_trans, p_trans = shapiro(y_trans)

    st.write(f"Original p-value: {p_orig:.4f}")
    st.write(f"Transformed p-value: {p_trans:.4f}")

    if p_trans > p_orig:
        st.success("Transformation improves normality.")
    else:
        st.warning("Transformation does NOT improve normality.")

    # ======================================================
    # FIT GAUSSIAN GLM
    # ======================================================

    st.header("5️⃣ Fit Gaussian GLM (Transformed Scale)")

    df_fit = df_model[[transformed_response] + predictors].dropna()

    formula_transformed = transformed_response + " ~ " + " + ".join(terms)

    model = smf.glm(
        formula=formula_transformed,
        data=df_fit,
        family=sm.families.Gaussian(link=sm.families.links.identity())
    ).fit()

    st.text(model.summary())

    # ======================================================
    # LR DEVIANCE (MATCH R)
    # ======================================================

    null_model = smf.glm(
        transformed_response + " ~ 1",
        data=df_fit,
        family=sm.families.Gaussian()
    ).fit()

    lr_deviance = -2 * (null_model.llf - model.llf)

    st.subheader("Likelihood Ratio Deviance")
    st.write(f"LR Deviance: {lr_deviance:.6f}")

    # ======================================================
    # PREDICTION
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

        if lam == -1.0:
            prediction = 1 / (1 - prediction_tr)
        elif lam == 0.0:
            prediction = np.exp(prediction_tr)
        elif lam == 0.5:
            prediction = (prediction_tr/2 + 1)**2
        elif lam == 1.0:
            prediction = prediction_tr + 1
        elif lam == 2.0:
            prediction = np.sqrt(2*prediction_tr + 1)
        else:
            prediction = (lam * prediction_tr + 1)**(1/lam)

        st.success(f"Predicted {response}: {prediction:.6f}")


if __name__ == "__main__":
    run()
