import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import shapiro, boxcox_normmax, chi2


# ======================================================
# Box–Cox Helper Functions
# ======================================================

def recommend_lambda(lambda_mle):
    if -2.5 <= lambda_mle < -1.5:
        return -2.0
    elif -1.5 <= lambda_mle < -0.75:
        return -1.0
    elif -0.75 <= lambda_mle < -0.25:
        return -0.5
    elif -0.25 <= lambda_mle < 0.25:
        return 0.0
    elif 0.25 <= lambda_mle < 0.75:
        return 0.5
    elif 0.75 <= lambda_mle < 1.5:
        return 1.0
    elif 1.5 <= lambda_mle <= 2.5:
        return 2.0
    else:
        return lambda_mle


def transformation_info(lam):
    lam_rounded = round(lam, 4)

    transformations = {
        -2.0: {"name": "Inverse Square",
               "formula": r"\tilde{y} = \frac{1}{2}\left(1 - \frac{1}{y^2}\right)"},
        -1.0: {"name": "Inverse (Reciprocal)",
               "formula": r"\tilde{y} = 1 - \frac{1}{y}"},
        -0.5: {"name": "Inverse Square Root",
               "formula": r"\tilde{y} = 2\left(1 - \frac{1}{\sqrt{y}}\right)"},
        0.0: {"name": "Natural Log",
              "formula": r"\tilde{y} = \ln(y)"},
        0.5: {"name": "Square Root",
              "formula": r"\tilde{y} = 2(\sqrt{y} - 1)"},
        1.0: {"name": "Linear",
              "formula": r"\tilde{y} = y - 1"},
        2.0: {"name": "Square",
              "formula": r"\tilde{y} = \frac{1}{2}(y^2 - 1)"}
    }

    return transformations.get(
        lam_rounded,
        {"name": "Custom λ",
         "formula": r"\tilde{y} = \frac{y^{\lambda}-1}{\lambda}"}
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

    y_original = pd.to_numeric(df[response], errors="coerce").dropna()

    if len(y_original) >= 3:
        _, p_orig = shapiro(y_original)
        st.write(f"Shapiro-Wilk p-value (original Y): {p_orig:.4f}")

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
    # ======================================================
    # 2️⃣ Box–Cox Transformation
    # ======================================================

    st.header("2️⃣ Box–Cox Transformation (Optional)")

    st.latex(r"""
    \tilde{y} =
    \begin{cases}
    \dfrac{y^{\lambda}-1}{\lambda}, & \lambda \neq 0 \\
    \ln(y), & \lambda = 0
    \end{cases}
    """)

    transformed = False
    chosen_lambda = None
    df_model = df.copy()

    y_clean = pd.to_numeric(df[response], errors="coerce").dropna()
    y_clean = y_clean[np.isfinite(y_clean)]

    can_boxcox = True

    if not np.issubdtype(y_clean.dtype, np.number):
        st.warning("Response must be numeric for Box–Cox.")
        can_boxcox = False

    if y_clean.nunique() < 2:
        st.warning("Response has no variation. Box–Cox skipped.")
        can_boxcox = False

    if (y_clean <= 0).any():
        st.warning("Box–Cox requires strictly positive values. Skipped.")
        can_boxcox = False

    if can_boxcox:
        try:
            lambda_mle = boxcox_normmax(y_clean, brack=(-3, 3))
        except Exception:
            st.warning("Box–Cox optimization failed for this dataset.")
            can_boxcox = False

    if can_boxcox:

        st.write(f"MLE λ = {lambda_mle:.4f}")

        recommended_lambdas = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
        rounded_lambda = recommended_lambdas[
            np.argmin(np.abs(recommended_lambdas - lambda_mle))
        ]

        st.write(f"Recommended rounded λ = {rounded_lambda}")

        use_exact = st.checkbox("Use exact MLE λ instead of rounded")

        if st.checkbox("Apply Box–Cox Transformation"):

            transformed = True
            chosen_lambda = lambda_mle if use_exact else rounded_lambda
            y = pd.to_numeric(df[response], errors="coerce")
            transformed_response = response + "_tr"

            if np.isclose(chosen_lambda, 0):
                df_model[transformed_response] = np.log(y)
            else:
                df_model[transformed_response] = (
                    y**chosen_lambda - 1
                ) / chosen_lambda

            y_tr_clean = df_model[transformed_response].dropna()

            if len(y_tr_clean) >= 3:
                stat_tr, p_tr = shapiro(y_tr_clean)
                st.write(f"Shapiro-Wilk p-value (transformed Y): {p_tr:.4f}")

                if p_tr > 0.05:
                    st.success("Transformed response appears normally distributed.")
                else:
                    st.warning("Transformed response does NOT appear normally distributed.")
            else:
                st.warning("Not enough data for Shapiro-Wilk test.")

            col1, col2 = st.columns(2)

            with col1:
                fig_y = px.histogram(
                    df,
                    x=response,
                    title="Original Y Distribution"
                )
                st.plotly_chart(fig_y)

            with col2:
                fig_ytr = px.histogram(
                    df_model,
                    x=transformed_response,
                    title="Transformed Y Distribution"
                )
                st.plotly_chart(fig_ytr)

            formula_transformed = (
                transformed_response + " ~ " + " + ".join(terms)
            )

    # ======================================================
    # 3️⃣ Model Fitting
    # ======================================================

    st.header("3️⃣ Model Fitting")

    model_original = smf.ols(
        formula=formula_original,
        data=df
    ).fit()

    st.subheader("Original Model Summary")
    st.text(model_original.summary())

    model = model_original
    active_response = response

    if transformed:

        model_transformed = smf.glm(
            formula=formula_transformed,
            data=df_model,
            family=sm.families.Gaussian()
        ).fit()

        st.subheader("Transformed Model Summary")
        st.text(model_transformed.summary())

        null_model = smf.glm(
            formula=transformed_response + " ~ 1",
            data=df_model,
            family=sm.families.Gaussian()
        ).fit()

        deviance = -2 * (null_model.llf - model_transformed.llf)
        df_diff = model_transformed.df_model
        p_value = 1 - chi2.cdf(deviance, df_diff)

        st.subheader("Deviance Test vs Null")
        st.write(f"Deviance: {deviance:.4f}")
        st.write(f"df: {df_diff}")
        st.write(f"p-value: {p_value:.4f}")

        model = model_transformed
        active_response = transformed_response

    # ======================================================
    # 4️⃣ Coefficient Interpretation
    # ======================================================

    st.header("4️⃣ Coefficient Interpretation")

    for name in model.params.index:
        coef = round(model.params[name], 4)
        pval = model.pvalues[name]

        st.markdown(
            f"**{name} (β = {coef})**: "
            f"{'Significant.' if pval < 0.05 else 'Not significant.'}"
        ) 
        

    # ======================================================
    # 5️⃣ Assumption Checks
    # ======================================================

    st.header("5️⃣ Assumption Checks")

    residuals = model.resid
    fitted = model.fittedvalues

    fig_resid = px.scatter(
        x=fitted,
        y=residuals,
        labels={'x': 'Fitted', 'y': 'Residuals'},
        title="Residuals vs Fitted"
    )
    fig_resid.add_hline(y=0)
    st.plotly_chart(fig_resid)

    # ======================================================
    # 6️⃣ Prediction
    # ======================================================

    st.header("6️⃣ Prediction")

    input_dict = {}

    for var in predictors:
        input_dict[var] = st.number_input(
            var,
            value=float(pd.to_numeric(df[var], errors="coerce").mean())
        )

    if st.button("Predict"):

        new_df = pd.DataFrame([input_dict])
        prediction_tr = model.predict(new_df)[0]

        if transformed:
            if chosen_lambda == 0:
                prediction = np.exp(prediction_tr)
            else:
                prediction = (chosen_lambda * prediction_tr + 1)**(1 / chosen_lambda)

            st.success(f"Predicted {response} (original scale): {prediction:.4f}")
        else:
            st.success(f"Predicted {response}: {prediction_tr:.4f}")

    # ======================================================
    # 7️⃣ Predicted vs Actual
    # ======================================================

    st.header("7️⃣ Predicted vs Actual")

    predicted_vals = model.predict(df_model)

    fig2 = px.scatter(
        x=predicted_vals,
        y=df_model[active_response],
        labels={'x': 'Predicted', 'y': 'Actual'},
        title="Predicted vs Actual"
    )

    st.plotly_chart(fig2)

  
if __name__ == "__main__":
    run()
