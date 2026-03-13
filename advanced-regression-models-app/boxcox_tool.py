import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2


def run():

    st.title("📘 Gaussian Linear Model with Ladder-of-Powers Transformation")

    # ======================================================
    # 1. DATA UPLOAD
    # ======================================================

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ======================================================
    # 2. VARIABLE SELECTION (WITH REFERENCES)
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
    # 3. LADDER-OF-POWERS TRANSFORMATION
    # ======================================================

    st.header("2️⃣ Ladder-of-Powers Transformation")

    if not pd.api.types.is_numeric_dtype(df[response]):
        st.error("Response must be numeric.")
        return

    if (df[response] <= 0).any():
        st.error("Transformation requires strictly positive response.")
        return

    ladder_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    lam = st.selectbox("Select λ", ladder_values, index=1)

    # Display transformation + inverse formulas

    st.subheader("Transformation Formula")

    if lam == -2.0:
        st.latex(r"\tilde{y} = \frac{1}{2}(1 - 1/y^2)")
        st.latex(r"y = (1/(1-2\tilde{y}))^{1/2}")

    elif lam == -1.0:
        st.latex(r"\tilde{y} = 1 - \frac{1}{y}")
        st.latex(r"y = \frac{1}{1-\tilde{y}}")

    elif lam == -0.5:
        st.latex(r"\tilde{y} = 2(1 - 1/\sqrt{y})")
        st.latex(r"y = (1/(1-\tilde{y}/2))^2")

    elif lam == 0.0:
        st.latex(r"\tilde{y} = \ln(y)")
        st.latex(r"y = e^{\tilde{y}}")

    elif lam == 0.5:
        st.latex(r"\tilde{y} = 2(\sqrt{y} - 1)")
        st.latex(r"y = (\tilde{y}/2 + 1)^2")

    elif lam == 1.0:
        st.latex(r"\tilde{y} = y - 1")
        st.latex(r"y = \tilde{y} + 1")

    elif lam == 2.0:
        st.latex(r"\tilde{y} = \frac{1}{2}(y^2 - 1)")
        st.latex(r"y = \sqrt{2\tilde{y}+1}")

    # Apply transformation

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

    y_trans = df_model[transformed_response].dropna()

    # ======================================================
    # 4. HISTOGRAM + NORMAL OVERLAY
    # ======================================================

    st.header("3️⃣ Histogram with Normal Overlay")

    fig, ax = plt.subplots(figsize=(8,6))

    ax.hist(y_trans, bins=9, density=True)

    mean_val = np.mean(y_trans)
    sd_val = np.std(y_trans)

    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean_val, sd_val)

    ax.plot(x, p, 'r')
    ax.set_title("Histogram of Transformed Response with Normal Curve")
    ax.set_xlabel("Transformed Response")

    st.pyplot(fig)

    # ======================================================
    # 5. SHAPIRO-WILK TEST
    # ======================================================

    st.header("4️⃣ Shapiro-Wilk Test")

    stat, p_value = stats.shapiro(y_trans)

    st.write(f"Shapiro-Wilk Statistic: {stat:.4f}")
    st.write(f"P-Value: {p_value:.4f}")

    if p_value > 0.05:
        st.success("Conclusion: Fail to reject H₀ → Data appears normally distributed.")
    else:
        st.warning("Conclusion: Reject H₀ → Data is NOT normally distributed.")

    # ======================================================
    # 6. FIT OLS MODEL
    # ======================================================

    st.header("5️⃣ Fit General Linear Model (OLS)")

    df_fit = df_model[[transformed_response] + predictors].dropna()

    formula_transformed = transformed_response + " ~ " + " + ".join(terms)

    model = smf.ols(formula=formula_transformed, data=df_fit).fit()

    st.text(model.summary())

    # ======================================================
    # 7. ERROR ESTIMATES
    # ======================================================

    st.subheader("Error Estimates")

    sigma_unbiased = np.sqrt(model.mse_resid)
    sigma_mle = np.sqrt(model.ssr / model.nobs)

    st.write(f"√MSE (Unbiased σ̂): {sigma_unbiased:.6f}")
    st.write(f"√(SSR/n) (MLE σ): {sigma_mle:.6f}")

    # ======================================================
    # 8. LR DEVIANCE (MATCH R)
    # ======================================================

    null_model = smf.ols(transformed_response + " ~ 1", data=df_fit).fit()

    lr_stat = 2 * (model.llf - null_model.llf)
    df_diff = int(model.df_model)
    p_lr = chi2.sf(lr_stat, df_diff)

    st.subheader("Likelihood Ratio Test")
    st.write(f"LR Statistic: {lr_stat:.6f}")
    st.write(f"Degrees of Freedom: {df_diff}")
    st.write(f"P-Value: {p_lr:.6f}")

    # ======================================================
    # 9. PREDICTION
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
            new_df[var] = pd.Categorical(new_df[var], categories=df[var].cat.categories)

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
