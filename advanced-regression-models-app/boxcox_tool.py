import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, chi2


def run():

    st.title("General Linear Regression Model Lab")

    # ======================================================
    # 1. DATA UPLOAD
    # ======================================================

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="glm_upload"
    )

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
    # 3. RESPONSE NORMALITY CHECK
    # ======================================================

    st.header("2️⃣ Response Normality Check")

    if not pd.api.types.is_numeric_dtype(df[response]):
        st.error("Response must be numeric.")
        return

    fig = px.histogram(
        df,
        x=response,
        title=f"Histogram of {response}",
        marginal="box"
    )
    st.plotly_chart(fig)

    y_clean = df[response].dropna()

    if len(y_clean) >= 3:
        qq_fig = sm.qqplot(y_clean, line='s')
        st.pyplot(qq_fig.figure)

        stat, p = shapiro(y_clean)

        st.write(f"Shapiro-Wilk Statistic: {stat:.4f}")
        st.write(f"p-value: {p:.4f}")

        if p > 0.05:
            st.success("Response appears normally distributed.")
        else:
            st.warning("Response does NOT appear normally distributed.")
    else:
        st.warning("Not enough data for Shapiro-Wilk test.")

    # ======================================================
    # 📘 BOX–COX TRANSFORMATION THEORY (TEXTBOOK FORMAT)
    # ======================================================

    st.header("📘 Box–Cox Power Transformation")

    st.markdown(
        """
If the response variable is right-skewed, a transformation may be applied 
to make its distribution more nearly normal.
The Box–Cox power transformation is defined as:
"""
    )

    st.latex(r"""
    \tilde{y} =
    \begin{cases}
    \dfrac{y^\lambda - 1}{\lambda}, & \lambda \neq 0 \\
    \ln y, & \lambda = 0
    \end{cases}
    """)

    st.markdown(
        """
The transformation is continuous in λ. By L'Hôpital's Rule:
"""
    )

    st.latex(r"""
    \lim_{\lambda \to 0}
    \frac{y^\lambda - 1}{\lambda}
    =
    \ln y
    """)

    st.subheader("Recommended Practical Transformations")

    st.latex(r"""
    \begin{array}{c|c|c|c}
    \textbf{Range of } \lambda & \textbf{Recommended } \lambda & \tilde{y} & \textbf{Transformation} \\
    \hline
    [-2.5,-1.5) & -2 & \frac{1}{2}\left(1-\frac{1}{y^2}\right) & \text{Inverse Square} \\
    [-1.5,-0.75) & -1 & 1-\frac{1}{y} & \text{Reciprocal} \\
    [-0.75,-0.25) & -0.5 & 2\left(1-\frac{1}{\sqrt{y}}\right) & \text{Inverse Square Root} \\
    [-0.25,0.25) & 0 & \ln y & \text{Natural Logarithm} \\
    [0.25,0.75) & 0.5 & 2(\sqrt{y}-1) & \text{Square Root} \\
    [0.75,1.5) & 1 & y-1 & \text{Linear} \\
    [1.5,2.5] & 2 & \frac{1}{2}(y^2-1) & \text{Square}
    \end{array}
    """)

    st.subheader("Fitted Model After Transformation")

    st.latex(r"""
    \mathbb{E}(\tilde{y}) =
    \hat{\beta}_0 +
    \hat{\beta}_1 x_1 +
    \cdots +
    \hat{\beta}_k x_k
    """)

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
    # 8. EQUATION BUILDER
    # ======================================================

    def build_equation(model, response):

        params = model.params
        equation = f"\\widehat{{\\mathbb{{E}}}}({response}) = {round(params['Intercept'],4)}"

        for name in params.index:
            if name == "Intercept":
                continue

            coef = round(params[name], 4)
            sign = "+" if coef >= 0 else "-"

            if name.startswith("C(") and "T." in name:
                var_name = name.split("[")[0]
                var_name = var_name.replace("C(", "").split(",")[0]
                level = name.split("T.")[1].rstrip("]")
                equation += f" {sign} {abs(coef)} D_{{{var_name}={level}}}"
            else:
                equation += f" {sign} {abs(coef)} \\cdot {name}"

        return equation

    st.subheader("Fitted Regression Equation (Full Model)")
    st.latex(build_equation(model, response))


if __name__ == "__main__":
    run()
