# ======================================================
# 4️⃣ BOX-COX TRANSFORMATION (AUTO IF NEEDED)
# ======================================================

st.header("3️⃣ Box-Cox Transformation (If Needed)")

transformed_response = None
lambda_rec = None
original_model = None
note = ""

if p <= 0.05:

    st.warning(
        "The response is not normally distributed. "
        "A Box-Cox transformation will be applied."
    )

    if (df[response] <= 0).any():
        st.error("Box-Cox requires strictly positive response values.")
    else:
        y_original = df[response].dropna()

        # R-style grid: seq(-3, 3, 1/4)
        lambdas = np.arange(-3, 3.25, 0.25)
        llf_vals = [boxcox(y_original, lmbda=l)[1] for l in lambdas]

        lambda_hat = lambdas[np.argmax(llf_vals)]

        st.write(f"Estimated λ (Grid Search): **{lambda_hat:.4f}**")

        # Snap to recommended λ
        if -2.5 <= lambda_hat < -1.5:
            lambda_rec = -2
            trans_name = "Inverse Square"
            formula_tex = r"\tilde{y} = \frac{1}{2}\left(1 - \frac{1}{y^2}\right)"
            note = "Strong compression of large values for extreme right skew."

        elif -1.5 <= lambda_hat < -0.75:
            lambda_rec = -1
            trans_name = "Reciprocal"
            formula_tex = r"\tilde{y} = 1 - \frac{1}{y}"
            note = "Common for rates; reduces influence of large values."

        elif -0.75 <= lambda_hat < -0.25:
            lambda_rec = -0.5
            trans_name = "Inverse Square Root"
            formula_tex = r"\tilde{y} = 2\left(1 - \frac{1}{\sqrt{y}}\right)"
            note = "Moderately reduces right skewness."

        elif -0.25 <= lambda_hat < 0.25:
            lambda_rec = 0
            trans_name = "Log Transformation"
            formula_tex = r"\tilde{y} = \ln(y)"
            note = "Most common transformation; stabilizes variance."

        elif 0.25 <= lambda_hat < 0.75:
            lambda_rec = 0.5
            trans_name = "Square Root"
            formula_tex = r"\tilde{y} = 2(\sqrt{y} - 1)"
            note = "Useful for count data and moderate skew."

        elif 0.75 <= lambda_hat < 1.5:
            lambda_rec = 1
            trans_name = "Linear"
            formula_tex = r"\tilde{y} = y - 1"
            note = "No transformation needed."

        else:
            lambda_rec = 2
            trans_name = "Square"
            formula_tex = r"\tilde{y} = \frac{1}{2}(y^2 - 1)"
            note = "Used for left-skewed data."

        st.write(f"**Recommended λ:** {lambda_rec}")
        st.write(f"**Transformation:** {trans_name}")
        st.latex(formula_tex)
        st.info(f"Teaching Note: {note}")

        # Apply transformation
        y_transformed = boxcox(y_original, lmbda=lambda_rec)
        transformed_response = f"{response}_boxcox"
        df[transformed_response] = y_transformed

        # Skewness comparison
        st.subheader("Skewness Comparison")

        skew_before = skew(y_original)
        skew_after = skew(y_transformed)

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(px.histogram(
                x=y_original,
                title=f"Original (Skew={skew_before:.3f})"
            ))

        with col2:
            st.plotly_chart(px.histogram(
                x=y_transformed,
                title=f"Transformed (Skew={skew_after:.3f})"
            ))

        if abs(skew_after) < abs(skew_before):
            st.success("Skewness reduced after transformation.")

        # Re-check normality
        stat_bc, p_bc = shapiro(y_transformed)
        st.write(f"Post-Transformation p-value: {p_bc:.4f}")

        # Fit original model for comparison
        original_formula = response_original + " ~ " + " + ".join(predictors)
        original_model = smf.ols(original_formula, data=df).fit()

        response = transformed_response
