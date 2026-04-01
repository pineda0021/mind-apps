    # ======================================================
    # 7️⃣ INTERPRETATION
    # ======================================================

    st.header("5️⃣ Interpretation")

    st.markdown("Interpretation uses $e^{\\beta}$.")

    for term in res.params.index:

        coef = res.params[term]
        pval = res.pvalues[term]
        exp_coef = np.exp(coef)

        if term.startswith("C("):
            var_name = term.split("[")[0].replace("C(", "").split(",")[0]
            level = term.split("T.")[-1].replace("]", "")
            label = f"{var_name}[{level}]"
        else:
            label = term

        st.subheader(label)
        st.latex(f"e^{{{coef:.4f}}} = {exp_coef:.4f}")

        if term == "Intercept":

            st.write(
                f"When all predictors are held at zero and all indicator variables are at their reference levels, "
                f"the estimated rate is {exp_coef:.4f}."
            )

        elif term.startswith("C("):

            var_name = term.split("[")[0].replace("C(", "").split(",")[0]
            level = term.split("T.")[-1].replace("]", "")
            ref = reference_dict.get(var_name, "reference")

            st.write(
                f"If {var_name} is an indicator variable, then the exponentiated estimated coefficient "
                f"$e^{{\\hat{{\\beta}}}} = {exp_coef:.4f}$ represents the ratio of the estimated rates for "
                f"{var_name} = {level} and {var_name} = {ref}, while the other predictors are held constant."
            )

            st.write(
                f"Equivalently, $e^{{\\hat{{\\beta}}}}\\cdot 100\\% = {exp_coef * 100:.2f}\\%$ represents "
                f"the estimated percent ratio of rates for {var_name} = {level} relative to {var_name} = {ref}."
            )

        else:

            percent_change = (exp_coef - 1) * 100

            st.write(
                f"If {label} is numeric, then the exponentiated estimated coefficient "
                f"$e^{{\\hat{{\\beta}}}} = {exp_coef:.4f}$ represents the estimated rate ratio corresponding "
                f"to a one-unit increase in {label}, while all the other predictors are held fixed."
            )

            st.write(
                f"Equivalently, $(e^{{\\hat{{\\beta}}}} - 1)\\cdot 100\\% = {percent_change:.2f}\\%$ may be interpreted "
                f"as the estimated percent change in rate when {label} increases by one unit."
            )

        st.write(f"Coefficient = {coef:.4f}")
        st.write(f"p-value = {pval:.4f}")

        if pval <= 0.05:
            st.success("Significant")
        else:
            st.warning("Not significant")
