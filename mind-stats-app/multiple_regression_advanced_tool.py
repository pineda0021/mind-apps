# ====================================================
# 📊 MIND: Multiple Regression (Advanced)
# Professor Edward Pineda-Castro, Los Angeles City College
# ====================================================

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from io import StringIO

# ====================================================
# MAIN FUNCTION
# ====================================================
def run():
    st.header("👨‍🏫 Multiple Regression (Advanced)")

    # ---------- Intro Section ----------
    st.markdown("""
    This advanced module allows students to:

    - Select which variable is the dependent variable (y)
    - Select one or more independent variables (x₁, x₂, …)
    - Run both **bivariate** and **multiple** regression
    - Perform **model comparison (nested F-test)**

    The model is estimated using **Ordinary Least Squares (OLS):**
    """)
    st.latex(r"\hat{y} = b_0 + b_1x_1 + b_2x_2 + \cdots + b_kx_k")

    # ---------- Example Dataset Button ----------
    st.markdown("### 📥 Optional: Load Example Dataset (Galton Family Data)")
    if st.button("Load Galton Family Data"):
        galton_data = '''son	father	mother	siblings
70	70	65	5
73	70	66.5	10
71	65.5	63	3
72	71	65.5	5
79	70	65	6
69.7	67	65	5
63	65	66	3
70.5	71.7	65.5	0
64	68	60	6
72	66	65.5	4
64	62	66	2
68.2	68.5	63.5	0
70.7	70	64	6
70	67	65.5	3
68	71	65	2
66	66	66	4
68.2	68.5	63.5	0
73.2	71	67	3
73	68.5	67	2
70.7	70.3	62.7	6
71	68	63	7
69.7	69.5	64.5	6
73	69	66.5	6
74	70.5	64	7
73	70	65	2
71	67.5	65	10
66	68	65.5	1
66	66	66	4
70	70	65	5
73	69.5	62	2
68.2	68.5	63.5	0
70.5	71.7	65.5	0
70.5	66	67	7
75	69	68.5	9
72	69.5	62	10
70	66.5	62.5	6
73	69	63	5
69.2	68.5	65	7
65	66	60	0
65	67	65	2
72.7	67	66.2	4
70.7	70.3	62.7	6
70	68	63	3
70	70	65	5
75	69	66.7	3
71	70	69	3
64	62	66	2
73	69	66.5	6
71.7	69.2	64	3
73.2	72.7	69	7
68	68	59	9
70	70	67	4
72	66	65.5	4
68.7	66.5	65	7
73	69	63	5
73	67	66	7
73	71.5	65.5	1
73	68.5	67	2
71	75	64	1
73	70	66.5	10
70	66	61.5	0
70	65	63	1
73	71.5	65.5	1
71	67.5	65	10
70	66	61.5	0
74	74	62	7
72	66	66	5
72	69.5	62	10
71	71	63	8
70.5	69	63.5	2
72	68.5	65	7
70	65	63	1
70.5	69	63.5	2
68	71	65	2
71	71	63	3
72	69.5	62	10
70	69	65	6
71	65.5	63	3
'''
        st.session_state["pasted_data"] = galton_data
        st.success("✅ Galton Family Data Loaded! Scroll down to select variables.")

    # ---------- File Upload Section ----------
    st.markdown("### 📂 Upload CSV or Excel File (optional)")
    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx"])

    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("✅ Data successfully loaded!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return

    # ---------- Manual Data Entry Section ----------
    st.markdown("### ✍️ Enter or Paste Your Data")
    st.markdown("""
    Paste your data directly below (tab, space, or comma separated).  
    The **first row** should contain column names.
    """)

    default_text = st.session_state.get("pasted_data", "")
    data_text = st.text_area(
        "Paste your dataset here:",
        height=250,
        value=default_text,
        placeholder="son father mother siblings\n70 70 65 5\n73 70 66.5 10\n..."
    )

    if data_text.strip():
        try:
            clean_text = data_text.replace("\t", ",")
            clean_text = "\n".join(
                [",".join(line.split()) for line in clean_text.splitlines()]
            )
            df = pd.read_csv(StringIO(clean_text))
            st.success("✅ Data successfully parsed!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"⚠️ Error parsing pasted data: {e}")
            return

    if df is None:
        st.info("👆 Upload a file, paste data, or load the Galton dataset to begin.")
        return

    # ---------- Variable Selection ----------
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.error("❌ The dataset must have at least two numeric columns.")
        return

    y_col = st.selectbox("Select dependent variable (y)", num_cols)
    x_cols = st.multiselect(
        "Select one or more independent variables (x₁, x₂, …)",
        [c for c in num_cols if c != y_col],
    )

    if not y_col or not x_cols:
        st.warning("Please select both y and at least one x variable.")
        return

    y = df[y_col]
    X = sm.add_constant(df[x_cols])

    # ---------- Regression Computation ----------
    try:
        model = sm.OLS(y, X).fit()
        st.subheader("📄 Regression Summary")
        st.text(model.summary())

        # ---------- Model Fit Summary ----------
        st.subheader("📊 Model Fit Summary")
        st.write(f"**R²:** {round(model.rsquared, 4)}")
        st.write(f"**Adjusted R²:** {round(model.rsquared_adj, 4)}")

        # ---------- Residual Plot ----------
        st.subheader("📉 Residual Plot (Residuals vs Fitted Values)")
        residuals = model.resid
        fitted = model.fittedvalues
        fig, ax = plt.subplots()
        ax.scatter(fitted, residuals)
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        ax.grid(True)
        st.pyplot(fig)

        # ---------- Optional Model Comparison ----------
        st.markdown("### 🔍 Optional: Compare with Simpler Model")
        smaller_x = st.text_input("Enter smaller model x’s (comma-separated):")
        if smaller_x.strip():
            try:
                smaller_vars = [x.strip() for x in smaller_x.split(",")]
                X_small = sm.add_constant(df[smaller_vars])
                model_small = sm.OLS(y, X_small).fit()
                f_test = model.compare_f_test(model_small)

                st.markdown("**F-Test for Model Comparison (Nested Models)**")
                st.write(f"F = {round(f_test[0], 4)}, p = {round(f_test[1], 4)}")

                if f_test[1] <= 0.05:
                    st.success("✅ The larger model significantly improves the fit (reject H₀).")
                else:
                    st.info("❌ The larger model does not significantly improve the fit (fail to reject H₀).")
            except Exception as e:
                st.error(f"⚠️ Could not run model comparison: {e}")

    except Exception as e:
        st.error(f"⚠️ Regression Error: {e}")
