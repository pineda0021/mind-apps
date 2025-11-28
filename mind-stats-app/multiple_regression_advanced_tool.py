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

    # ---------- Intro Section (Dark/Light Mode Safe) ----------
    st.markdown(
        """
        <div style="
            background-color:#2B2B2B;
            padding:18px;
            border-radius:10px;
            color:white;
            font-size:18px;
            line-height:1.6;
        ">
            Use this module to explore bivariate and multivariate regression 
            using <b>Ordinary Least Squares (OLS)</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.latex(r"\hat{y} = b_0 + b_1x_1 + b_2x_2 + \cdots + b_kx_k")

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
    st.markdown(
        """
        Paste your data directly below (tab, space, or comma separated).  
        The **first row must contain column names**.

        **Example:**
        ```
        son  father  mother  siblings
        70   70      65      5
        73   70      66.5    10
        71   65.5    63      3
        ```
        """
    )

    data_text = st.text_area(
        "Paste your dataset here:",
        height=250,
        placeholder="son father mother siblings\n70 70 65 5\n73 70 66.5 10\n..."
    )

    if data_text.strip():
        try:
            # Replace tabs with commas
            clean_text = data_text.replace("\t", ",")
            # Replace multiple spaces with commas
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
        st.info("👆 Upload a file or paste your data above to begin.")
        return

    # ---------- Variable Selection ----------
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.error("❌ The dataset must contain at least two numeric columns.")
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

        st.markdown("### 📄 Regression Summary")
        st.text(model.summary())

        # ---------- Model Fit Summary ----------
        st.markdown("### 📊 Model Fit Summary")
        st.write(f"**R²:** {round(model.rsquared, 4)}")
        st.write(f"**Adjusted R²:** {round(model.rsquared_adj, 4)}")

        # ---------- Residual Plot ----------
        st.markdown("### 📉 Residual Plot (Residuals vs Fitted Values)")
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

    except Exception as e:
        st.error(f"⚠️ Regression Error: {e}")


