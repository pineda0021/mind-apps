# ====================================================
# 📊 MIND: Multiple Regression (Advanced)
# Updated for Universal Readability (Dark & Light Mode Safe)
# Professor Edward Pineda-Castro, Los Angeles City College
# ====================================================

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from io import StringIO

# ====================================================
# UNIVERSAL READABILITY COLORS
# ====================================================
BACKGROUND = "#2B2B2B"
TEXT = "white"
ACCENT = "#4da3ff"

plt.rcParams.update({
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.edgecolor": TEXT,
})

# ====================================================
# Step Box
# ====================================================
def step_box(text):
    st.markdown(
        f"""
        <div style="
            background-color:{BACKGROUND};
            padding:12px;
            border-radius:10px;
            border-left:6px solid {ACCENT};
            margin-bottom:12px;">
            <p style="color:{TEXT};margin:0;font-weight:bold;">{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ====================================================
# MAIN FUNCTION
# ====================================================
def run():

    # ---------- Header ----------
    st.markdown(
        f"<h1 style='color:{TEXT};'>🧠 MIND: Multiple Regression (Advanced)</h1>",
        unsafe_allow_html=True,
    )

    step_box("Use this module to explore bivariate and multivariate regression using Ordinary Least Squares (OLS).")

    st.latex(r"\hat{y} = b_0 + b_1x_1 + b_2x_2 + \dots + b_kx_k")

    # ====================================================
    # Upload Section
    # ====================================================
    st.markdown(f"<h3 style='color:{TEXT};'>📂 Upload CSV or Excel File (optional)</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx"])

    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            st.success("✅ Data successfully loaded!")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return

    # ====================================================
    # Manual Paste Section
    # ====================================================
    st.markdown(f"<h3 style='color:{TEXT};'>✍️ Enter or Paste Your Data</h3>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <p style='color:{TEXT};'>
        Paste your dataset below (tab, space, or comma separated).  
        The <b>first row</b> must contain column names.
        </p>

        <pre style='color:{TEXT};background-color:{BACKGROUND};padding:10px;border-radius:10px;'>
son  father  mother  siblings
70   70      65      5
73   70      66.5    10
71   65.5    63      3
        </pre>
        """,
        unsafe_allow_html=True,
    )

    data_text = st.text_area(
        "Paste your dataset here:",
        height=250,
        placeholder="son father mother siblings\n70 70 65 5\n73 70 66.5 10\n..."
    )

    # Parse pasted data
    if data_text.strip():
        try:
            clean_text = data_text.replace("\t", ",")
            clean_text = "\n".join([",".join(line.split()) for line in clean_text.splitlines()])
            df = pd.read_csv(StringIO(clean_text))
            st.success("✅ Data successfully parsed!")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Error parsing pasted data: {e}")
            return

    # No data yet
    if df is None:
        st.info("👆 Upload a file or paste your data above to begin.")
        return

    # ====================================================
    # Variable Selection
    # ====================================================
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.error("❌ The dataset must have at least two numeric columns.")
        return

    st.markdown(f"<h3 style='color:{TEXT};'>🎛 Variable Selection</h3>", unsafe_allow_html=True)

    y_col = st.selectbox("Select dependent variable (y)", num_cols)
    x_cols = st.multiselect(
        "Select one or more independent variables (x₁, x₂, …)",
        [c for c in num_cols if c != y_col],
    )

    if not x_cols:
        st.warning("Please select at least one independent variable.")
        return

    y = df[y_col]
    X = sm.add_constant(df[x_cols])

    # ====================================================
    # Regression Computation
    # ====================================================
    try:
        model = sm.OLS(y, X).fit()

        st.markdown(f"<h3 style='color:{TEXT};'>📄 Regression Summary</h3>", unsafe_allow_html=True)
        st.text(model.summary())

        # -----------------------------------------------
        # Model Fit Summary
        # -----------------------------------------------
        st.markdown(f"<h3 style='color:{TEXT};'>📊 Model Fit Summary</h3>", unsafe_allow_html=True)

        st.write(f"**R²:** {round(model.rsquared, 4)}")
        st.write(f"**Adjusted R²:** {round(model.rsquared_adj, 4)}")

        # -----------------------------------------------
        # Residual Plot
        # -----------------------------------------------
        st.markdown(f"<h3 style='color:{TEXT};'>📉 Residual Plot (Residuals vs Fitted Values)</h3>", unsafe_allow_html=True)

        residuals = model.resid
        fitted = model.fittedvalues

        fig, ax = plt.subplots(facecolor=BACKGROUND)
        fig.patch.set_facecolor(BACKGROUND)

        ax.scatter(fitted, residuals, color=ACCENT)
        ax.axhline(y=0, color="red", linestyle="--")
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot", color=TEXT)
        ax.grid(True, color="#555555")

        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Regression Error: {e}")

# ====================================================
# End of Module
# ====================================================

