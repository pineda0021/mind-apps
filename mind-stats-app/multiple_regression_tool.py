# ==========================================================
# multiple_regression_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ==========================================================
# Helper Functions
# ==========================================================
def round_value(value, decimals=4):
    try:
        return round(float(value), decimals)
    except Exception:
        return value


def step_box(text):
    """Stylized blue step display box."""
    st.markdown(
        f"""
        <div style="background-color:#f0f6ff;padding:10px;border-radius:10px;
        border-left:5px solid #007acc;margin-bottom:10px;">
        <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


def print_regression_summary(model, decimals):
    """Display model summary and fit stats."""
    st.subheader("📄 Regression Summary")
    st.text(model.summary())

    st.markdown("### 📈 Model Fit Statistics")
    st.write(f"**R²:** {round_value(model.rsquared, decimals)}")
    st.write(f"**Adjusted R²:** {round_value(model.rsquared_adj, decimals)}")
    st.write(f"**F-statistic:** {round_value(model.fvalue, decimals)}")
    st.write(f"**p-value (F-test):** {round_value(model.f_pvalue, decimals)}")

    st.markdown("### 📄 Coefficients Table")
    coef_table = model.summary2().tables[1]
    st.dataframe(np.round(coef_table, decimals))


def plot_residuals(model):
    """Residuals diagnostics."""
    residuals = model.resid
    fitted = model.fittedvalues

    st.markdown("### 📉 Residual Diagnostics")

    fig1, ax1 = plt.subplots()
    ax1.scatter(fitted, residuals, color="#007acc", edgecolor="black")
    ax1.axhline(y=0, color="gray", linestyle="--")
    ax1.set_xlabel("Fitted Values (ŷ)")
    ax1.set_ylabel("Residuals (y - ŷ)")
    ax1.set_title("Residuals vs. Fitted Values")
    ax1.grid(True)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.hist(residuals, bins=10, color="#72bcd4", edgecolor="black")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Histogram of Residuals")
    st.pyplot(fig2)


# ==========================================================
# Main App
# ==========================================================
def run():
    st.header("👨‍🏫 Multiple Regression Analysis")

    st.markdown("""
    This tool estimates the best-fitting line  
    \\[
    \\hat{y} = b_0 + b_1x_1 + b_2x_2 + \\dots + b_kx_k
    \\]
    using the **Ordinary Least Squares (OLS)** method.
    """)

    st.subheader("📘 Data Input")
    uploaded_file = st.file_uploader("📂 Upload CSV or Excel file (optional)", type=["csv", "xlsx"])

    st.markdown("### ✏️ Or enter data manually (guided mode)")
    st.caption("""
    **Step 1:** Enter your dependent variable (y) values separated by commas.  
    Example: `10, 15, 20, 25`  

    **Step 2:** Choose how many independent variables (x’s) you want to include.  
    You’ll then be prompted to enter values for each \(x_i\).
    """)

    # ======================================================
    # Guided Manual Input Mode
    # ======================================================
    y_input = st.text_area("Dependent variable (y):", placeholder="10, 15, 20, 25", height=80)
    n_x = st.number_input("How many independent variables (x’s)?", min_value=1, max_value=10, value=1, step=1)

    x_inputs = []
    for i in range(n_x):
        x_val = st.text_area(
            f"Independent variable x{i+1} values (comma-separated):",
            placeholder=f"{', '.join(str(v) for v in range(1, 5))}",
            height=80,
            key=f"x_input_{i}"
        )
        x_inputs.append(x_val)

    decimals = st.number_input("Decimal places for rounding", min_value=1, max_value=10, value=4, step=1)

    # ======================================================
    # CASE 1: Uploaded Data
    # ======================================================
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("📊 Data Preview:")
            st.dataframe(df.head())

            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) < 2:
                st.error("File must contain at least two numeric columns (y and predictors).")
                return

            st.markdown("### 🧩 Variable Selection")
            y_col = st.selectbox("Select dependent variable (y):", num_cols, index=0)
            X_cols = st.multiselect("Select independent variable(s) (x):", [c for c in num_cols if c != y_col])

            if st.button("👨‍💻 Run Regression (Uploaded Data)"):
                if not X_cols:
                    st.error("Please select at least one independent variable.")
                    return

                data = df.dropna(subset=[y_col] + X_cols)
                y = data[y_col].to_numpy(dtype=float)
                X = data[X_cols].to_numpy(dtype=float)
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()

                step_box("**Step 1:** Fit the multiple regression model")
                st.latex(r"y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_kx_k + \varepsilon")

                step_box("**Step 2:** Review Model Summary and Fit Statistics")
                print_regression_summary(model, decimals)

                step_box("**Step 3:** Examine Residuals")
                plot_residuals(model)

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

    # ======================================================
    # CASE 2: Manual Input Guided Mode
    # ======================================================
    elif (y_input.strip() and any(x.strip() for x in x_inputs)) and st.button("👨‍💻 Run Regression (Manual Data)"):
        try:
            # Parse y
            y = np.array(list(map(float, y_input.replace(",", " ").split())))

            # Parse X for each entered predictor
            X_data = []
            for i, x_str in enumerate(x_inputs):
                if not x_str.strip():
                    st.error(f"Please enter values for x{i+1}.")
                    return
                x_vals = np.array(list(map(float, x_str.replace(",", " ").split())))
                if len(x_vals) != len(y):
                    st.error(f"x{i+1} must have the same number of observations as y ({len(y)}).")
                    return
                X_data.append(x_vals)

            X = np.column_stack(X_data)
            X = sm.add_constant(X)
            k = X.shape[1] - 1
            st.success(f"✅ Data entered successfully: {len(y)} observations and {k} predictor(s).")

            model = sm.OLS(y, X).fit()

            step_box("**Step 1:** Fit the multiple regression model")
            st.latex(r"y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_kx_k + \varepsilon")

            step_box("**Step 2:** Review Model Summary and Fit Statistics")
            print_regression_summary(model, decimals)

            step_box("**Step 3:** Examine Residuals")
            plot_residuals(model)

            step_box("**Step 4:** Predict New Values")
            new_x_input = st.text_input(f"Enter {k} predictor value(s) (x₁, …, xₖ):", placeholder=", ".join(str(i+1) for i in range(k)))
            if new_x_input.strip():
                try:
                    new_x = np.array(list(map(float, new_x_input.replace(",", " ").split())))
                    if len(new_x) != k:
                        st.error(f"Please enter {k} predictor values.")
                    else:
                        y_hat = model.predict([np.insert(new_x, 0, 1)])[0]
                        st.success(f"Predicted y = **{round_value(y_hat, decimals)}**")
                except Exception as e:
                    st.error(f"Error: {e}")

        except Exception as e:
            st.error(f"❌ Error: {e}")

    else:
        st.info("👆 Upload a dataset or enter y and x values manually to begin.")


# ==========================================================
# Run Script
# ==========================================================
if __name__ == "__main__":
    run()

# ✅ Compatibility alias for integration with MIND Suite
run_multiple_regression_tool = run

