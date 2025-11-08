# ==========================================================
# multiple_regression_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite (CLI Version)
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats


# ==========================================================
# Helper: Input and Data Parsing
# ==========================================================
def get_user_data():
    """Collect multiple regression data from the user."""
    print("\n📘 Enter your data for Multiple Regression")
    print("-----------------------------------------------------------")
    print("Format: Each row corresponds to one observation.")
    print("Example for 3 predictors: y,x1,x2,x3 (comma-separated per row)")
    print("Enter all rows separated by semicolons (;)")
    print("-----------------------------------------------------------")
    raw_input_str = input("Enter your data: ").strip()

    try:
        data_rows = raw_input_str.split(';')
        data_matrix = [list(map(float, row.strip().split(','))) for row in data_rows]
        data_array = np.array(data_matrix)

        if data_array.shape[1] < 2:
            raise ValueError("Each row must include at least one predictor variable (x).")

        y = data_array[:, 0]
        X = data_array[:, 1:]
        print(f"\n✅ Data successfully entered: {len(y)} observations, {X.shape[1]} predictors.")
        return X, y

    except Exception as e:
        print(f"❌ Error reading data: {e}")
        return None, None


# ==========================================================
# Regression Summary
# ==========================================================
def multiple_regression_summary(X, y):
    """Compute multiple regression and print formatted summary."""
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    print("\n📊 MULTIPLE REGRESSION SUMMARY")
    print("-----------------------------------------------------------")
    print(model.summary())

    print("-----------------------------------------------------------")
    print("📈 MODEL FIT STATISTICS")
    print(f"R-squared (R²): {model.rsquared:.4f}")
    print(f"Adjusted R²   : {model.rsquared_adj:.4f}")
    print(f"F-statistic   : {model.fvalue:.4f}")
    print(f"Prob (F-stat) : {model.f_pvalue:.4e}")
    print("-----------------------------------------------------------")

    # Coefficient table
    coef_df = model.summary2().tables[1]
    print(coef_df.to_string(float_format=lambda x: f"{x:8.4f}"))

    # Residual stats
    print("\n📉 RESIDUAL ANALYSIS")
    print(f"Mean of residuals: {np.mean(model.resid):.4f}")
    print(f"Std of residuals : {np.std(model.resid):.4f}")
    print("-----------------------------------------------------------")

    return model


# ==========================================================
# Prediction
# ==========================================================
def predict_value(model, X_columns):
    """Predict new y value given user input for X variables."""
    print(f"\n🔮 Prediction Mode")
    print("-----------------------------------------------------------")
    print(f"Enter new values for the {X_columns} independent variables to predict y:")
    variable_labels = [f"x{i+1}" for i in range(X_columns)]
    print(f"Variable order: {', '.join(variable_labels)}")
    new_input = input(f"Enter {X_columns} values (comma-separated): ")

    try:
        new_x = np.array([float(v) for v in new_input.strip().split(',')])
        if len(new_x) != X_columns:
            print("❌ Error: Number of values does not match number of predictors.")
            return
        pred_y = model.predict([1, *new_x])[0]
        print(f"\n✅ Predicted y = {pred_y:.4f}")
    except Exception as e:
        print(f"❌ Error in prediction: {e}")


# ==========================================================
# Residual Plot
# ==========================================================
def residual_plot(model, X, y):
    """Generate residual scatter plot and optional histogram."""
    residuals = model.resid
    fitted_vals = model.fittedvalues

    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_vals, residuals, color='purple', edgecolor='k', s=60)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title("Residual Plot (Residuals vs Fitted Values)")
    plt.xlabel("Fitted Values (ŷ)")
    plt.ylabel("Residuals (y - ŷ)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Histogram
    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=10, color='lightblue', edgecolor='black')
    plt.title("Histogram of Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# ==========================================================
# Main Menu Controller
# ==========================================================
def run_multiple_regression_tool():
    """Main menu loop for the CLI-based Multiple Regression tool."""
    print("==========================================================")
    print("👨‍🏫 MIND: Multiple Regression Analysis Tool (CLI Version)")
    print("Professor Edward Pineda-Castro, Los Angeles City College")
    print("==========================================================")

    X = y = model = None

    while True:
        print("\nMenu Options:")
        print("1️⃣  Enter Data")
        print("2️⃣  Show Regression Summary")
        print("3️⃣  Predict New Value")
        print("4️⃣  Plot Residuals & Histogram")
        print("5️⃣  Exit Program")
        print("----------------------------------------------------------")

        choice = input("Enter your choice (1–5): ").strip()

        if choice == "1":
            X, y = get_user_data()
            if X is not None:
                model = multiple_regression_summary(X, y)

        elif choice == "2":
            if model is None:
                print("⚠️ Please enter data first using Option 1.")
            else:
                multiple_regression_summary(X, y)

        elif choice == "3":
            if model is None:
                print("⚠️ Please enter data first using Option 1.")
            else:
                predict_value(model, X.shape[1])

        elif choice == "4":
            if model is None:
                print("⚠️ Please enter data first using Option 1.")
            else:
                residual_plot(model, X, y)

        elif choice == "5":
            print("\n👋 Exiting program. Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please select a number between 1 and 5.")


# ==========================================================
# Run Script
# ==========================================================
if __name__ == "__main__":
    run_multiple_regression_tool()
