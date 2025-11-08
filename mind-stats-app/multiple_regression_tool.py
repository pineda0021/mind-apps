# ==========================================================
# multiple_regression_tool.py
# Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite (CLI Edition)
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ==========================================================
# Helper — Get User Data
# ==========================================================
def get_user_data():
    """Collect multiple regression data from user with clear guidance."""
    print("\n📘 HOW TO ENTER YOUR DATA FOR MULTIPLE REGRESSION")
    print("-----------------------------------------------------------")
    print("👉 Each row represents ONE observation (student, case, or trial).")
    print("👉 Separate each value in a row with commas (,).")
    print("👉 Separate each row with semicolons (;).")
    print("-----------------------------------------------------------")
    print("Example: Predicting students’ final exam scores (y)")
    print("from hours studied (x₁) and quiz average (x₂):")
    print("\n    85,10,90; 78,8,85; 92,12,95; 70,5,75")
    print("\nHere:")
    print(" - 85,78,92,70 → Exam scores (y)")
    print(" - 10,8,12,5   → Hours studied (x₁)")
    print(" - 90,85,95,75 → Quiz averages (x₂)")
    print("-----------------------------------------------------------")
    print("📏 Order of columns must be:  y, x₁, x₂, x₃, ...")
    print("-----------------------------------------------------------")

    raw_input_str = input("✏️  Enter your data below: ").strip()

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
    """Compute regression, display summary, and print key statistics."""
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

    # Coefficients table (β, SE, t, p)
    coef_table = model.summary2().tables[1]
    print("📄 COEFFICIENT ESTIMATES")
    print(coef_table.to_string(float_format=lambda x: f"{x:8.4f}"))

    # Residual stats
    print("\n📉 RESIDUAL ANALYSIS")
    print(f"Mean of residuals: {np.mean(model.resid):.4f}")
    print(f"Std. deviation    : {np.std(model.resid):.4f}")
    print("-----------------------------------------------------------")

    return model


# ==========================================================
# Prediction
# ==========================================================
def predict_value(model, X_columns):
    """Predict new y value from user-supplied x values."""
    print("\n🔮 PREDICTION MODE")
    print("-----------------------------------------------------------")
    var_labels = [f"x{i+1}" for i in range(X_columns)]
    print(f"Variable order: {', '.join(var_labels)}")
    new_input = input(f"Enter {X_columns} values (comma-separated): ")

    try:
        new_x = np.array([float(v) for v in new_input.strip().split(',')])
        if len(new_x) != X_columns:
            print("❌ Number of values does not match number of predictors.")
            return
        pred_y = model.predict([1, *new_x])[0]
        print(f"\n✅ Predicted y = {pred_y:.4f}")
    except Exception as e:
        print(f"❌ Error: {e}")


# ==========================================================
# Residual Plot
# ==========================================================
def residual_plot(model, X, y):
    """Plot residual scatter and histogram."""
    residuals = model.resid
    fitted_vals = model.fittedvalues

    # Scatterplot
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_vals, residuals, color='purple', edgecolor='k', s=60)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title("Residuals vs Fitted Values")
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
    """Interactive CLI for Multiple Regression Analysis."""
    print("==========================================================")
    print("👨‍🏫 MIND: Multiple Regression Analysis Tool (CLI Edition)")
    print("Professor Edward Pineda-Castro — Los Angeles City College")
    print("==========================================================")

    X = y = model = None

    while True:
        print("\nMENU OPTIONS")
        print("1️⃣  Enter Data")
        print("2️⃣  Show Regression Summary")
        print("3️⃣  Predict New Value")
        print("4️⃣  Plot Residuals & Histogram")
        print("5️⃣  Exit Program")
        print("-----------------------------------------------------------")

        choice = input("Enter your choice (1–5): ").strip()

        if choice == "1":
            X, y = get_user_data()
            if X is not None:
                model = multiple_regression_summary(X, y)

        elif choice == "2":
            if model is None:
                print("⚠️ Please enter data first (Option 1).")
            else:
                multiple_regression_summary(X, y)

        elif choice == "3":
            if model is None:
                print("⚠️ Please enter data first (Option 1).")
            else:
                predict_value(model, X.shape[1])

        elif choice == "4":
            if model is None:
                print("⚠️ Please enter data first (Option 1).")
            else:
                residual_plot(model, X, y)

        elif choice == "5":
            print("\n👋 Exiting program. Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please select 1–5.")


# ==========================================================
# Run Script
# ==========================================================
if __name__ == "__main__":
    run_multiple_regression_tool()
