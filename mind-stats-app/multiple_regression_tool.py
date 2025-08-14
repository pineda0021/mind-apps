# multiple_regression_tool.py
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

def get_user_data():
    print("Enter your data for Multiple Regression:")
    print("Format: Each row corresponds to one observation.")
    print("Example for 3 predictors: y,x1,x2,x3 (comma-separated per row)")
    print("Enter all rows separated by semicolons.")
    raw_input_str = input("Enter your data: ")

    data_rows = raw_input_str.strip().split(';')
    data_matrix = [list(map(float, row.strip().split(','))) for row in data_rows]

    data_array = np.array(data_matrix)
    y = data_array[:, 0]
    X = data_array[:, 1:]
    return X, y

def multiple_regression_summary(X, y):
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    print("\nMultiple Regression Summary")
    print("----------------------------")
    print(model.summary())
    return model

def predict_value(model, X_columns):
    print(f"\nEnter new values for the {X_columns} independent variables to predict y (comma-separated):")
    new_input = input(f"x1,x2,...,x{X_columns}: ")
    new_x = np.array([float(v) for v in new_input.strip().split(',')])
    if len(new_x) != X_columns:
        print("Number of values does not match number of predictors.")
        return
    pred_y = model.predict([1, *new_x])[0]
    print(f"Predicted y: {round(pred_y, 4)}")

def residual_plot(model, X, y):
    residuals = model.resid
    plt.scatter(range(len(residuals)), residuals, color='purple')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title("Residual Plot")
    plt.xlabel("Observation Index")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.show()

def run_multiple_regression_tool():
    print("Welcome to the Multiple Regression Analysis Tool\n")
    X = y = None
    model = None

    while True:
        print("\nMenu:")
        print("1. Enter Data (Required First)")
        print("2. Multiple Regression Summary")
        print("3. Predict New Value")
        print("4. Residual Plot")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            try:
                X, y = get_user_data()
                print("Data successfully entered.")
                model = multiple_regression_summary(X, y)
            except Exception as e:
                print(f"Error: {e}")
                X = y = model = None

        elif choice == "2":
            if model is None:
                print("Please enter data first using Option 1.")
            else:
                multiple_regression_summary(X, y)

        elif choice == "3":
            if model is None:
                print("Please enter data first using Option 1.")
            else:
                predict_value(model, X.shape[1])

        elif choice == "4":
            if model is None:
                print("Please enter data first using Option 1.")
            else:
                residual_plot(model, X, y)

        elif choice == "5":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    run_multiple_regression_tool()
