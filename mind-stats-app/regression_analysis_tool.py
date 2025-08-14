# regression_analysis_tool.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

def get_user_data():
    y_input = input("Enter your dependent variable (y) values, separated by commas: ").replace(',', ' ')
    x_input = input("Enter your independent variable (x) values, separated by commas: ").replace(',', ' ')
    y = np.array(list(map(float, y_input.strip().split())))
    x = np.array(list(map(float, x_input.strip().split())))
    if len(y) != len(x):
        raise ValueError("x and y must have the same number of values.")
    return x, y

def scatter_plot(x, y):
    plt.scatter(x, y, c='blue', marker='o')
    for i in range(len(x)):
        plt.text(x[i], y[i], f"({x[i]}, {y[i]})", fontsize=9, ha='right')
    plt.title("Scatter Plot: X vs Y")
    plt.xlabel("Independent Variable (x)")
    plt.ylabel("Dependent Variable (y)")
    plt.grid(True)
    plt.show()

def correlation_analysis(x, y):
    corr_coef, p_value = stats.pearsonr(x, y)
    print("\nCorrelation Analysis")
    print("--------------------")
    print(f"Correlation Coefficient: {round(corr_coef, 4)}")
    print(f"P-value: {round(p_value, 4)}")

def linear_regression_summary(x, y):
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    print("\nLinear Regression Summary")
    print("--------------------------")
    print(model.summary())

def predict_value_and_residual(x, y):
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    
    try:
        new_x = float(input("Enter a value of x to predict y: "))
        pred_y = model.predict([1, new_x])[0]
        print(f"Predicted y for x = {new_x}: {round(pred_y, 4)}")

        actual_y_input = input("If you have the actual y value, enter it to compute residual (or press Enter to skip): ")
        if actual_y_input.strip():
            actual_y = float(actual_y_input)
            residual = actual_y - pred_y
            print(f"Residual = Actual y - Predicted y = {round(residual, 4)}")
    except Exception as e:
        print(f"Error: {e}")

def residual_plot(x, y):
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    residuals = model.resid
    plt.scatter(x, residuals, color='purple')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title("Residual Plot")
    plt.xlabel("Independent Variable (x)")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.show()

def run_regression_tool():
    print("Welcome to the Simple Linear Regression Tool\n")
    x = y = None

    while True:
        print("\nMenu:")
        print("1. Enter Data (Required First)")
        print("2. Scatter Plot")
        print("3. Correlation Analysis")
        print("4. Linear Regression Summary")
        print("5. Predict Value & Compute Residual")
        print("6. Residual Plot")
        print("7. Exit")

        choice = input("Enter your choice (1-7): ")

        if choice == "1":
            try:
                x, y = get_user_data()
                print("Data successfully entered.")
            except Exception as e:
                print(f"Error: {e}")
                x = y = None

        elif choice in ["2", "3", "4", "5", "6"]:
            if x is None or y is None:
                print("Please enter data first using Option 1.")
            else:
                if choice == "2":
                    scatter_plot(x, y)
                elif choice == "3":
                    correlation_analysis(x, y)
                elif choice == "4":
                    linear_regression_summary(x, y)
                elif choice == "5":
                    predict_value_and_residual(x, y)
                elif choice == "6":
                    residual_plot(x, y)

        elif choice == "7":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 7.")

# For standalone run
if __name__ == "__main__":
    run_regression_tool()
