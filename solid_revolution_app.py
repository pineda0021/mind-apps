import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sympy import symbols, pi, integrate, sqrt, latex, simplify

# --- Function Parser ---
def parse_function(expr):
    return lambda x: eval(expr, {"x": x, "np": np})

# --- Plotting the function f(x) and f(x)^2 ---
def plot_function_and_square():
    x_vals = np.linspace(0, 1, 200)
    f_top = np.sqrt(x_vals)  # f(x) = sqrt(x)
    f_top_squared = f_top**2  # f(x)^2

    # Plot f(x) and f(x)^2
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_vals, f_top, label=r"$f(x) = \sqrt{x}$", color="blue", linewidth=2)
    ax.plot(x_vals, f_top_squared, label=r"$f(x)^2$", color="red", linestyle="--", linewidth=2)

    ax.set_xlabel("x")
    ax.set_ylabel("f(x), f(x)^2")
    ax.legend()
    st.pyplot(fig)

# --- Display the integral formula ---
def display_integral_formula():
    st.markdown("### 📘 Volume Formula using the Disk Method")
    st.latex(r"V = \pi \int_0^1 \left( \sqrt{x} \right)^2 \, dx = \pi \int_0^1 x \, dx")

# --- Compute exact volume ---
def compute_exact_volume():
    # Use sympy to handle the symbolic math
    x = symbols('x')
    volume_integral = pi * integrate(x, (x, 0, 1))  # V = pi * integral of x from 0 to 1
    volume_result = volume_integral.evalf()  # Compute the volume numerically
    return volume_result, volume_integral

# --- Display step-by-step solution ---
def step_by_step_solution():
    st.markdown("### 📋 Step-by-Step Solution:")

    # Step 1: Setup
    st.markdown("#### 🧮 Step 1: Set up the integral using the chosen method")
    st.latex(r"V = \pi \int_0^1 \left( \sqrt{x} \right)^2 \, dx = \pi \int_0^1 x \, dx")

    # Step 2: Evaluate
    st.markdown("#### ✅ Step 2: Perform the integration and compute the result")
    volume_result, volume_integral = compute_exact_volume()
    st.latex(f"= {latex(volume_integral)}")
    st.markdown(f"**Exact Volume:** ${latex(volume_integral)} \\approx {volume_result:.4f}$")

# --- Main Function to Display Everything ---
def main():
    st.set_page_config(layout="wide")
    st.title("🧠 MIND: Solid Revolution - Volume by Disk Method")
    st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

    plot_function_and_square()
    display_integral_formula()
    step_by_step_solution()

# --- Run the App ---
main()
