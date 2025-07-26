
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Page Configuration ---
st.set_page_config(layout="wide")
st.title("🧠 MIND: Solid Revolution")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

# --- Function Parser ---
def parse_function(expr):
    return lambda x: eval(expr, {"x": x, "np": np})

# --- Compute Volume (symbolically shown) ---
def compute_volume_display(top_expr, bottom_expr, method, axis):
    steps = []
    steps.append(f"Top function: $f(x) = {top_expr}$")
    steps.append(f"Bottom function: $g(x) = {bottom_expr}$")

    if method == "Disk/Washer":
        if axis == "x-axis":
            steps.append(r"Volume formula: $V = \pi \int_a^b \left[f(x)^2 - g(x)^2\right] \, dx$")
        else:
            steps.append(r"Volume formula: $V = \pi \int_c^d \left[f(y)^2 - g(y)^2\right] \, dy$")
    else:
        if axis == "x-axis":
            steps.append(r"Volume formula: $V = 2\pi \int_a^b (y)(f(y) - g(y)) \, dy$")
        else:
            steps.append(r"Volume formula: $V = 2\pi \int_a^b (x)(f(x) - g(x)) \, dx$")
    return steps

# --- 3D Plot Function ---
def plot_solid(top_expr, bottom_expr, method, axis):
    f_top = parse_function(top_expr)
    f_bot = parse_function(bottom_expr)

    x = np.linspace(0, 1, 200)
    theta = np.linspace(0, 2 * np.pi, 100)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if method == 'Disk/Washer' and axis == 'x-axis':
        X, T = np.meshgrid(x, theta)
        Y_top = f_top(X) * np.cos(T)
        Z_top = f_top(X) * np.sin(T)
        ax.plot_surface(X, Y_top, Z_top, alpha=0.7, cmap='coolwarm', edgecolor='none')
        Y_bot = f_bot(X) * np.cos(T)
        Z_bot = f_bot(X) * np.sin(T)
        ax.plot_surface(X, Y_bot, Z_bot, alpha=1, color='white', edgecolor='none')

    elif method == 'Disk/Washer' and axis == 'y-axis':
        y = np.linspace(0, 1, 200)
        T, Y = np.meshgrid(theta, y)
        R_out = parse_function(top_expr)(Y)
        R_in = parse_function(bottom_expr)(Y)
        X_outer = R_out * np.cos(T)
        Z_outer = R_out * np.sin(T)
        ax.plot_surface(X_outer, Y, Z_outer, alpha=0.7, cmap='viridis', edgecolor='none')
        X_inner = R_in * np.cos(T)
        Z_inner = R_in * np.sin(T)
        ax.plot_surface(X_inner, Y, Z_inner, alpha=1, color='white', edgecolor='none')

    elif method == 'Cylindrical Shell' and axis == 'y-axis':
        X, T = np.meshgrid(x, theta)
        Height = f_top(X) - f_bot(X)
        R = X
        Y_shell = R * np.cos(T)
        Z_shell = R * np.sin(T)
        ax.plot_surface(Y_shell, Height, Z_shell, alpha=0.8, cmap='plasma', edgecolor='none')

    elif method == 'Cylindrical Shell' and axis == 'x-axis':
        y = np.linspace(0, 1, 200)
        T, Y = np.meshgrid(theta, y)
        Height = parse_function(top_expr)(Y) - parse_function(bottom_expr)(Y)
        R = Y
        X_shell = R * np.cos(T)
        Z_shell = R * np.sin(T)
        ax.plot_surface(X_shell, R, Z_shell, alpha=0.8, cmap='plasma', edgecolor='none')

    ax.set_title(f'{method} about the {axis}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(30, 45)
    st.pyplot(fig)

# --- Streamlit UI ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔣 Enter Your Functions")
    top_expr = st.text_input("Top Function:", value="x")
    bottom_expr = st.text_input("Bottom Function:", value="x**2")
    method = st.selectbox("Method:", ["Disk/Washer", "Cylindrical Shell"])
    axis = st.selectbox("Axis of Rotation:", ["x-axis", "y-axis"])
    if st.button("Plot and Compute Volume"):
        try:
            steps = compute_volume_display(top_expr, bottom_expr, method, axis)
            st.markdown("### 📘 Formula Steps")
            for step in steps:
                st.latex(step)
            plot_solid(top_expr, bottom_expr, method, axis)
        except Exception as e:
            st.error(f"❌ Error: {e}")

with col2:
    st.markdown("### 🎯 Example Visual (GeoGebra Style)")
    st.image("47f80030-9dfb-4598-ac06-ee0b135d745d.png", caption="Solid of Revolution", use_column_width=True)
