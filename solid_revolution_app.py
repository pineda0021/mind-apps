import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad

# --- Page config ---
st.set_page_config(page_title="MIND: Solid Revolution Tool", layout="wide")
st.title("🧠 MIND: Solid of Revolution Tool")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

# --- Sidebar inputs ---
st.sidebar.header("🔧 Parameters")

top_expr = st.sidebar.text_input("Top function f(x):", value="x")
bottom_expr = st.sidebar.text_input("Bottom function g(x):", value="x**2")
method = st.sidebar.selectbox("Method:", ["Disk/Washer", "Cylindrical Shell"])
axis = st.sidebar.selectbox("Axis of rotation:", ["x-axis", "y-axis"])

# Students can enter custom intervals a, b
a = st.sidebar.number_input("Start of interval (a):", value=0.0)
b = st.sidebar.number_input("End of interval (b):", value=1.0)

compute = st.sidebar.button("🔄 Compute and Visualize")

# --- Function parser ---
def parse_function(expr):
    return lambda x: eval(expr, {"x": x, "np": np})

# --- Compute exact volume ---
def compute_exact_volume(top_expr, bottom_expr, method, axis, a, b):
    f_top = parse_function(top_expr)
    f_bot = parse_function(bottom_expr)

    if method == "Disk/Washer":
        if axis == "x-axis":
            integrand = lambda x: np.pi * (f_top(x)**2 - f_bot(x)**2)
        else:
            integrand = lambda y: np.pi * (f_top(y)**2 - f_bot(y)**2)
    else:
        if axis == "x-axis":
            integrand = lambda x: 2 * np.pi * x * (f_top(x) - f_bot(x))
        else:
            integrand = lambda y: 2 * np.pi * y * (f_top(y) - f_bot(y))

    # Integrating using quad from scipy
    volume, error = quad(integrand, a, b)
    return volume

# --- Display formula ---
def show_formula(method, axis, f_expr, g_expr):
    st.markdown("### 📘 Setup and Formula")
    st.latex(f"f(x) = {f_expr}")
    st.latex(f"g(x) = {g_expr}")
    if method == "Disk/Washer":
        if axis == "x-axis":
            st.latex(r"V = \pi \int_a^b \left[f(x)^2 - g(x)^2\right] \, dx")
        else:
            st.latex(r"V = \pi \int_c^d \left[f(y)^2 - g(y)^2\right] \, dy")
    else:
        if axis == "x-axis":
            st.latex(r"V = 2\pi \int_a^b y(f(y) - g(y)) \, dy")
        else:
            st.latex(r"V = 2\pi \int_a^b x(f(x) - g(x)) \, dx")

# --- 3D plot function ---
def plot_solid(top_expr, bottom_expr, method, axis):
    f_top = parse_function(top_expr)
    f_bot = parse_function(bottom_expr)
    x = np.linspace(0, 1, 200)
    theta = np.linspace(0, 2 * np.pi, 100)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if method == "Disk/Washer" and axis == "x-axis":
        X, T = np.meshgrid(x, theta)
        Y_top = f_top(X) * np.cos(T)
        Z_top = f_top(X) * np.sin(T)
        ax.plot_surface(X, Y_top, Z_top, alpha=0.7, cmap="coolwarm", edgecolor="none")
        Y_bot = f_bot(X) * np.cos(T)
        Z_bot = f_bot(X) * np.sin(T)
        ax.plot_surface(X, Y_bot, Z_bot, alpha=1, color="white", edgecolor="none")

    elif method == "Disk/Washer" and axis == "y-axis":
        y = np.linspace(0, 1, 200)
        T, Y = np.meshgrid(theta, y)
        R_out = parse_function(top_expr)(Y)
        R_in = parse_function(bottom_expr)(Y)
        X_outer = R_out * np.cos(T)
        Z_outer = R_out * np.sin(T)
        ax.plot_surface(X_outer, Y, Z_outer, alpha=0.7, cmap="viridis", edgecolor="none")
        X_inner = R_in * np.cos(T)
        Z_inner = R_in * np.sin(T)
        ax.plot_surface(X_inner, Y, Z_inner, alpha=1, color="white", edgecolor="none")

    elif method == "Cylindrical Shell" and axis == "y-axis":
        X, T = np.meshgrid(x, theta)
        Height = f_top(X) - f_bot(X)
        R = X
        Y_shell = R * np.cos(T)
        Z_shell = R * np.sin(T)
        ax.plot_surface(Y_shell, Height, Z_shell, alpha=0.8, cmap="plasma", edgecolor="none")

    elif method == "Cylindrical Shell" and axis == "x-axis":
        y = np.linspace(0, 1, 200)
        T, Y = np.meshgrid(theta, y)
        Height = parse_function(top_expr)(Y) - parse_function(bottom_expr)(Y)
        R = Y
        X_shell = R * np.cos(T)
        Z_shell = R * np.sin(T)
        ax.plot_surface(X_shell, R, Z_shell, alpha=0.8, cmap="plasma", edgecolor="none")

    ax.set_title(f"{method} about the {axis}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(30, 45)
    st.pyplot(fig)

# --- Recommendation box ---
def show_method_tip(method, axis):
    st.markdown("### ✅ Which Method is Better?")
    if method == "Disk/Washer":
        if axis == "x-axis":
            st.success("Great choice! Since your functions are in terms of x, the Disk/Washer method about the x-axis is simple and direct.")
        else:
            st.warning("Careful! Using Disk/Washer about the y-axis may require solving for x as a function of y.")
    else:
        if axis == "y-axis":
            st.success("Perfect! Cylindrical Shells work very well with vertical rectangles and rotation about the y-axis.")
        else:
            st.warning("Cylindrical Shells about the x-axis may be more complex if you can't easily express x as a function of y.")

# --- Main Display Logic ---
if compute:
    col_left, col_right = st.columns([1.2, 0.8])

    with col_left:
        show_formula(method, axis, top_expr, bottom_expr)
        plot_solid(top_expr, bottom_expr, method, axis)
        
        # Compute exact volume and display result
        exact_volume = compute_exact_volume(top_expr, bottom_expr, method, axis, a, b)
        st.markdown(f"### ✅ Exact Volume: {exact_volume:.4f}")

    with col_right:
        show_method_tip(method, axis)
        with st.expander("🪞 What does this visualization mean?", expanded=True):
            st.info(
                "This 3D plot shows a solid formed by revolving the region between two curves around a specific axis.\n\n"
                "- **Disk/Washer Method**: cuts horizontal/vertical slices perpendicular to the axis.\n"
                "- **Cylindrical Shell Method**: wraps vertical slices around the axis.\n\n"
                "This helps visualize how integrals compute volume — just like Riemann sums approximate area!"
            )

