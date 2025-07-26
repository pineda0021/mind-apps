import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sympy import symbols, integrate, pi, Rational, latex, simplify, sympify
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# --- Page config ---
st.set_page_config(page_title="MIND: Solid Revolution Tool", layout="wide")
st.title("🧠 MIND: Solid of Revolution Tool")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

# --- Sidebar inputs ---
st.sidebar.header("🔧 Parameters")

function_option = st.sidebar.selectbox("Do you have one function or two functions?", ["One Function", "Two Functions"])

if function_option == "One Function":
    top_expr = st.sidebar.text_input("Function f(x):", value="x**(1/2)")
    bottom_expr = None
    method = st.sidebar.selectbox("Method:", ["Disk/Washer", "Cylindrical Shell"])
    axis = "y-axis" if method == "Cylindrical Shell" else st.sidebar.selectbox("Axis of rotation:", ["x-axis", "y-axis"])
else:
    top_expr = st.sidebar.text_input("Top function f(x):", value="x")
    bottom_expr = st.sidebar.text_input("Bottom function g(x):", value="x**2")
    method = st.sidebar.selectbox("Method:", ["Disk/Washer", "Cylindrical Shell"])
    axis = st.sidebar.selectbox("Axis of rotation:", ["x-axis", "y-axis"])

show_3d = st.sidebar.checkbox("Show 3D Plot", value=True)
a = st.sidebar.number_input("Start of interval (a):", value=0.0)
b = st.sidebar.number_input("End of interval (b):", value=1.0)
compute = st.sidebar.button("🔄 Compute and Visualize")

# --- Function parser ---
def parse_function(expr):
    return lambda x: eval(expr, {"x": x, "np": np})

# --- Plot functions ---
def plot_functions(top_expr, bottom_expr=None):
    x_vals = np.linspace(a, b, 200)
    f_top = parse_function(top_expr)
    f_bot = parse_function(bottom_expr) if bottom_expr else None

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_vals, f_top(x_vals), label=f"Top: f(x) = {top_expr}", color="blue", linewidth=2)

    if f_bot:
        ax.plot(x_vals, f_bot(x_vals), label=f"Bottom: g(x) = {bottom_expr}", color="red", linewidth=2)
        ax.fill_between(x_vals, f_top(x_vals), f_bot(x_vals), color='gray', alpha=0.5)
    else:
        ax.fill_between(x_vals, f_top(x_vals), color='gray', alpha=0.3)

    ax.set_xlabel("x")
    ax.set_ylabel("f(x), g(x)")
    ax.legend()
    st.pyplot(fig)

# --- Static 3D Plot ---
def plot_3d_solid(top_expr, bottom_expr=None, method="Disk/Washer", axis="x-axis"):
    f_top = parse_function(top_expr)
    f_bot = parse_function(bottom_expr) if bottom_expr else (lambda x: 0)

    x_vals = np.linspace(a, b, 200)
    theta_vals = np.linspace(0, 2 * np.pi, 100)
    X, T = np.meshgrid(x_vals, theta_vals)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if method == "Disk/Washer" and axis == "x-axis":
        R_outer = f_top(X)
        R_inner = f_bot(X)
        Y_outer = R_outer * np.cos(T)
        Z_outer = R_outer * np.sin(T)
        Y_inner = R_inner * np.cos(T)
        Z_inner = R_inner * np.sin(T)

        ax.plot_surface(X, Y_outer, Z_outer, cmap=cm.coolwarm, alpha=0.8)
        if bottom_expr:
            ax.plot_surface(X, Y_inner, Z_inner, color='white', alpha=1)

    elif method == "Cylindrical Shell" and axis == "y-axis":
        R = f_top(X)
        Y = X
        X_shell = R * np.cos(T)
        Z_shell = R * np.sin(T)
        ax.plot_surface(X_shell, Y, Z_shell, cmap=cm.viridis, alpha=0.8)

    else:
        st.warning("3D plot only supported for specific method/axis combinations.")
        return

    ax.set_title(f"{method} about the {axis}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    st.pyplot(fig)

# --- Compute volume ---
def compute_exact_volume(top_expr, bottom_expr, method, axis, a, b):
    f_top = parse_function(top_expr)
    f_bot = parse_function(bottom_expr) if bottom_expr else None

    if method == "Disk/Washer":
        integrand = (lambda x: np.pi * (f_top(x)**2)) if not f_bot else (lambda x: np.pi * (f_top(x)**2 - f_bot(x)**2))
    else:
        integrand = lambda x: 2 * np.pi * x * f_top(x)

    volume, _ = quad(integrand, a, b)
    return volume

# --- Display formula ---
def show_formula(method, axis, f_expr, g_expr=None):
    st.markdown("### 📘 Setup and Formula")
    st.latex(f"f(x) = {f_expr}")
    if g_expr:
        st.latex(f"g(x) = {g_expr}")
    if method == "Disk/Washer":
        st.latex(r"V = \pi \int_a^b [f(x)^2 - g(x)^2] \,dx" if g_expr else r"V = \pi \int_a^b [f(x)^2] \,dx")
    else:
        st.latex(r"V = 2\pi \int_a^b x f(x)\,dx")

# --- Step-by-step ---
def step_by_step_solution(top_expr, bottom_expr, method, axis, a, b):
    st.markdown("### 📋 Step-by-Step Solution:")
    x = symbols('x')
    f_top = sympify(top_expr)
    f_bot = sympify(bottom_expr) if bottom_expr else None

    if method == "Disk/Washer":
        integrand = f_top**2 if not f_bot else f_top**2 - f_bot**2
        symbolic_integral = pi * integrate(integrand, (x, a, b))
        st.latex(f"V = \pi \int_{{{a}}}^{{{b}}} {latex(integrand)} \\, dx")
    else:
        integrand = x * f_top
        symbolic_integral = 2 * pi * integrate(integrand, (x, a, b))
        st.latex(f"V = 2\pi \int_{{{a}}}^{{{b}}} x \cdot {latex(f_top)} \\, dx")

    st.markdown("#### ✅ Step 2: Evaluate the integral")
    simplified_expr = simplify(symbolic_integral)
    if simplified_expr.has(pi):
        coeff = simplified_expr / pi
        fraction_result = Rational(coeff).limit_denominator()
        numer, denom = fraction_result.as_numer_denom()
        st.latex(f"= \\frac{{{numer}}}{{{denom}}} \\pi")
        st.markdown(f"**Exact Volume:** {numer}/{denom}π ≈ {float(symbolic_integral):.4f}")
    else:
        st.latex(f"= {latex(symbolic_integral)}")
        st.markdown(f"**Exact Volume:** {float(symbolic_integral):.4f}")

# --- Method tip ---
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

# --- Main run ---
if compute:
    col_left, col_right = st.columns([1.2, 0.8])
    with col_left:
        plot_functions(top_expr, bottom_expr)
        show_formula(method, axis, top_expr, bottom_expr)
        step_by_step_solution(top_expr, bottom_expr, method, axis, a, b)
        exact_volume = compute_exact_volume(top_expr, bottom_expr, method, axis, a, b)
        st.markdown(f"### ✅ Exact Volume: {exact_volume:.4f}")
        if show_3d:
            plot_3d_solid(top_expr, bottom_expr, method, axis)

    with col_right:
        show_method_tip(method, axis)
        with st.expander("🪞 What does this visualization mean?", expanded=True):
            st.info(
                "This tool helps visualize solids of revolution.\n\n"
                "- **Disk/Washer Method**: uses horizontal/vertical slices perpendicular to the axis.\n"
                "- **Cylindrical Shell Method**: wraps vertical slices around the axis.\n\n"
                "Integrals here compute volume — just like Riemann sums approximate area!"
            )
