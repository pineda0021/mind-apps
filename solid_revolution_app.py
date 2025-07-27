import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sympy import symbols, integrate, pi, Rational, latex, simplify, sympify
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("Agg")

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

show_riemann = st.sidebar.checkbox("Show Riemann Sum in 3D", value=True)
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

# --- 3D Plot of Solid or Riemann ---
def plot_3d_solid(f_expr, g_expr=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    x_vals = np.linspace(a, b, 100)
    theta_vals = np.linspace(0, 2*np.pi, 60)
    X, T = np.meshgrid(x_vals, theta_vals)
    f = parse_function(f_expr)
    g = parse_function(g_expr) if g_expr else (lambda x: 0)

    Y_top = f(X)
    Y_bot = g(X)

    R_outer = Y_top
    R_inner = Y_bot

    X_outer = R_outer * np.cos(T)
    Z_outer = R_outer * np.sin(T)

    if g_expr:
        X_inner = R_inner * np.cos(T)
        Z_inner = R_inner * np.sin(T)
        for i in range(len(x_vals)):
            ax.plot([x_vals[i]]*len(theta_vals), X_outer[:, i], Z_outer[:, i], color='blue', alpha=0.6)
            ax.plot([x_vals[i]]*len(theta_vals), X_inner[:, i], Z_inner[:, i], color='white', alpha=0.3)
    else:
        for i in range(len(x_vals)):
            ax.plot([x_vals[i]]*len(theta_vals), X_outer[:, i], Z_outer[:, i], color='blue', alpha=0.6)

    ax.set_title("3D Solid of Revolution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    st.pyplot(fig)

# --- Riemann Simulation ---
def plot_riemann_sum(f_expr):
    x_vals = np.linspace(a, b, 10)
    dx = (b - a) / 10
    f = parse_function(f_expr)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for x in x_vals:
        height = f(x)
        theta = np.linspace(0, 2 * np.pi, 30)
        X = height * np.cos(theta)
        Z = height * np.sin(theta)
        ax.plot([x]*len(X), X, Z, color='purple')
    ax.set_title("3D Riemann Sum Slices for Volume")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    st.pyplot(fig)

# --- Volume ---
def compute_exact_volume(top_expr, bottom_expr, method, axis, a, b):
    f_top = parse_function(top_expr)
    f_bot = parse_function(bottom_expr) if bottom_expr else None

    if method == "Disk/Washer":
        integrand = (lambda x: np.pi * (f_top(x)**2)) if not f_bot else (lambda x: np.pi * (f_top(x)**2 - f_bot(x)**2))
    else:
        integrand = lambda x: 2 * np.pi * x * f_top(x)

    volume, _ = quad(integrand, a, b)
    return volume

# --- Formula ---
def show_formula(method, axis, f_expr, g_expr=None):
    st.markdown("### 📘 Setup and Formula")
    st.latex(f"f(x) = {f_expr}")
    if g_expr:
        st.latex(f"g(x) = {g_expr}")
    if method == "Disk/Washer":
        st.latex(r"V = \pi \int_a^b [f(x)^2 - g(x)^2] \,dx" if g_expr else r"V = \pi \int_a^b [f(x)^2] \,dx")
    else:
        st.latex(r"V = 2\pi \int_a^b x f(x)\,dx")

# --- Steps ---
def step_by_step_solution(top_expr, bottom_expr, method, axis, a, b):
    st.markdown("### 📋 Step-by-Step Solution:")
    x = symbols('x')
    f_top = sympify(top_expr)
    f_bot = sympify(bottom_expr) if bottom_expr else None

    if method == "Disk/Washer":
        integrand = f_top**2 if not f_bot else f_top**2 - f_bot**2
        symbolic_integral = pi * integrate(integrand, (x, a, b))
        st.latex(f"V = \pi \int_{{{a}}}^{{{b}}} {latex(integrand)} \, dx")
    else:
        integrand = x * f_top
        symbolic_integral = 2 * pi * integrate(integrand, (x, a, b))
        st.latex(f"V = 2\pi \int_{{{a}}}^{{{b}}} x \cdot {latex(f_top)} \, dx")

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

# --- Tips ---
def show_method_tip(method, axis):
    st.markdown("### ✅ Which Method is Better?")
    if method == "Disk/Washer":
        if axis == "x-axis":
            st.success("Great choice! Disk/Washer method about x-axis is simple.")
        else:
            st.warning("May require solving x = f(y) for y-axis revolutions.")
    else:
        if axis == "y-axis":
            st.success("Shells about y-axis works well with vertical rectangles.")
        else:
            st.warning("Shells about x-axis may be tricky if f(y) isn't easy.")

# --- Main ---
if compute:
    col_left, col_right = st.columns([1.2, 0.8])
    with col_left:
        plot_functions(top_expr, bottom_expr)
        show_formula(method, axis, top_expr, bottom_expr)
        step_by_step_solution(top_expr, bottom_expr, method, axis, a, b)
        exact_volume = compute_exact_volume(top_expr, bottom_expr, method, axis, a, b)
        st.markdown(f"### ✅ Exact Volume: {exact_volume:.4f}")
        plot_3d_solid(top_expr, bottom_expr)
        if show_riemann:
            plot_riemann_sum(top_expr)

    with col_right:
        show_method_tip(method, axis)
        with st.expander("🪞 What does this visualization mean?", expanded=True):
            st.info(
                "This tool helps visualize solids of revolution.\n\n"
                "- **Disk/Washer Method**: horizontal/vertical slices.\n"
                "- **Shell Method**: wraps vertical slices around the axis.\n\n"
                "Integrals compute volume — just like Riemann sums approximate area!"
            )
