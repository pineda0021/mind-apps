import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sympy import symbols, integrate, pi, Rational, latex, simplify, sympify, Abs
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import plotly.graph_objs as go

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
else:
    top_expr = st.sidebar.text_input("Top function f(x):", value="x")
    bottom_expr = st.sidebar.text_input("Bottom function g(x):", value="x**2")

method = st.sidebar.selectbox("Method:", ["Disk/Washer", "Cylindrical Shell"])
axis = st.sidebar.selectbox("Axis of rotation:", ["x-axis", "y-axis"])
rotation_line = st.sidebar.text_input("Axis of rotation value (e.g., x=2 or y=3):", value="x=0")
show_riemann = st.sidebar.checkbox("Show Riemann Sum in 3D", value=True)
show_animation = st.sidebar.checkbox("Show Animated Revolution", value=True)
a = st.sidebar.number_input("Start of interval (a):", value=0.0)
b = st.sidebar.number_input("End of interval (b):", value=1.0)
compute = st.sidebar.button("🔄 Compute and Visualize")

# --- Function parser ---
def parse_function(expr):
    return lambda x: eval(expr, {"x": x, "np": np})

# --- Extract shift from axis of rotation ---
def extract_shift(axis_expr):
    try:
        axis_name, val = axis_expr.replace(" ", "").split("=")
        return axis_name, float(val)
    except:
        return ("x", 0.0)

# --- Plot functions ---
def plot_functions(top_expr, bottom_expr):
    f = parse_function(top_expr)
    x = np.linspace(a, b, 300)
    plt.figure(figsize=(6, 4))
    plt.plot(x, f(x), label="Top: f(x)", color='blue')
    if bottom_expr:
        g = parse_function(bottom_expr)
        plt.plot(x, g(x), label="Bottom: g(x)", color='red')
        plt.fill_between(x, g(x), f(x), color='gray', alpha=0.3)
    else:
        plt.fill_between(x, 0, f(x), color='gray', alpha=0.3)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x), g(x)")
    st.pyplot(plt.gcf())
    plt.close()

# --- Show formula ---
def show_formula(method, axis, f_expr, g_expr):
    x = symbols('x')
    f = sympify(f_expr)
    g = sympify(g_expr) if g_expr else 0
    if method == "Disk/Washer" and axis == "x-axis":
        expr = pi * ((f ** 2) - (g ** 2))
        integral = integrate(expr, (x, a, b))
        st.markdown("#### 📘 Setup and Formula")
        st.latex(r"f(x) = " + latex(f))
        if g_expr:
            st.latex(r"g(x) = " + latex(g))
        st.latex(r"V = \pi \int_{{{}}}^{{{}}} \left[{}^2 - {}^2\right] \,dx".format(a, b, latex(f), latex(g)))

# --- Step-by-step solution ---
def step_by_step_solution(f_expr, g_expr, method, axis, a, b, axis_shift):
    x = symbols('x')
    f = sympify(f_expr)
    g = sympify(g_expr) if g_expr else 0
    integrand = (f ** 2 - g ** 2) * pi
    result = integrate(integrand, (x, a, b))
    st.markdown("#### 📝 Step-by-Step Solution:")
    st.latex(r"V = \pi \int_{{{}}}^{{{}}} \left[{}^2 - {}^2\right] \, dx = {}".format(a, b, latex(f), latex(g), latex(simplify(result))))

# --- Compute exact volume ---
def compute_exact_volume(f_expr, g_expr, method, axis, a, b, axis_shift):
    f = parse_function(f_expr)
    g = parse_function(g_expr) if g_expr else (lambda x: 0)
    integrand = lambda x: np.pi * (f(x) ** 2 - g(x) ** 2)
    result, _ = quad(integrand, a, b)
    return result

# --- Placeholder animation function ---
def animate_revolution(f_expr):
    st.info("⚠️ Animated revolution not yet implemented. Coming soon!")

# --- Method Tip ---
def show_method_tip(method, axis):
    st.markdown("### ✅ Which Method is Better?")
    if method == "Disk/Washer" and axis == "x-axis":
        st.success("The Disk/Washer Method is generally preferred for solids rotated around the x-axis.")
    elif method == "Cylindrical Shell" and axis == "y-axis":
        st.success("Shell Method is often more convenient when rotating around the y-axis.")
    else:
        st.warning("This method/axis combo might require transforming the function or limits.")

# --- 3D Disk Riemann Visualization ---
def plot_disk_riemann_3d(top_expr):
    f = parse_function(top_expr)
    x_vals = np.linspace(a, b, 20)
    fig = go.Figure()

    for i in range(len(x_vals) - 1):
        x0 = x_vals[i]
        x1 = x_vals[i + 1]
        x_mid = (x0 + x1) / 2
        r = f(x_mid)
        theta = np.linspace(0, 2 * np.pi, 30)
        X = x_mid * np.ones((30, 30))
        T, Z = np.meshgrid(theta, np.linspace(0, r, 30))
        Y = Z * np.cos(T)
        Z = Z * np.sin(T)

        fig.add_trace(go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.6, colorscale='blues'))

    fig.update_layout(title="Riemann Slices Forming the Solid (Disk Method)",
                      scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'), height=500)
    st.plotly_chart(fig)

# --- Main ---
if compute:
    axis_name, axis_shift = extract_shift(rotation_line)
    col_left, col_right = st.columns([1.2, 0.8])
    with col_left:
        plot_functions(top_expr, bottom_expr)
        show_formula(method, axis, top_expr, bottom_expr)
        step_by_step_solution(top_expr, bottom_expr, method, axis, a, b, axis_shift)
        exact_volume = compute_exact_volume(top_expr, bottom_expr, method, axis, a, b, axis_shift)
        st.markdown(f"### ✅ Exact Volume: {exact_volume:.4f}")
        if show_riemann:
            if method == "Cylindrical Shell" and axis == "y-axis":
                st.warning("Shell method Riemann 3D visualization coming soon.")
            elif method == "Disk/Washer" and axis == "x-axis":
                plot_disk_riemann_3d(top_expr)
            else:
                st.info("Riemann 3D visualization for this method/axis not yet available.")
        if show_animation and method == "Disk/Washer" and axis == "x-axis":
            animate_revolution(top_expr)

    with col_right:
        show_method_tip(method, axis)
        with st.expander("🪞 What does this visualization mean?", expanded=True):
            st.info(
                "This tool helps visualize solids of revolution.\n\n"
                "- **Disk/Washer Method**: horizontal/vertical slices.\n"
                "- **Shell Method**: wraps vertical slices around the axis.\n\n"
                "Integrals compute volume — just like Riemann sums approximate area!"
            )
