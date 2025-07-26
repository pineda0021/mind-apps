import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sympy import symbols, integrate, pi, Rational, latex, simplify, sympify

# --- Page config ---
st.set_page_config(page_title="MIND: Solid Revolution Tool", layout="wide")
st.title("🧠 MIND: Solid of Revolution Tool")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

# --- Sidebar inputs ---
st.sidebar.header("🔧 Parameters")

# Option for selecting one or two functions
function_option = st.sidebar.selectbox("Do you have one function or two functions?", ["One Function", "Two Functions"])

if function_option == "One Function":
    top_expr = st.sidebar.text_input("Function f(x):", value="x**(1/2)")
    bottom_expr = None
    method = st.sidebar.selectbox("Method:", ["Disk/Washer", "Cylindrical Shell"])
    if method == "Cylindrical Shell":
        axis = "y-axis"
    else:
        axis = st.sidebar.selectbox("Axis of rotation:", ["x-axis", "y-axis"])
else:
    top_expr = st.sidebar.text_input("Top function f(x):", value="x")
    bottom_expr = st.sidebar.text_input("Bottom function g(x):", value="x**2")
    method = st.sidebar.selectbox("Method:", ["Disk/Washer", "Cylindrical Shell"])
    axis = st.sidebar.selectbox("Axis of rotation:", ["x-axis", "y-axis"])

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
    ax.set_xlabel("x")
    ax.set_ylabel("f(x), g(x)")
    ax.legend()
    st.pyplot(fig)

# --- Compute volume ---
def compute_exact_volume(top_expr, bottom_expr, method, axis, a, b):
    f_top = parse_function(top_expr)
    f_bot = parse_function(bottom_expr) if bottom_expr else None

    if method == "Disk/Washer":
        if axis == "x-axis":
            integrand = lambda x: np.pi * (f_top(x)**2) if not f_bot else np.pi * (f_top(x)**2 - f_bot(x)**2)
        else:
            integrand = lambda y: np.pi * (f_top(y)**2) if not f_bot else np.pi * (f_top(y)**2 - f_bot(y)**2)
    else:
        if axis == "y-axis":
            integrand = lambda x: 2 * np.pi * x * f_top(x)
        else:
            integrand = lambda y: 2 * np.pi * y * f_top(y)

    volume, error = quad(integrand, a, b)
    return volume

# --- Display formula ---
def show_formula(method, axis, f_expr, g_expr=None):
    st.markdown("### 📘 Setup and Formula")
    st.latex(f"f(x) = {f_expr}")
    if g_expr:
        st.latex(f"g(x) = {g_expr}")
    if method == "Disk/Washer":
        if axis == "x-axis":
            st.latex(r"V = \pi \int_a^b \left[f(x)^2 - g(x)^2\right] \, dx" if g_expr else r"V = \pi \int_a^b \left[f(x)^2\right] \, dx")
        else:
            st.latex(r"V = \pi \int_c^d \left[f(y)^2 - g(y)^2\right] \, dy" if g_expr else r"V = \pi \int_c^d \left[f(y)^2\right] \, dy")
    else:
        if axis == "y-axis":
            st.latex(r"V = 2\pi \int_a^b x f(x) \, dx")
        else:
            st.latex(r"V = 2\pi \int_a^b y f(y) \, dy")

# --- Step-by-step ---
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
        st.latex(f"= \\frac{{{numer}}}{{{denom}}} \pi")
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

    with col_right:
        show_method_tip(method, axis)
        with st.expander("🪞 What does this visualization mean?", expanded=True):
            st.info(
                "This tool helps visualize solids of revolution.\n\n"
                "- **Disk/Washer Method**: uses horizontal/vertical slices perpendicular to the axis.\n"
                "- **Cylindrical Shell Method**: wraps vertical slices around the axis.\n\n"
                "Integrals here compute volume — just like Riemann sums approximate area!"
            )
