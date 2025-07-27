import streamlit as st
import numpy as np
from sympy import symbols, sympify, pi, latex, simplify, integrate
import plotly.graph_objs as go

# --- App Config ---
st.set_page_config(page_title="MIND: Solid Revolution Tool", layout="wide")
st.title("🧠 MIND: Solid of Revolution Tool")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

# --- Inputs ---
st.sidebar.header("🔧 Parameters")

function_option = st.sidebar.selectbox("Do you have one or two functions?", ["One Function", "Two Functions"])
top_expr = st.sidebar.text_input("Top Function f(x):", value="x**2")
bottom_expr = None if function_option == "One Function" else st.sidebar.text_input("Bottom Function g(x):", value="x")

method = st.sidebar.selectbox("Method:", ["Disk/Washer", "Cylindrical Shell"])
axis = st.sidebar.selectbox("Axis of Rotation:", ["x-axis", "y-axis"])
a = st.sidebar.number_input("Start of interval (a):", value=0.0)
b = st.sidebar.number_input("End of interval (b):", value=1.0)
show_animation = st.sidebar.checkbox("Show Animated Revolution", value=True)

# --- Utility Functions ---
def parse_function(expr):
    return lambda x: eval(expr, {"x": x, "np": np})

def show_formula_and_steps(f_expr, g_expr, method, axis, a, b):
    x = symbols('x')
    f = sympify(f_expr)
    g = sympify(g_expr) if g_expr else 0
    st.markdown("### 📝 Step-by-Step Setup")

    if method == "Disk/Washer" and axis == "x-axis":
        st.latex(r"f(x) = " + latex(f))
        if g_expr:
            st.latex(r"g(x) = " + latex(g))
        integral_expr = pi * (f**2 - g**2)
        st.latex(
            r"V = \pi \int_{{{}}}^{{{}}} \left[{}^2 - {}^2\right] \, dx = {}".format(
                a, b, latex(f), latex(g), latex(simplify(integrate(integral_expr, (x, a, b))))
            )
        )
    elif method == "Cylindrical Shell" and axis == "y-axis":
        st.latex(r"f(x) = " + latex(f))
        if g_expr:
            st.latex(r"g(x) = " + latex(g))
        shell_expr = 2 * pi * x * (f - g)
        st.latex(
            r"V = 2\pi \int_{{{}}}^{{{}}} x({} - {}) \, dx = {}".format(
                a, b, latex(f), latex(g), latex(simplify(integrate(shell_expr, (x, a, b))))
            )
        )
    else:
        st.warning("This axis/method combo not yet supported for symbolic steps.")

def compute_numeric_volume(f_expr, g_expr, method, axis, a, b):
    f = parse_function(f_expr)
    g = parse_function(g_expr) if g_expr else (lambda x: 0)
    if method == "Disk/Washer" and axis == "x-axis":
        return np.pi * np.trapz(f(np.linspace(a, b, 100))**2 - g(np.linspace(a, b, 100))**2, dx=(b - a)/100)
    elif method == "Cylindrical Shell" and axis == "y-axis":
        x_vals = np.linspace(a, b, 100)
        return 2 * np.pi * np.trapz(x_vals * (f(x_vals) - g(x_vals)), dx=(b - a)/100)
    else:
        return None

def animate_revolution(f_expr, a, b):
    f = parse_function(f_expr)
    x_vals = np.linspace(a, b, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    X, T = np.meshgrid(x_vals, theta)
    R = f(X)

    Y = R * np.cos(T)
    Z = R * np.sin(T)

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='blues', opacity=0.8)])
    fig.update_layout(
        title="🔁 Animated Solid of Revolution",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z"
        ),
        height=600
    )
    st.plotly_chart(fig)

def method_tip(method, axis):
    st.markdown("### 🤔 Which Method is Better?")
    if method == "Disk/Washer" and axis == "x-axis":
        st.success("Use the Disk/Washer method when rotating around the x-axis with vertical slices.")
    elif method == "Cylindrical Shell" and axis == "y-axis":
        st.success("Use the Shell method for rotation about the y-axis with vertical slices.")
    else:
        st.info("Consider algebraic manipulation or changing integration variable.")

# --- Main Output ---
if st.button("🔄 Compute and Visualize"):
    col1, col2 = st.columns([2, 1])
    with col1:
        show_formula_and_steps(top_expr, bottom_expr, method, axis, a, b)
        vol = compute_numeric_volume(top_expr, bottom_expr, method, axis, a, b)
        if vol:
            st.markdown(f"### ✅ Approximate Volume: `{vol:.5f}` units³")
        if show_animation and method == "Disk/Washer" and axis == "x-axis":
            animate_revolution(top_expr, a, b)
    with col2:
        method_tip(method, axis)
        with st.expander("📘 What’s Going On Here?", expanded=True):
            st.info(
                "You're revolving a region around an axis to create a 3D solid.\n\n"
                "- Disk/Washer: Think of slicing the solid horizontally or vertically.\n"
                "- Shell: Think of wrapping thin shells around the axis.\n\n"
                "This app helps visualize the integral that calculates the volume!"
            )
