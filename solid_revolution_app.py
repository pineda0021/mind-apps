import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sympy import symbols, pi, simplify, latex, integrate, sympify
import plotly.graph_objs as go

st.set_page_config(page_title="MIND: Solid of Revolution Tool", layout="wide")
st.title("🧠 MIND: Solid of Revolution Tool")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

# Sidebar Inputs
st.sidebar.header("🔧 Input Parameters")
function_option = st.sidebar.selectbox("Function Type:", ["One Function", "Two Functions"])
method = st.sidebar.selectbox("Method:", ["Disk/Washer", "Cylindrical Shell"])
axis = st.sidebar.selectbox("Axis of Rotation:", ["x-axis", "y-axis"])
a = st.sidebar.number_input("Start of interval (a):", value=0.0)
b = st.sidebar.number_input("End of interval (b):", value=1.0)
show_riemann = st.sidebar.checkbox("Show 3D Riemann Slices", value=True)
compute = st.sidebar.button("🔄 Compute and Visualize")

# Function Input
if function_option == "One Function":
    f_expr = st.sidebar.text_input("f(x):", value="x**(1/2)")
    g_expr = None
else:
    f_expr = st.sidebar.text_input("Top function f(x):", value="x")
    g_expr = st.sidebar.text_input("Bottom function g(x):", value="x**2")

# Function Parser
def parse_function(expr):
    return lambda x: eval(expr, {"x": x, "np": np})

# Formula Display
def display_formula(f_expr, g_expr, method, axis):
    x = symbols('x')
    f = sympify(f_expr)
    g = sympify(g_expr) if g_expr else 0
    if method == "Disk/Washer" and axis == "x-axis":
        formula = f"V = \\pi \\int_{{{a}}}^{{{b}}} \\left[{latex(f)}^2 - {latex(g)}^2\\right] \\, dx"
    elif method == "Cylindrical Shell" and axis == "y-axis":
        formula = f"V = 2\\pi \\int_{{{a}}}^{{{b}}} x\\left[{latex(f)} - {latex(g)}\\right] \\, dx"
    else:
        formula = "Unsupported combination."
    st.markdown("### 📘 Volume Formula")
    st.latex(formula)

# Step-by-step volume
def compute_symbolic_volume(f_expr, g_expr, method, axis, a, b):
    x = symbols('x')
    f = sympify(f_expr)
    g = sympify(g_expr) if g_expr else 0
    if method == "Disk/Washer" and axis == "x-axis":
        integrand = pi * (f**2 - g**2)
    elif method == "Cylindrical Shell" and axis == "y-axis":
        integrand = 2 * pi * x * (f - g)
    else:
        st.warning("Unsupported combination.")
        return None, None
    exact = simplify(integrate(integrand, (x, a, b)))
    numeric = float(exact.evalf())
    return exact, numeric

# Plot 2D region
def plot_region(f_expr, g_expr, a, b):
    x_vals = np.linspace(a, b, 300)
    f = parse_function(f_expr)
    plt.figure()
    plt.plot(x_vals, f(x_vals), label="f(x)", color="blue")
    if g_expr:
        g = parse_function(g_expr)
        plt.plot(x_vals, g(x_vals), label="g(x)", color="red")
        plt.fill_between(x_vals, g(x_vals), f(x_vals), color='gray', alpha=0.3)
    else:
        plt.fill_between(x_vals, 0, f(x_vals), color='gray', alpha=0.3)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    st.pyplot(plt.gcf())
    plt.close()

# 3D Riemann Visual
def plot_riemann_3d(f_expr, g_expr, method, axis, a, b):
    f = parse_function(f_expr)
    g = parse_function(g_expr) if g_expr else lambda x: 0
    x_vals = np.linspace(a, b, 20)
    fig = go.Figure()

    for i in range(len(x_vals)-1):
        x0 = x_vals[i]
        x1 = x_vals[i+1]
        x_mid = (x0 + x1)/2
        height = f(x_mid) - g(x_mid)
        if height < 0: continue
        if method == "Disk/Washer" and axis == "x-axis":
            r = f(x_mid)
            theta = np.linspace(0, 2*np.pi, 30)
            T, Z = np.meshgrid(theta, np.linspace(g(x_mid), f(x_mid), 2))
            X = x_mid * np.ones_like(T)
            Y = (Z) * np.cos(T)
            Z = (Z) * np.sin(T)
            fig.add_trace(go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.6, colorscale='blues'))
        elif method == "Cylindrical Shell" and axis == "y-axis":
            r = x_mid
            h = height
            theta = np.linspace(0, 2*np.pi, 30)
            Z, T = np.meshgrid(np.linspace(0, h, 2), theta)
            X = r * np.cos(T)
            Y = r * np.sin(T)
            fig.add_trace(go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.6, colorscale='greens'))

    fig.update_layout(scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'), height=500,
                      title="3D Riemann Approximation")
    st.plotly_chart(fig)

# Main App
if compute:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("## ✏️ Region Bounded by Curves")
        plot_region(f_expr, g_expr, a, b)
        display_formula(f_expr, g_expr, method, axis)
        exact, approx = compute_symbolic_volume(f_expr, g_expr, method, axis, a, b)
        if exact is not None:
            st.markdown("### 🧮 Exact Volume:")
            st.latex(f"V = {latex(exact)} ≈ {approx:.4f}")
    with col2:
        st.markdown("## 📊 3D Visualization")
        if show_riemann:
            plot_riemann_3d(f_expr, g_expr, method, axis, a, b)
        else:
            st.info("Riemann visualization not selected.")

    st.markdown("### 💡 Interpretation Tip")
    st.info(
        "- Disk/Washer: Good when rotating around the x-axis.\n"
        "- Shell: Better for y-axis.\n"
        "This tool helps students see how volume is built from slices!"
    )
