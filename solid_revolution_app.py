import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sympy import symbols, integrate, pi, latex, simplify, sympify
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import plotly.graph_objs as go
import io

# --- Setup ---
st.set_page_config(page_title="MIND: Solid of Revolution Tool", layout="wide")
st.title("🧠 MIND: Solid of Revolution Tool")
st.caption("Created by Professor Edward Pineda-Castro — built with the students in MIND.")

# --- Sidebar ---
st.sidebar.header("🔧 Parameters")
function_type = st.sidebar.selectbox("Do you have one or two functions?", ["One Function", "Two Functions"])

top_expr = st.sidebar.text_input("Top function f(x):", "x")
bottom_expr = st.sidebar.text_input("Bottom function g(x):", "x**2") if function_type == "Two Functions" else None
method = st.sidebar.selectbox("Method:", ["Disk/Washer"])
axis = st.sidebar.selectbox("Axis of rotation:", ["x-axis"])
a = st.sidebar.number_input("Start of interval a:", value=0.0)
b = st.sidebar.number_input("End of interval b:", value=1.0)
show_riemann = st.sidebar.checkbox("Show Riemann Sum in 3D", value=True)
show_animation = st.sidebar.checkbox("Show Animated Revolution", value=True)
compute = st.sidebar.button("🔄 Compute and Visualize")

# --- Helpers ---
def parse_function(expr):
    return lambda x: eval(expr, {"x": x, "np": np})

def show_formula(f_expr, g_expr):
    x = symbols('x')
    f = sympify(f_expr)
    g = sympify(g_expr) if g_expr else 0
    st.markdown("#### 📘 Setup and Formula")
    st.latex(f"f(x) = {latex(f)}")
    if g_expr:
        st.latex(f"g(x) = {latex(g)}")
    st.latex(r"V = \pi \int_{" + str(a) + r"}^{" + str(b) + r"} \left[" + latex(f**2) + " - " + latex(g**2) + r"\right] \,dx")

def step_by_step_solution(f_expr, g_expr):
    x = symbols('x')
    f = sympify(f_expr)
    g = sympify(g_expr) if g_expr else 0
    integrand = pi * (f**2 - g**2)
    result = integrate(integrand, (x, a, b))
    st.markdown("#### 📝 Step-by-Step Solution:")
    st.latex(r"V = \pi \int_{" + str(a) + r"}^{" + str(b) + r"} \left[" + latex(f**2) + " - " + latex(g**2) + r"\right] \,dx = " + latex(simplify(result)))

def compute_exact_volume(f_expr, g_expr):
    f = parse_function(f_expr)
    g = parse_function(g_expr) if g_expr else (lambda x: 0)
    integrand = lambda x: np.pi * (f(x)**2 - g(x)**2)
    result, _ = quad(integrand, a, b)
    return result

def plot_2d_functions(f_expr, g_expr):
    f = parse_function(f_expr)
    g = parse_function(g_expr) if g_expr else (lambda x: 0)
    x_vals = np.linspace(a, b, 300)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_vals, f(x_vals), label="Top: f(x)", color='blue')
    if g_expr:
        ax.plot(x_vals, g(x_vals), label="Bottom: g(x)", color='red')
        ax.fill_between(x_vals, g(x_vals), f(x_vals), color='gray', alpha=0.3)
    else:
        ax.fill_between(x_vals, 0, f(x_vals), color='gray', alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x), g(x)")
    ax.legend()
    st.pyplot(fig)

def plot_riemann_3d(f_expr):
    f = parse_function(f_expr)
    x_vals = np.linspace(a, b, 20)
    fig = go.Figure()
    for i in range(len(x_vals) - 1):
        x0, x1 = x_vals[i], x_vals[i + 1]
        x_mid = (x0 + x1) / 2
        r = f(x_mid)
        theta = np.linspace(0, 2 * np.pi, 30)
        T, Z = np.meshgrid(theta, np.linspace(0, r, 10))
        X = np.full_like(T, x_mid)
        Y = Z * np.cos(T)
        Z = Z * np.sin(T)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.6, colorscale='blues'))

    fig.update_layout(title="Riemann Slices Forming the Solid (Disk Method)",
                      scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'), height=500)
    st.plotly_chart(fig)

def animate_revolution(f_expr):
    f = parse_function(f_expr)
    x_vals = np.linspace(a, b, 100)
    y_vals = f(x_vals)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    def surface(theta, r):
        X = np.outer(x_vals, np.cos(theta))
        Y = np.outer(x_vals, np.sin(theta))
        Z = np.outer(y_vals, np.ones_like(theta))
        return X, Y, Z

    theta_vals = np.linspace(0, 2*np.pi, 60)

    def update(i):
        ax.clear()
        X, Y, Z = surface(theta_vals[:i+1], y_vals)
        ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.7)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, max(y_vals)])
        ax.set_title("Solid of Revolution Animation")

    ani = animation.FuncAnimation(fig, update, frames=len(theta_vals), interval=50)
    buf = io.BytesIO()
    ani.save(buf, format='gif')
    st.image(buf.getvalue(), caption="Animated Solid of Revolution")

# --- Main App ---
if compute:
    col1, col2 = st.columns([1.4, 0.6])

    with col1:
        plot_2d_functions(top_expr, bottom_expr)
        show_formula(top_expr, bottom_expr)
        step_by_step_solution(top_expr, bottom_expr)
        volume = compute_exact_volume(top_expr, bottom_expr)
        st.markdown(f"### ✅ Exact Volume: {volume:.4f}")
        if show_riemann:
            plot_riemann_3d(top_expr)
        if show_animation:
            animate_revolution(top_expr)

    with col2:
        st.markdown("### ✅ Which Method is Better?")
        st.success("The Disk/Washer Method is generally preferred for solids rotated around the x-axis.")
        with st.expander("📘 What does this visualization mean?", expanded=True):
            st.info("""
This tool helps visualize solids of revolution:

- **Disk/Washer Method**: slices perpendicular to axis of revolution.
- **Shell Method**: slices wrapped around the axis (not yet implemented).

Integrals compute exact volume — just like Riemann sums approximate area.
            """)
