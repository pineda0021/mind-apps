import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sympy import symbols, sympify, pi, integrate, latex, simplify, Rational

st.set_page_config("MIND: Solid of Revolution Tool", layout="wide")
st.title("🧠 MIND: Solid of Revolution Tool")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

# --- Sidebar ---
st.sidebar.header("Parameters")
function_mode = st.sidebar.selectbox("Function setup", ["One Function", "Two Functions"])
top_expr = st.sidebar.text_input("Top function f(x):", "x")
bottom_expr = st.sidebar.text_input("Bottom function g(x):", "x**2") if function_mode == "Two Functions" else "0"
method = st.sidebar.selectbox("Method", ["Disk/Washer", "Shell"])
axis = st.sidebar.selectbox("Axis of rotation", ["x-axis", "y-axis"])
a = st.sidebar.number_input("Start of interval a", 0.0)
b = st.sidebar.number_input("End of interval b", 1.0)
show_3d = st.sidebar.checkbox("Show 3D Visualization", True)
compute = st.sidebar.button("🔄 Compute and Visualize")

x = symbols('x')
f_expr = sympify(top_expr)
g_expr = sympify(bottom_expr)

# --- Helper: numerical eval ---
def parse(expr):
    return lambda x: eval(expr, {"x": x, "np": np})

# --- 2D Region Plot ---
def plot_region():
    fx = parse(top_expr)
    gx = parse(bottom_expr)
    xs = np.linspace(a, b, 400)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, fx(xs), label="f(x)", color='blue')
    if bottom_expr != "0":
        plt.plot(xs, gx(xs), label="g(x)", color='red')
        plt.fill_between(xs, gx(xs), fx(xs), color='gray', alpha=0.3)
    else:
        plt.fill_between(xs, 0, fx(xs), color='gray', alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

# --- Volume and Formula Display ---
def display_formula_and_steps():
    if method == "Disk/Washer" and axis == "x-axis":
        vol_expr = pi * (f_expr**2 - g_expr**2)
        int_f2 = integrate(f_expr**2, (x, a, b))
        int_g2 = integrate(g_expr**2, (x, a, b))
        step_expr = simplify(int_f2 - int_g2)
        final_expr = simplify(pi * step_expr)

        st.markdown("### 📘 Volume Formula")
        st.latex(rf"V = \pi \int_{{{a}}}^{{{b}}} \left({latex(f_expr)}^2 - {latex(g_expr)}^2\right)\,dx")

        st.markdown("### 📝 Step-by-Step")
        st.latex(rf"""
        \begin{{aligned}}
        V &= \pi \left( \int_{{{a}}}^{{{b}}} {latex(f_expr**2)}\,dx - \int_{{{a}}}^{{{b}}} {latex(g_expr**2)}\,dx \right) \\
        &= \pi \left( {latex(int_f2)} - {latex(int_g2)} \right) \\
        &= \pi \cdot \left( {latex(step_expr)} \right) \\
        &= {latex(final_expr)}
        \end{{aligned}}
        """)

        return float(final_expr.evalf())

    elif method == "Shell" and axis == "y-axis":
        shell_expr = 2 * pi * x * (f_expr - g_expr)
        integral = simplify(integrate(shell_expr, (x, a, b)))
        st.markdown("### 📘 Volume Formula")
        st.latex(rf"V = 2\pi \int_{{{a}}}^{{{b}}} x({latex(f_expr)} - {latex(g_expr)})\,dx = {latex(integral)}")
        return float(integral.evalf())

    else:
        st.warning("Unsupported combination.")
        return None

# --- 3D Disk/Washer Slices ---
def plot_disk_riemann():
    fx = parse(top_expr)
    gx = parse(bottom_expr)
    xs = np.linspace(a, b, 20)
    fig = go.Figure()
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i+1]
        x_mid = (x0 + x1) / 2
        r_outer = fx(x_mid)
        r_inner = gx(x_mid)
        theta = np.linspace(0, 2*np.pi, 30)
        T, R = np.meshgrid(theta, np.linspace(r_inner, r_outer, 2))
        X = x_mid * np.ones_like(R)
        Y = R * np.cos(T)
        Z = R * np.sin(T)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.6, colorscale='blues'))
    fig.update_layout(title="3D Riemann Slices (Disk/Washer)", height=500,
                      scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'))
    st.plotly_chart(fig)

# --- 3D Shell Slices ---
def plot_shell_riemann():
    fx = parse(top_expr)
    gx = parse(bottom_expr)
    xs = np.linspace(a, b, 20)
    fig = go.Figure()
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i+1]
        h = fx((x0 + x1)/2) - gx((x0 + x1)/2)
        r = (x0 + x1)/2
        theta = np.linspace(0, 2*np.pi, 30)
        X = r * np.cos(theta)
        Y = np.linspace(0, h, 2)
        X, Y = np.meshgrid(X, Y)
        Z = r * np.sin(theta)[None, :]
        fig.add_trace(go.Surface(x=X, y=Y.T, z=Z, showscale=False, opacity=0.6, colorscale='blues'))
    fig.update_layout(title="3D Cylindrical Shells", height=500,
                      scene=dict(xaxis_title='x', yaxis_title='height', zaxis_title='z'))
    st.plotly_chart(fig)

# --- Main Run ---
if compute:
    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.markdown("## ✏️ Region Bounded by Curves")
        plot_region()
    with col2:
        st.markdown("## 📊 3D Visualization")
        if show_3d:
            if method == "Disk/Washer" and axis == "x-axis":
                plot_disk_riemann()
            elif method == "Shell" and axis == "y-axis":
                plot_shell_riemann()
            else:
                st.warning("3D visualization not available for this method/axis.")

    volume = display_formula_and_steps()
    if volume is not None:
        st.markdown(f"### ✅ Exact Volume: `{volume:.4f}`")

    st.markdown("## 💡 Interpretation Tip")
    st.info(
        "- **Disk/Washer**: Good when rotating around the x-axis.\n"
        "- **Shell**: Better for y-axis. This tool helps students see how volume is built from slices!"
    )
