import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sympy import symbols, sympify, integrate, pi, latex, simplify, Rational

# --- Page Config ---
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

# --- Parse expressions ---
x = symbols('x')
f_expr = sympify(top_expr)
g_expr = sympify(bottom_expr)

def parse(expr):
    return lambda x: eval(expr, {"x": x, "np": np})

# --- Region plot ---
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

# --- Symbolic formula + steps ---
from sympy import Rational

def display_formula():
    if method == "Disk/Washer" and axis == "x-axis":
        st.markdown("### 📘 Volume Formula")
        st.latex(r"V = \pi \int_{%.2f}^{%.2f} \left[f(x)^2 - g(x)^2\right] dx" % (a, b))
        f_sq = f_expr**2
        g_sq = g_expr**2
        integrand = pi * (f_sq - g_sq)
        exact = integrate(integrand, (x, a, b))
        simplified = simplify(exact)
        st.markdown("### 📝 Step-by-Step")
        st.latex("f(x)^2 = " + latex(f_sq))
        st.latex("g(x)^2 = " + latex(g_sq))
        st.latex(r"V = \pi \int_{%.2f}^{%.2f} \left[%s - %s\right] dx = %s" %
                 (a, b, latex(f_sq), latex(g_sq), latex(simplified)))
        return simplified  # return symbolic (not float)

    elif method == "Shell" and axis == "y-axis":
        st.markdown("### 📘 Volume Formula")
        st.latex(r"V = 2\pi \int_{%.2f}^{%.2f} x \cdot \left[f(x) - g(x)\right] dx" % (a, b))
        shell_expr = f_expr - g_expr
        shell_integrand = 2 * pi * x * shell_expr
        exact = integrate(shell_integrand, (x, a, b))
        simplified = simplify(exact)
        st.markdown("### 📝 Step-by-Step")
        st.latex("f(x) - g(x) = " + latex(shell_expr))
        st.latex(r"V = 2\pi \int_{%.2f}^{%.2f} x \cdot (%s) dx = %s" %
                 (a, b, latex(shell_expr), latex(simplified)))
        return simplified  # return symbolic (not float)
    else:
        st.warning("Method and axis combination not supported.")
        return None


# --- 3D Disk/Washer Visualization ---
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

# --- 3D Shell Visualization ---
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
        T, H = np.meshgrid(theta, np.linspace(0, h, 2))
        X = r * np.cos(T)
        Y = H
        Z = r * np.sin(T)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.6, colorscale='blues'))
    fig.update_layout(title="3D Cylindrical Shells", height=500,
                      scene=dict(xaxis_title='radius', yaxis_title='height', zaxis_title='z'))
    st.plotly_chart(fig)

# --- Main Display ---
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

    volume = display_formula()
    if volume is not None:
    st.markdown(f"### ✅ Exact Volume: ${latex(volume)}$")

    st.markdown("## 💡 Interpretation Tip")
    st.info(
        "- **Disk/Washer**: Good when rotating around the x-axis.\n"
        "- **Shell**: Better for y-axis. This tool helps students see how volume is built from slices!"
    )

