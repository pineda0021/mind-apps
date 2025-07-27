import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sympy import symbols, sympify, integrate, pi, latex, simplify, Rational

# --- Page Config ---
st.set_page_config("MIND: Solid of Revolution Tool", layout="wide")
st.title("🧠 MIND: Solid of Revolution Tool")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

# --- Sidebar Inputs ---
st.sidebar.header("Parameters")
input_mode = st.sidebar.selectbox("Input in terms of:", ["y = f(x)", "x = f(y)"])
function_mode = st.sidebar.selectbox("Function setup", ["One Function", "Two Functions"])
top_expr = st.sidebar.text_input("Top function" if input_mode == "y = f(x)" else "Right function", "x")
bottom_expr = st.sidebar.text_input("Bottom function" if function_mode == "Two Functions" else "Left function", "x**2") if function_mode == "Two Functions" else "0"
method = st.sidebar.selectbox("Method", ["Disk/Washer", "Shell"])
axis = st.sidebar.selectbox("Axis of rotation", ["x-axis", "y-axis"])
a = Rational(st.sidebar.text_input("Start of interval a", "0"))
b = Rational(st.sidebar.text_input("End of interval b", "1"))
show_3d = st.sidebar.checkbox("Show 3D Visualization", True)
compute = st.sidebar.button("🔄 Compute and Visualize")

# --- Symbol Setup ---
var = symbols('x') if input_mode == "y = f(x)" else symbols('y')
f_expr = sympify(top_expr)
g_expr = sympify(bottom_expr)

def parse(expr):
    return lambda v: eval(expr, {"x": v, "y": v, "np": np})

# --- Region Plot ---
def plot_region():
    fx = parse(top_expr)
    gx = parse(bottom_expr)
    vs = np.linspace(float(a), float(b), 400)
    plt.figure(figsize=(6, 4))
    if input_mode == "y = f(x)":
        plt.plot(vs, fx(vs), label="f(x)", color='blue')
        plt.plot(vs, gx(vs), label="g(x)", color='red')
        plt.fill_between(vs, gx(vs), fx(vs), color='gray', alpha=0.3)
        plt.xlabel("x")
        plt.ylabel("y")
    else:
        plt.plot(fx(vs), vs, label="x = f(y)", color='blue')
        plt.plot(gx(vs), vs, label="x = g(y)", color='red')
        plt.fill_betweenx(vs, gx(vs), fx(vs), color='gray', alpha=0.3)
        plt.xlabel("x")
        plt.ylabel("y")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

# --- Volume Formula ---
def display_formula():
    v = var
    if input_mode == "y = f(x)":
        if method == "Disk/Washer" and axis == "x-axis":
            f_sq = simplify(f_expr**2)
            g_sq = simplify(g_expr**2)
            integrand = f_sq - g_sq
            result = pi * integrate(integrand, (v, a, b))
            st.latex(r"V = \pi \int_{%s}^{%s} \left[%s - %s\right] dx = %s" % (latex(a), latex(b), latex(f_sq), latex(g_sq), latex(result)))
            return result

        elif method == "Shell" and axis == "y-axis":
            shell_expr = simplify(v * (f_expr - g_expr))
            result = 2 * pi * integrate(shell_expr, (v, a, b))
            st.latex(r"V = 2\pi \int_{%s}^{%s} x \cdot \left[%s - %s\right] dx = %s" % (latex(a), latex(b), latex(f_expr), latex(g_expr), latex(result)))
            return result

    elif input_mode == "x = f(y)":
        if method == "Disk/Washer" and axis == "y-axis":
            f_sq = simplify(f_expr**2)
            g_sq = simplify(g_expr**2)
            integrand = f_sq - g_sq
            result = pi * integrate(integrand, (v, a, b))
            st.latex(r"V = \pi \int_{%s}^{%s} \left[%s - %s\right] dy = %s" % (latex(a), latex(b), latex(f_sq), latex(g_sq), latex(result)))
            return result

        elif method == "Shell" and axis == "x-axis":
            shell_expr = simplify(v * (f_expr - g_expr))
            result = 2 * pi * integrate(shell_expr, (v, a, b))
            st.latex(r"V = 2\pi \int_{%s}^{%s} y \cdot \left[%s - %s\right] dy = %s" % (latex(a), latex(b), latex(f_expr), latex(g_expr), latex(result)))
            return result

    st.warning("Combination not supported.")
    return None

# --- Main Display ---
if compute:
    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.subheader("✏️ Region Bounded by Curves")
        plot_region()

    with col2:
        st.subheader("📊 3D Visualization")
        if input_mode == "y = f(x)" and axis == "x-axis" and method == "Disk/Washer":
            st.info("3D for Disk/Washer around x-axis supported here (not shown in this demo).")
        else:
            st.warning("3D view not yet available for this mode.")

    volume = display_formula()
    if volume is not None:
        st.success("✅ Exact Volume (Symbolic Answer):")
        st.latex(r"V = " + latex(volume))

    st.markdown("## 💡 Tips")
    st.info(
        "- Use `x = f(y)` for horizontal shapes (rotation around y-axis).\n"
        "- You can enter exact values like `1/2`, `3/4`, or `sqrt(2)` for bounds."
    )
