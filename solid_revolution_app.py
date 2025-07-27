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
function_mode = st.sidebar.selectbox("Function setup", ["One Function", "Two Functions"])
top_expr = st.sidebar.text_input("Top function f(x):", "x")
bottom_expr = st.sidebar.text_input("Bottom function g(x):", "x**2") if function_mode == "Two Functions" else "0"
method = st.sidebar.selectbox("Method", ["Disk/Washer", "Shell"])
axis = st.sidebar.selectbox("Axis of rotation", ["x-axis", "y-axis"])
a = Rational(st.sidebar.text_input("Start of interval a", "0"))
b = Rational(st.sidebar.text_input("End of interval b", "1"))
show_3d = st.sidebar.checkbox("Show 3D Visualization", True)
compute = st.sidebar.button("🔄 Compute and Visualize")

# --- Variables ---
x = symbols('x')
f_expr = sympify(top_expr)
g_expr = sympify(bottom_expr)

def parse(expr):
    return lambda x_val: eval(expr, {"x": x_val, "np": np})

# --- Region Plot ---
def plot_region():
    fx = parse(top_expr)
    gx = parse(bottom_expr)
    xs = np.linspace(float(a), float(b), 400)
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

# --- Volume with Steps ---
def display_formula():
    st.markdown("### 📘 Volume Formula")

    if method == "Disk/Washer" and axis == "x-axis":
        f_sq = simplify(f_expr**2)
        g_sq = simplify(g_expr**2)
        integrand = simplify(f_sq - g_sq)
        definite_integral = integrate(integrand, (x, a, b))
        volume = pi * definite_integral

        st.markdown("#### Step 1: Square the functions")
        st.latex(r"f(x)^2 = " + latex(f_sq))
        st.latex(r"g(x)^2 = " + latex(g_sq))

        st.markdown("#### Step 2: Subtract the squares")
        st.latex(r"f(x)^2 - g(x)^2 = " + latex(integrand))

        st.markdown("#### Step 3: Set up the definite integral")
        st.latex(r"\int_{%s}^{%s} \left[%s\right] dx" % (latex(a), latex(b), latex(integrand)))

        st.markdown("#### Step 4: Integrate and multiply by π")
        st.latex(r"\int = " + latex(definite_integral))
        st.latex(r"V = \pi \cdot " + latex(definite_integral) + r" = " + latex(volume))

        return volume

    elif method == "Shell" and axis == "y-axis":
        diff = simplify(f_expr - g_expr)
        shell_expr = simplify(x * diff)
        definite_integral = integrate(shell_expr, (x, a, b))
        volume = 2 * pi * definite_integral

        st.markdown("#### Step 1: Subtract the functions")
        st.latex(r"f(x) - g(x) = " + latex(diff))

        st.markdown("#### Step 2: Multiply by x")
        st.latex(r"x \cdot (f(x) - g(x)) = " + latex(shell_expr))

        st.markdown("#### Step 3: Set up the definite integral")
        st.latex(r"\int_{%s}^{%s} %s \, dx" % (latex(a), latex(b), latex(shell_expr)))

        st.markdown("#### Step 4: Integrate and multiply by 2π")
        st.latex(r"\int = " + latex(definite_integral))
        st.latex(r"V = 2\pi \cdot " + latex(definite_integral) + r" = " + latex(volume))

        return volume

    st.warning("This method and axis combination is not supported.")
    return None

# --- Main App ---
if compute:
    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.subheader("✏️ Region Bounded by Curves")
        plot_region()

    with col2:
        st.subheader("📊 3D Visualization")
        if show_3d:
            st.info("3D available for educational visualization (experimental).")

    volume = display_formula()
    if volume is not None:
        st.success("✅ Final Answer (Exact Volume)")
        st.latex(r"V = " + latex(volume))

    st.markdown("## 💡 Tips")
    st.info(
        "- Use expressions like `x`, `x**2`, `sqrt(x)`, etc.\n"
        "- Fractions like `1/3` preserve exact symbolic results.\n"
        "- Works best with continuous and integrable functions over [a, b]."
    )
