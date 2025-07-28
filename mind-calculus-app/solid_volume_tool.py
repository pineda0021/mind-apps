# solid_volume_tool.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import sympy as sp
from sympy.abc import x

def parse_expr(expr_str):
    expr_str = expr_str.replace('^', '**').replace('sqrt', 'sp.sqrt')
    return sp.sympify(expr_str)

def make_3d_surface_diskwasher(fx, gx, xs):
    fig = go.Figure()
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        x_mid = (x0 + x1) / 2
        r_outer = fx(x_mid)
        r_inner = gx(x_mid)
        theta = np.linspace(0, 2 * np.pi, 30)
        T, R = np.meshgrid(theta, np.linspace(r_inner, r_outer, 2))
        X = x_mid * np.ones_like(R)
        Y = R * np.cos(T)
        Z = R * np.sin(T)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.6, colorscale='blues'))
    return fig

def make_3d_surface_shell(fx, gx, xs):
    fig = go.Figure()
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        h = fx((x0 + x1) / 2) - gx((x0 + x1) / 2)
        r = (x0 + x1) / 2
        theta = np.linspace(0, 2 * np.pi, 30)
        T, H = np.meshgrid(theta, np.linspace(0, h, 2))
        X = r * np.cos(T)
        Y = H
        Z = r * np.sin(T)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.6, colorscale='blues'))
    return fig

def run():
    st.subheader("🧊 Solid of Revolution Tool")
    st.markdown("""
    Visualize and compute volumes of solids formed by rotating a region around the x- or y-axis using the Disk/Washer or Shell method.
    """)

    method = st.selectbox("Method:", ["Disk/Washer", "Shell"])
    axis = st.selectbox("Axis of Rotation:", ["x-axis", "y-axis"])
    top_expr = st.text_input("Top/Outer function:", "x")
    bottom_expr = st.text_input("Bottom/Inner function:", "x**2") if method == "Disk/Washer" else st.text_input("Bottom function (for height):", "0")
    a = st.number_input("Start of interval a:", value=0.0)
    b = st.number_input("End of interval b:", value=1.0)

    try:
        f = parse_expr(top_expr)
        g = parse_expr(bottom_expr)
    except Exception as e:
        st.error(f"Function parsing error: {e}")
        return

    # Symbolic setup
    if method == "Disk/Washer" and axis == "x-axis":
        integrand = sp.pi * (f ** 2 - g ** 2)
        formula = rf"V = \pi \int_{{{a}}}^{{{b}}} \left[ {sp.latex(f)}^2 - {sp.latex(g)}^2 \right] \, dx"
        volume = sp.integrate(integrand, (x, a, b))
    elif method == "Shell" and axis == "y-axis":
        integrand = 2 * sp.pi * x * (f - g)
        formula = rf"V = 2\pi \int_{{{a}}}^{{{b}}} x({sp.latex(f)} - {sp.latex(g)}) \, dx"
        volume = sp.integrate(integrand, (x, a, b))
    else:
        st.warning("Only x-axis for Disk/Washer and y-axis for Shell are currently supported.")
        return

    # Display formula and volume
    st.markdown("### 📘 Volume Formula")
    st.latex(formula)

    st.markdown("### 📝 Integration Steps")
    st.latex(r"\text{Integrand: }" + sp.latex(integrand))
    st.latex(r"\text{Integral result: }" + sp.latex(volume))
    st.success(f"Exact volume: {sp.latex(sp.simplify(volume))}")

    # 3D Visualization
    if st.checkbox("🔭 Show 3D Visualization"):
        f_lambdified = sp.lambdify(x, f, modules=["numpy"])
        g_lambdified = sp.lambdify(x, g, modules=["numpy"])
        xs = np.linspace(a, b, 20)

        if method == "Disk/Washer" and axis == "x-axis":
            fig = make_3d_surface_diskwasher(f_lambdified, g_lambdified, xs)
        elif method == "Shell" and axis == "y-axis":
            fig = make_3d_surface_shell(f_lambdified, g_lambdified, xs)
        else:
            st.error("3D visualization only supports Disk/Washer around x-axis and Shell around y-axis.")
            return

        fig.update_layout(title="3D Solid of Revolution", height=500,
                          scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'))
        st.plotly_chart(fig)

    # Tips
    st.markdown("## 💡 Interpretation Tip")
    st.info(
        "- **Disk/Washer**: Best for x-axis rotations when the region is vertical.\n"
        "- **Shell**: Ideal for y-axis rotations with vertical slices.\n"
        "- Visualize volume as accumulation of circular or cylindrical slices!"
    )

if __name__ == "__main__":
    run()
