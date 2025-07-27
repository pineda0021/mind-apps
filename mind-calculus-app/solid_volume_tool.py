# solid_volume_tool.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import sympy as sp
from sympy.abc import x

def run():
    st.subheader("🧊 Solid of Revolution Tool")
    st.markdown("""
    Visualize and compute volumes of solids formed by rotating a region around the x- or y-axis using the Disk/Washer or Shell method.
    """)

    method = st.selectbox("Method:", ["Disk/Washer", "Shell"])
    axis = st.selectbox("Axis of Rotation:", ["x-axis", "y-axis"])
    top_expr = st.text_input("Top/Outer function:", "x")
    bottom_expr = st.text_input("Bottom/Inner function:", "x**2") if method == "Disk/Washer" else "0"
    a = st.number_input("Start of interval a:", value=0.0)
    b = st.number_input("End of interval b:", value=1.0)

    try:
        f = sp.sympify(top_expr)
        g = sp.sympify(bottom_expr)
    except Exception as e:
        st.error(f"Function parsing error: {e}")
        return

    # Setup integral
    if method == "Disk/Washer" and axis == "x-axis":
        integrand = sp.pi * (f**2 - g**2)
        formula = rf"V = \pi \int_{{{a}}}^{{{b}}} \left[ {sp.latex(f)}^2 - {sp.latex(g)}^2 \right] dx"
        volume = sp.integrate(integrand, (x, a, b))
    elif method == "Shell" and axis == "y-axis":
        integrand = 2 * sp.pi * x * (f - sp.sympify(bottom_expr))
        formula = rf"V = 2\pi \int_{{{a}}}^{{{b}}} x({sp.latex(f)} - {sp.latex(bottom_expr)}) dx"
        volume = sp.integrate(integrand, (x, a, b))
    else:
        st.warning("Only x-axis for Disk/Washer and y-axis for Shell are supported.")
        return

    st.markdown("### 📘 Formula")
    st.latex(formula)

    st.markdown("### 📝 Steps")
    st.latex(r"\text{Integrand: }" + sp.latex(integrand))
    st.latex(r"\text{Integral result: }" + sp.latex(volume))
    st.success(f"Exact volume: {sp.latex(volume)}")

    # 3D Visualization
    if st.checkbox("🔭 Show 3D Visualization"):
        def parse(expr):
            return lambda x: eval(str(expr), {"x": x, "np": np})

        fx = parse(top_expr)
        gx = parse(bottom_expr)
        xs = np.linspace(a, b, 20)
        fig = go.Figure()

        if method == "Disk/Washer" and axis == "x-axis":
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
        elif method == "Shell" and axis == "y-axis":
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

        fig.update_layout(title="3D Solid of Revolution", height=500,
                          scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'))
        st.plotly_chart(fig)

    st.markdown("## 💡 Interpretation Tip")
    st.info(
        "- **Disk/Washer**: Best for x-axis rotations when the region is vertical.\n"
        "- **Shell**: Ideal for y-axis rotations with vertical slices.\n"
        "- Visualize volume as accumulation of circular or cylindrical slices!"
    )

if __name__ == "__main__":
    run()
