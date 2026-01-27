import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.abc import x

def newton_steps(f_expr, x0, tol, max_iter):
    """Return iteration history for Newton's Method."""
    fx = sp.lambdify(x, f_expr, modules=["numpy"])
    fpx_expr = sp.diff(f_expr, x)
    fpx = sp.lambdify(x, fpx_expr, modules=["numpy"])

    rows = []
    xn = float(x0)

    for n in range(max_iter):
        fn = float(fx(xn))
        fpn = float(fpx(xn))

        rows.append({
            "n": n,
            "x_n": xn,
            "f(x_n)": fn,
            "f'(x_n)": fpn
        })

        if abs(fn) < tol:
            break
        if fpn == 0 or not np.isfinite(fpn):
            break

        xn = xn - fn / fpn

        if not np.isfinite(xn):
            break

    # Add x_{n+1} column (computed from previous row when possible)
    for i in range(len(rows) - 1):
        rows[i]["x_{n+1}"] = rows[i + 1]["x_n"]
    rows[-1]["x_{n+1}"] = np.nan

    return rows, fx, sp.lambdify(x, sp.diff(f_expr, x), modules=["numpy"]), sp.diff(f_expr, x)

def plot_newton(f, x_n, fpx_n, x_next, x_min, x_max, title="Newton's Method"):
    X = np.linspace(x_min, x_max, 800)
    Y = f(X)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, linewidth=1)
    ax.plot(X, Y, "k", label="f(x)")

    # current point
    y_n = f(x_n)
    ax.plot([x_n], [y_n], marker="o", linestyle="None", label="(x_n, f(x_n))")

    # tangent line at x_n: y = f(x_n) + f'(x_n)(x - x_n)
    if np.isfinite(fpx_n):
        T = y_n + fpx_n * (X - x_n)
        ax.plot(X, T, label="tangent at x_n")

    # show where tangent hits x-axis (x_{n+1})
    if np.isfinite(x_next):
        ax.plot([x_next], [0], marker="o", linestyle="None", label="x_{n+1}")
        ax.vlines([x_n, x_next], ymin=min(0, y_n), ymax=max(0, y_n), linestyles="dotted")

    ax.set_xlim(x_min, x_max)

    # y-limits with padding
    candidates = [0, np.nanmin(Y), np.nanmax(Y), y_n]
    y_lo, y_hi = float(np.nanmin(candidates)), float(np.nanmax(candidates))
    pad = 0.08 * (y_hi - y_lo if y_hi != y_lo else 1.0)
    ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend()
    return fig

def run():
    st.title("Newton's Method Explorer")

    st.markdown(r"""
Newton's Method iteration:
\[
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
\]
""")

    col1, col2 = st.columns(2)
    with col1:
        f_input = st.text_input("Function f(x):", "x**2 - 2")
        x0 = st.number_input("Initial guess x0:", value=1.0)
        tol = st.number_input("Tolerance:", value=1e-6, format="%.1e")
    with col2:
        max_iter = st.slider("Max iterations:", 1, 50, 10)
        x_window = st.number_input("Plot half-window (around x0):", value=5.0)

    try:
        f_expr = sp.sympify(f_input)
        rows, f_num, fp_num, fp_expr = newton_steps(f_expr, x0, tol, max_iter)
    except Exception as e:
        st.error(f"Invalid function: {e}")
        return

    st.subheader("Iteration Table")
    st.dataframe(rows, use_container_width=True)

    # Determine which step to display on plot
    st.subheader("Visualization")
    step_index = st.slider("Show step n:", 0, max(0, len(rows) - 1), min(0, len(rows) - 1))

    x_n = rows[step_index]["x_n"]
    fpx_n = rows[step_index]["f'(x_n)"]

    # compute x_{n+1} for plotting if possible
    fn = rows[step_index]["f(x_n)"]
    x_next = np.nan
    if np.isfinite(fpx_n) and fpx_n != 0:
        x_next = x_n - fn / fpx_n

    fig = plot_newton(
        f=f_num,
        x_n=x_n,
        fpx_n=fpx_n,
        x_next=x_next,
        x_min=float(x0 - x_window),
        x_max=float(x0 + x_window),
        title=f"Newton Step n={step_index}"
    )
    st.pyplot(fig)

    # Report final approximation
    st.subheader("Result")
    last = rows[-1]
    st.latex(rf"x \approx {last['x_n']:.10f}")
    st.latex(rf"f(x) \approx {last['f(x_n)']:.4e}")
    st.latex(rf"f'(x) = {sp.latex(fp_expr)}")

    # Basic stop reason hint
    if abs(last["f(x_n)"]) < tol:
        st.success("Converged (|f(x)| < tolerance).")
    elif last["f'(x_n)"] == 0:
        st.warning("Stopped: derivative became zero.")
    else:
        st.info("Stopped: reached max iterations or numerical issue.")

if __name__ == "__main__":
    run()
