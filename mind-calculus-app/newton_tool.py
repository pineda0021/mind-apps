import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd
from sympy.abc import x

def newton_history(f_expr, x0, tol, max_iter):
    fx = sp.lambdify(x, f_expr, modules=["numpy"])
    fpx_expr = sp.diff(f_expr, x)
    fpx = sp.lambdify(x, fpx_expr, modules=["numpy"])

    rows = []
    xn = float(x0)

    for n in range(max_iter):
        fn = float(fx(xn))
        fpn = float(fpx(xn))

        # stop reasons
        if not np.isfinite(fn) or not np.isfinite(fpn):
            rows.append((n, xn, fn, fpn, np.nan, "non-finite"))
            break
        if fpn == 0:
            rows.append((n, xn, fn, fpn, np.nan, "f'(x)=0"))
            break

        xnext = xn - fn / fpn
        rows.append((n, xn, fn, fpn, xnext, ""))

        if abs(fn) < tol:
            rows[-1] = (n, xn, fn, fpn, xnext, "converged")
            break

        xn = float(xnext)

    df = pd.DataFrame(rows, columns=["n", "x_n", "f(x_n)", "f'(x_n)", "x_{n+1}", "status"])
    return df, fx, fpx, fpx_expr

def make_plot(f, x_n, fpx_n, x_next, x_min, x_max, local_halfwidth):
    # Function curve
    X = np.linspace(x_min, x_max, 800)
    Y = f(X)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.axhline(0, linewidth=1)

    ax.plot(X, Y, linewidth=2, label="f(x)")

    # current point
    y_n = float(f(x_n))
    ax.plot([x_n], [y_n], marker="o", linestyle="None", label="(x_n, f(x_n))")

    # Tangent line (LOCAL only, so it doesn't blow up the scale)
    if np.isfinite(fpx_n):
        Xt = np.linspace(x_n - local_halfwidth, x_n + local_halfwidth, 200)
        Yt = y_n + fpx_n * (Xt - x_n)
        ax.plot(Xt, Yt, linewidth=2, label="tangent at x_n")

    # x_{n+1}
    if np.isfinite(x_next):
        ax.plot([x_next], [0], marker="o", linestyle="None", label="x_{n+1}")
        ax.vlines([x_n, x_next], ymin=min(0, y_n), ymax=max(0, y_n), linestyles="dotted")

    # limits
    ax.set_xlim(x_min, x_max)

    # choose y-limits from the function values in view + important points
    y_candidates = [0, np.nanmin(Y), np.nanmax(Y), y_n]
    if np.isfinite(x_next):
        y_candidates.append(0)

    y_lo, y_hi = float(np.nanmin(y_candidates)), float(np.nanmax(y_candidates))
    pad = 0.10 * (y_hi - y_lo if y_hi != y_lo else 1.0)
    ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.set_title("Newton Step (Zoomed)", fontsize=16)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=10, loc="upper left")

    return fig

def run():
    st.header("Newton's Method Explorer")
    st.latex(r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}")

    # Inputs
    c1, c2 = st.columns(2)
    with c1:
        f_input = st.text_input("Function f(x):", "x**3 + x - 3")
        x0 = st.number_input("Initial guess x0:", value=1.0)
        tol = st.number_input("Tolerance:", value=1e-6, format="%.1e")
    with c2:
        max_iter = st.slider("Max iterations:", 1, 50, 10)
        # Better default: smaller window for nicer visuals
        plot_half_window = st.number_input("Plot half-window (zoom):", value=1.25)
        tangent_half_window = st.number_input("Tangent half-window (local):", value=0.75)

    # Parse function
    try:
        f_expr = sp.sympify(f_input)
    except Exception as e:
        st.error(f"Invalid function: {e}")
        return

    # Compute
    df, f_num, fp_num, fp_expr = newton_history(f_expr, x0, tol, max_iter)

    # Format table
    df_show = df.copy()
    for col in ["x_n", "f(x_n)", "f'(x_n)", "x_{n+1}"]:
        df_show[col] = pd.to_numeric(df_show[col], errors="coerce").round(6)

    st.subheader("Iteration Table")
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    # Plot step selector
    st.subheader("Visualization")
    if len(df) == 0:
        st.warning("No iterations computed.")
        return

    step = st.slider("Show step n:", 0, len(df) - 1, 0)

    x_n = float(df.loc[step, "x_n"])
    fn = float(df.loc[step, "f(x_n)"])
    fpn = float(df.loc[step, "f'(x_n)"])
    x_next = df.loc[step, "x_{n+1}"]
    x_next = float(x_next) if pd.notna(x_next) else np.nan

    x_min = x_n - float(plot_half_window)
    x_max = x_n + float(plot_half_window)

    fig = make_plot(
        f=f_num,
        x_n=x_n,
        fpx_n=fpn,
        x_next=x_next,
        x_min=x_min,
        x_max=x_max,
        local_halfwidth=float(tangent_half_window),
    )
    st.pyplot(fig, use_container_width=True)

    # Result summary
    st.subheader("Result")
    last = df.iloc[-1]
    st.latex(rf"x \approx {float(last['x_n']):.10f}")
    st.latex(rf"f(x) \approx {float(last['f(x_n)']):.4e}")
    st.latex(rf"f'(x) = {sp.latex(fp_expr)}")

    if "converged" in str(last["status"]):
        st.success("Converged (|f(x)| < tolerance).")
    elif str(last["status"]).strip():
        st.warning(f"Stopped: {last['status']}")
    else:
        st.info("Stopped: reached max iterations.")

if __name__ == "__main__":
    run()
