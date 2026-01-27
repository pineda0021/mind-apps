import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd
from sympy.abc import x

# -----------------------------
# Core Newton engine (C-only)
# -----------------------------
def newton_history(f_expr, x0, tol, max_iter):
    """
    Returns:
      df columns: n, x_n, f(x_n), f'(x_n), x_{n+1}, status
    Status is ALWAYS one of:
      - converged
      - derivative_zero
      - non_finite
      - max_iter
    """
    fx = sp.lambdify(x, f_expr, modules=["numpy"])
    fpx_expr = sp.diff(f_expr, x)
    fpx = sp.lambdify(x, fpx_expr, modules=["numpy"])

    rows = []
    xn = float(x0)

    for n in range(max_iter):
        fn = fx(xn)
        fpn = fpx(xn)

        # coerce safely to float when possible
        try:
            fn = float(fn)
            fpn = float(fpn)
        except Exception:
            rows.append((n, xn, np.nan, np.nan, np.nan, "non_finite"))
            break

        # non-finite check
        if (not np.isfinite(fn)) or (not np.isfinite(fpn)):
            rows.append((n, xn, fn, fpn, np.nan, "non_finite"))
            break

        # convergence check (best to check before derivative-zero stop)
        if abs(fn) < tol:
            rows.append((n, xn, fn, fpn, xn, "converged"))
            break

        # derivative zero stop
        if fpn == 0:
            rows.append((n, xn, fn, fpn, np.nan, "derivative_zero"))
            break

        xnext = xn - fn / fpn

        # xnext finite?
        if not np.isfinite(xnext):
            rows.append((n, xn, fn, fpn, np.nan, "non_finite"))
            break

        # normal step
        rows.append((n, xn, fn, fpn, float(xnext), ""))  # temp status
        xn = float(xnext)

    # If we ran out of iterations without a terminal status, set max_iter
    if len(rows) == 0:
        df = pd.DataFrame(columns=["n", "x_n", "f(x_n)", "f'(x_n)", "x_{n+1}", "status"])
        return df, fx, fpx, fpx_expr

    if rows[-1][5] == "":
        # last row was a normal step but loop ended -> max_iter
        n, xn, fn, fpn, xnext, _ = rows[-1]
        rows[-1] = (n, xn, fn, fpn, xnext, "max_iter")

    # Fill status for intermediate normal rows (optional: leave blank or label "iter")
    # Here we label non-terminal normal rows as "iter" for clarity.
    fixed = []
    for i, r in enumerate(rows):
        n, xn, fn, fpn, xnext, status = r
        if status == "":
            status = "iter"
        fixed.append((n, xn, fn, fpn, xnext, status))

    df = pd.DataFrame(fixed, columns=["n", "x_n", "f(x_n)", "f'(x_n)", "x_{n+1}", "status"])
    return df, fx, fpx, fpx_expr

# -----------------------------
# Plot helper (zoomed + local tangent)
# -----------------------------
def make_plot(f, x_n, fpx_n, x_next, x_min, x_max, tangent_half_window):
    X = np.linspace(x_min, x_max, 800)
    Y = f(X)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.axhline(0, linewidth=1)
    ax.plot(X, Y, linewidth=2, label="f(x)")

    y_n = float(f(x_n))
    ax.plot([x_n], [y_n], marker="o", linestyle="None", label="(x_n, f(x_n))")

    # Tangent (local)
    if np.isfinite(fpx_n):
        Xt = np.linspace(x_n - tangent_half_window, x_n + tangent_half_window, 200)
        Yt = y_n + fpx_n * (Xt - x_n)
        ax.plot(Xt, Yt, linewidth=2, label="tangent at x_n")

    # Next point (if defined)
    if np.isfinite(x_next):
        ax.plot([x_next], [0], marker="o", linestyle="None", label="x_{n+1}")
        ax.vlines([x_n, x_next], ymin=min(0, y_n), ymax=max(0, y_n), linestyles="dotted")

    ax.set_xlim(x_min, x_max)

    # y-limits
    y_candidates = [0, np.nanmin(Y), np.nanmax(Y), y_n]
    y_lo, y_hi = float(np.nanmin(y_candidates)), float(np.nanmax(y_candidates))
    pad = 0.10 * (y_hi - y_lo if y_hi != y_lo else 1.0)
    ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.set_title("Newton Step (Zoomed)", fontsize=16)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=10, loc="upper left")
    return fig

# -----------------------------
# Streamlit UI
# -----------------------------
def run():
    st.header("Newton's Method Explorer")
    st.latex(r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}")

    c1, c2 = st.columns(2)
    with c1:
        f_input = st.text_input("Function f(x):", "x**3 + x - 3")
        x0 = st.number_input("Initial guess x0:", value=1.0)
        tol = st.number_input("Tolerance:", value=1e-6, format="%.1e")
    with c2:
        max_iter = st.slider("Max iterations:", 1, 50, 10)
        plot_half_window = st.number_input("Plot half-window (zoom):", value=1.25)
        tangent_half_window = st.number_input("Tangent half-window (local):", value=0.75)

    try:
        f_expr = sp.sympify(f_input)
    except Exception as e:
        st.error(f"Invalid function: {e}")
        return

    df, f_num, fp_num, fp_expr = newton_history(f_expr, x0, tol, max_iter)

    if len(df) == 0:
        st.error("No iterations computed.")
        return

    # Clean table display
    df_show = df.copy()
    for col in ["x_n", "f(x_n)", "f'(x_n)", "x_{n+1}"]:
        df_show[col] = pd.to_numeric(df_show[col], errors="coerce").round(6)

    st.subheader("Iteration Table")
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    # One best-practice status box driven ONLY by final df status (C-only)
    last_status = str(df.iloc[-1]["status"])

    if last_status == "converged":
        st.success("✅ Converged: |f(xₙ)| is below tolerance.")
    elif last_status == "derivative_zero":
        st.error("❌ Did not converge: stopped because f'(xₙ) = 0 (horizontal tangent).")
        st.info("Try a different initial guess x₀ (even a small change can help).")
    elif last_status == "non_finite":
        st.error("❌ Did not converge: values became non-finite (overflow/undefined).")
        st.info("Try an x₀ closer to the root, or check the function’s domain.")
    elif last_status == "max_iter":
        st.error("❌ Did not converge: reached max iterations.")
        st.info("Increase max iterations or choose a better initial guess x₀.")
    else:
        # 'iter' shouldn't be terminal, but handle anyway
        st.warning(f"Stopped with status: {last_status}")

    # Visualization
    st.subheader("Visualization")

    # slider fix when only 1 row
    if len(df) == 1:
        step = 0
        st.caption("Only one step available.")
    else:
        step = st.slider("Show step n:", 0, len(df) - 1, 0)

    x_n = float(df.loc[step, "x_n"])
    fpn = float(df.loc[step, "f'(x_n)"]) if np.isfinite(df.loc[step, "f'(x_n)"]) else np.nan
    x_next = df.loc[step, "x_{n+1}"]
    x_next = float(x_next) if pd.notna(x_next) and np.isfinite(x_next) else np.nan

    # If derivative zero at this step, don't pretend x_{n+1} exists
    if fpn == 0:
        x_next = np.nan

    x_min = x_n - float(plot_half_window)
    x_max = x_n + float(plot_half_window)

    fig = make_plot(
        f=f_num,
        x_n=x_n,
        fpx_n=fpn,
        x_next=x_next,
        x_min=x_min,
        x_max=x_max,
        tangent_half_window=float(tangent_half_window),
    )
    st.pyplot(fig, use_container_width=True)

    # Result summary
    st.subheader("Result")
    st.latex(rf"x \approx {float(df.iloc[-1]['x_n']):.10f}")
    st.latex(rf"f(x) \approx {float(df.iloc[-1]['f(x_n)']):.4e}")
    st.latex(rf"f'(x) = {sp.latex(fp_expr)}")

if __name__ == "__main__":
    run()
