import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd
from sympy.abc import x

# -----------------------------
# Helpers
# -----------------------------
def normalize_real_roots(expr: sp.Expr) -> sp.Expr:
    """
    Best-practice normalization for common student input:
    - Converts x**(1/3) into sign(x)*abs(x)**(1/3) so it's REAL for negative x.
    This avoids NumPy's (-0.05)**(1/3) -> NaN behavior.
    """
    r13 = sp.Rational(1, 3)

    def is_one_third_power(e):
        return isinstance(e, sp.Pow) and e.exp == r13

    return expr.replace(
        is_one_third_power,
        lambda e: sp.sign(e.base) * sp.Abs(e.base) ** r13
    )

def safe_float(val):
    """Convert to float if possible; otherwise return np.nan."""
    try:
        out = float(val)
        return out if np.isfinite(out) else np.nan
    except Exception:
        return np.nan

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
      - iter (for intermediate rows)
    """
    fx = sp.lambdify(x, f_expr, modules=["numpy"])
    fpx_expr = sp.diff(f_expr, x)
    fpx = sp.lambdify(x, fpx_expr, modules=["numpy"])

    rows = []
    xn = float(x0)

    for n in range(max_iter):
        fn = safe_float(fx(xn))
        fpn = safe_float(fpx(xn))

        # non-finite checks
        if not np.isfinite(fn) or not np.isfinite(fpn):
            rows.append((n, xn, fn, fpn, np.nan, "non_finite"))
            break

        # convergence check
        if abs(fn) < tol:
            rows.append((n, xn, fn, fpn, xn, "converged"))
            break

        # derivative zero check
        if fpn == 0:
            rows.append((n, xn, fn, fpn, np.nan, "derivative_zero"))
            break

        xnext = xn - fn / fpn
        if not np.isfinite(xnext):
            rows.append((n, xn, fn, fpn, np.nan, "non_finite"))
            break

        rows.append((n, xn, fn, fpn, float(xnext), "iter"))
        xn = float(xnext)

    # If we ran out of iterations after normal steps, set max_iter
    if len(rows) == 0:
        df = pd.DataFrame(columns=["n", "x_n", "f(x_n)", "f'(x_n)", "x_{n+1}", "status"])
        return df, fx, fpx, fpx_expr

    if rows[-1][5] == "iter" and len(rows) == max_iter:
        n, xn, fn, fpn, xnext, _ = rows[-1]
        rows[-1] = (n, xn, fn, fpn, xnext, "max_iter")

    df = pd.DataFrame(rows, columns=["n", "x_n", "f(x_n)", "f'(x_n)", "x_{n+1}", "status"])
    return df, fx, fpx, fpx_expr

# -----------------------------
# Plot helper (robust to NaNs)
# -----------------------------
def make_plot(f, x_n, fpx_n, x_next, x_min, x_max, tangent_half_window):
    X = np.linspace(x_min, x_max, 800)

    # Compute Y safely and mask non-finite values
    Y = f(X)
    Y = np.array(Y, dtype=float)
    mask = np.isfinite(Y)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.axhline(0, linewidth=1)

    if np.any(mask):
        ax.plot(X[mask], Y[mask], linewidth=2, label="f(x)")
    else:
        ax.text(0.5, 0.5, "f(x) is not finite on this window",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.35)
        return fig

    # Current point
    y_n = safe_float(f(x_n))
    if np.isfinite(y_n):
        ax.plot([x_n], [y_n], marker="o", linestyle="None", label="(x_n, f(x_n))")

        # Local tangent
        if np.isfinite(fpx_n):
            Xt = np.linspace(x_n - tangent_half_window, x_n + tangent_half_window, 200)
            Yt = y_n + fpx_n * (Xt - x_n)
            Yt = np.array(Yt, dtype=float)
            m2 = np.isfinite(Yt)
            if np.any(m2):
                ax.plot(Xt[m2], Yt[m2], linewidth=2, label="tangent at x_n")

        # Next point if defined
        if np.isfinite(x_next):
            ax.plot([x_next], [0], marker="o", linestyle="None", label="x_{n+1}")
            ax.vlines([x_n, x_next], ymin=min(0, y_n), ymax=max(0, y_n), linestyles="dotted")
    else:
        # If y_n is non-finite, skip point/tangent and just show curve
        pass

    ax.set_xlim(x_min, x_max)

    # y-limits from finite values only
    y_candidates = [0, np.nanmin(Y[mask]), np.nanmax(Y[mask])]
    if np.isfinite(y_n):
        y_candidates.append(y_n)

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

    # Parse + normalize
    try:
        f_expr = sp.sympify(f_input)
        f_expr = normalize_real_roots(f_expr)  # ✅ fixes x**(1/3) on negatives
    except Exception as e:
        st.error(f"Invalid function: {e}")
        return

    df, f_num, fp_num, fp_expr = newton_history(f_expr, x0, tol, max_iter)

    if len(df) == 0:
        st.error("No iterations computed.")
        return

    # Table display
    df_show = df.copy()
    for col in ["x_n", "f(x_n)", "f'(x_n)", "x_{n+1}"]:
        df_show[col] = pd.to_numeric(df_show[col], errors="coerce").round(6)

    st.subheader("Iteration Table")
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    # Status box (C-only)
    last_status = str(df.iloc[-1]["status"])
    if last_status == "converged":
        st.success("✅ Converged: |f(xₙ)| is below tolerance.")
    elif last_status == "derivative_zero":
        st.error("❌ Did not converge: stopped because f'(xₙ) = 0 (horizontal tangent).")
        st.info("Try a different initial guess x₀ (even a small change can help).")
    elif last_status == "non_finite":
        st.error("❌ Did not converge: values became non-finite (overflow/undefined).")
        st.info("Try a different x₀, or check the function’s domain/real-valued behavior.")
    elif last_status == "max_iter":
        st.error("❌ Did not converge: reached max iterations.")
        st.info("Increase max iterations or choose a better initial guess x₀.")

    # Visualization (with slider guard)
    st.subheader("Visualization")
    if len(df) == 1:
        step = 0
        st.caption("Only one step available.")
    else:
        step = st.slider("Show step n:", 0, len(df) - 1, 0)

    x_n = float(df.loc[step, "x_n"])
    fpn = safe_float(df.loc[step, "f'(x_n)"])
    x_next = df.loc[step, "x_{n+1}"]
    x_next = safe_float(x_next)

    if fpn == 0:
        x_next = np.nan  # no Newton step exists

    # If current f(x_n) isn't finite, avoid crashing and explain
    y_n = safe_float(f_num(x_n))
    if not np.isfinite(y_n):
        st.warning(
            "Cannot plot this step because f(xₙ) is not a real finite number.\n\n"
            "Tip: For cube roots, prefer `cbrt(x)` or use the built-in normalization of `x**(1/3)` "
            "with a real-valued cube-root form."
        )
        return

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
