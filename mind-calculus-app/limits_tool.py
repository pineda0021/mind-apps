            with np.errstate(all="ignore"):
                yy = np.broadcast_to(np.asarray(f(xx), dtype=complex), xx.shape).copy()
            yy = np.where((np.abs(yy.imag) < 1e-10) & np.isfinite(yy.real), yy.real, np.nan)
        except Exception:
            continue
        common = dict(mode="lines", name="f(x)", line=dict(color="#2563eb", width=3),
                      showlegend=not any(t.mode == "lines" for t in fig.data), connectgaps=False)
        trace = go.Scatter3d(x=xx, y=np.zeros_like(xx), z=yy, **common) if use_3d else go.Scatter(x=xx, y=yy, **common)
        fig.add_trace(trace)
    marked = []
    for lim in (left, right):
        if finite_real(lim) and not same(lim, fa) and not any(same(lim, z) for z in marked):
            marked.append(lim)
            marker = dict(size=9, symbol="circle-open", color="#dc2626", line=dict(width=2))
            kwargs = dict(mode="markers", marker=marker, name="Open endpoint at a")
            fig.add_trace(go.Scatter3d(x=[float(a)], y=[0], z=[float(lim)], **kwargs) if use_3d
                          else go.Scatter(x=[float(a)], y=[float(lim)], **kwargs))
    if fa is not None:
        kwargs = dict(mode="markers", marker=dict(size=8, color="#111827"), name="f(a)")
        fig.add_trace(go.Scatter3d(x=[float(a)], y=[0], z=[float(fa)], **kwargs) if use_3d
                      else go.Scatter(x=[float(a)], y=[float(fa)], **kwargs))
    fig.update_layout(height=480, margin=dict(l=20, r=20, t=20, b=20))
    if use_3d:
        fig.update_layout(scene=dict(xaxis=dict(title="x", range=[lo, hi]),
                          yaxis=dict(title="", showticklabels=False), zaxis_title="f(x)"))
    else:
        fig.update_layout(xaxis=dict(title="x", range=[lo, hi]), yaxis_title="f(x)")
        fig.add_vline(x=float(a), line_dash="dot", line_color="#64748b")
    return fig


def run():
    st.header("♾️ Limits Visualizer")
    st.write("Enter a function, explore its graph, and compare its one-sided limits.")
    st.caption("Use x as the variable. Trig functions use radians. ln is natural log; log is base 10.")
    if not hasattr(st.components, "v2"):
        st.error('Install Streamlit 1.51 or newer: pip install --upgrade "streamlit>=1.51,<2"')
        return
    piecewise = st.radio("Function type", ["Single expression", "Piecewise: two branches"], horizontal=True).startswith("Piecewise")
    boundary = None
    if piecewise:
        boundary = sp.Rational(str(st.number_input("Branch boundary b", value=2.0, step=0.1)))
        labels, initial = ["Left branch: x < b", "Right branch: x ≥ b"], ["x^2", "3x"]
    else:
        labels, initial = ["Function f(x)"], [r"\frac{x^2-1}{x-1}"]
    st.caption("Type / for a fraction and ^ for a power. Use arrow keys to leave an exponent or denominator. Select Analyze after editing.")
    with st.expander("Examples and supported input"):
        st.code("\\frac{x^2-1}{x-1}\n\\frac{\\sin(x)}{x}\n\\sqrt{x+2}\n|x|\ne^x\n\\ln(x)", language="latex")
        st.write("Supports +, −, multiplication, division, powers, square roots, sin, cos, tan, ln, log, exp, absolute value, e, and π. Use the piecewise option for two branches. Equations, indexed roots, and other LaTeX commands are not supported.")
    result = get_editor()(data={"labels": labels, "initial": initial},
                          key=f"equation_editor_{piecewise}", on_value_change=lambda: None)
    if result.value is None:
        st.info("Select Analyze above to begin.")
        return
    try:
        payload = result.value
        if len(json.dumps(payload)) > 16000 or len(payload["asts"]) != len(labels):
            raise ValueError("Please use a shorter expression.")
        branches = [build_expression(t) for t in payload["asts"]]
    except Exception as exc:
        st.error(f"Please check your expression: {exc}")
        return
    a = sp.Rational(str(st.number_input("Approach x → a", value=2.0 if piecewise else 1.0,
                                        step=0.1, key=f"approach_{piecewise}")))
    c1, c2 = st.columns(2)
    with c1: lo = st.number_input("x-axis minimum", value=float(a)-4, step=0.5)
    with c2: hi = st.number_input("x-axis maximum", value=float(a)+4, step=0.5)
    if lo >= hi:
        st.warning("The x-axis minimum must be less than the maximum.")
        return
    original = branches[0][0] if not piecewise else sp.Piecewise((branches[0][0], x < boundary), (branches[1][0], True), evaluate=False)
    st.subheader("Function being analyzed")
    st.latex("f(x)=" + sp.latex(original))
    left = side_limit(branch_at(branches, a, boundary, "-"), a, "-")
    right = side_limit(branch_at(branches, a, boundary, "+"), a, "+")
    fa = point_value(branch_at(branches, a, boundary), a)
    equal = same(left, right)
    two_sided = left if equal else ("Undetermined" if "Undetermined" in (left, right) else "Does not exist")
    st.subheader("Compare the limits")
    for col, label, val in zip(st.columns(4), ["Left-hand limit", "Right-hand limit", "Two-sided limit", "f(a)"], [left, right, two_sided, fa]):
        with col: show_value(label, val)
    if equal and finite_real(left):
        if same(left, fa): st.success("The two-sided limit equals f(a): the function is continuous at a.")
        else: st.info("Removable discontinuity: both sides approach the same finite value, but f(a) is missing or different.")
    elif equal:
        st.info("Both sides grow without bound in the same direction. There is no finite limit.")
    elif finite_real(left) and finite_real(right):
        st.warning("Jump discontinuity: the one-sided limits are different, so the two-sided limit does not exist.")
    elif two_sided == "Undetermined": st.info("The symbolic calculation is inconclusive. The table and graph are supporting evidence, not a proof.")
    else: st.warning("There is no real two-sided limit. Compare the one-sided results above.")
    st.subheader("Interactive graph")
    st.plotly_chart(make_graph(branches, a, boundary, lo, hi, left, right, fa), use_container_width=True)
    st.caption("Open circles show excluded limiting endpoints at a; a filled point shows f(a). Graphs are numerical samples.")
    if st.checkbox("Show 3D view"):
        st.caption("The same curve is placed in the plane y = 0.")
        st.plotly_chart(make_graph(branches, a, boundary, lo, hi, left, right, fa, True), use_container_width=True)
    st.subheader("Table of nearby values")
    rows = []
    for side, sign in [("From the left", -1), ("From the right", 1)]:
        for k in (1, 2, 3, 4):
            p = a + sign * sp.Rational(1, 10**k)
            v = point_value(branch_at(branches, p, boundary), p)
            rows.append({"Approach": side, "x": str(sp.N(p, 12)),
                         "f(x)": str(sp.N(v, 10)) if v is not None else "Undefined"})
    st.table(rows)
    with st.expander("Show the algebra and limit reasoning", expanded=True):
        st.write("1. Keep the domain of the original expression, even when factors cancel.")
        for j, (expr, domain) in enumerate(branches):
            if piecewise: st.write(labels[j])
            st.latex(r"\text{Original branch domain: }" + sp.latex(domain))
            st.latex(r"\text{Factored: }" + sp.latex(sp.factor(expr)))
            st.latex(r"\text{Simplified: }" + sp.latex(sp.simplify(expr)))
        if piecewise: st.write("Each branch also follows its displayed condition, x < b or x ≥ b.")
        st.write("2. Approach a separately from the left and right, using the applicable branch.")
        st.write("3. A finite two-sided limit exists when both one-sided limits have the same finite value.")
        st.write("4. Compare that limit with f(a) to check continuity.")
    st.subheader("Reflection")
    st.text_area("What do the graph, table, and one-sided limits tell you? Does f(a) equal the limit?")
    st.caption("Your reflection stays in this session; it is not submitted to your teacher.")


if __name__ == "__main__":
    st.set_page_config(page_title="Limits Visualizer", page_icon="♾️", layout="wide")
    run()
