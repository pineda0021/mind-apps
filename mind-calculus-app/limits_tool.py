# -*- coding: utf-8 -*-
"""Limits tool: import limits_tool and call limits_tool.run()."""

import json
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from sympy.calculus.util import continuous_domain

x = sp.Symbol("x", real=True)

# A recursive-descent JavaScript parser: LaTeX -> a small expression tree.
# No JavaScript eval(), Python eval(), or sympify(student_text) is used.
PARSER_JS = r"""
function parseLatex(source) {
  if (!source.trim() || source.length > 300)
    throw Error('Enter an expression with 1–300 characters.');
  const s = source.replace(/\\(?:left|right)\b/g, '')
    .replace(/\\[,!;: ]/g, ' ')
    .replace(/\\operatorname\{(sin|cos|tan|ln|log|exp|abs)\}/g, '\\$1');
  const tokens = [];
  const pattern = /\s+|\\[a-zA-Z]+|(?:\d+(?:\.\d*)?|\.\d+)|[xe+\-*/^(){}|]/y;
  let pos = 0;
  while (pos < s.length) {
    pattern.lastIndex = pos;
    const m = pattern.exec(s);
    if (!m) throw Error('Unsupported character near: ' + s.slice(pos, pos + 12));
    pos = pattern.lastIndex;
    if (!/^\s+$/.test(m[0])) tokens.push(m[0]);
  }
  let i = 0, depth = 0;
  const peek = () => tokens[i];
  function take(t) { if (peek() === t) { i++; return true; } return false; }
  function expect(t) { if (!take(t)) throw Error('Expected ' + t); }
  const funcs = ['sin', 'cos', 'tan', 'ln', 'log', 'exp', 'abs'];
  function startsAtom(t) {
    return t && (/^[\d.]/.test(t) || ['x','e','(','{','|','\\pi','\\frac','\\sqrt'].includes(t)
      || funcs.some(f => t === '\\' + f));
  }
  function group() { expect('{'); const v = sum('}'); expect('}'); return v; }
  function atom(stop) {
    if (++depth > 25) throw Error('Please use a shorter expression.');
    let v, t = tokens[i++];
    if (t && /^[\d.]/.test(t)) v = ['num', t];
    else if (t === 'x' || t === 'e' || t === '\\pi') v = [t === '\\pi' ? 'pi' : t];
    else if (t === '(' || t === '{') {
      const end = t === '(' ? ')' : '}'; v = sum(end); expect(end);
    } else if (t === '|') { v = ['abs', sum('|')]; expect('|'); }
    else if (t === '\\frac') v = ['div', group(), group()];
    else if (t === '\\sqrt') v = ['sqrt', group()];
    else if (t && funcs.includes(t.slice(1)) && t[0] === '\\') {
      const f = t.slice(1);
      const exponent = take('^') ? atom(stop) : null;
      v = [f, ['(', '{'].includes(peek()) ? atom(stop) : unary(stop)];
      if (exponent) v = ['pow', v, exponent];
    } else throw Error('Expected a number, x, or a supported function.');
    depth--; return v;
  }
  function power(stop) {
    const a = atom(stop);
    return take('^') ? ['pow', a, unary(stop)] : a;
  }
  function unary(stop) {
    if (take('+')) return unary(stop);
    if (take('-')) return ['neg', unary(stop)];
    return power(stop);
  }
  function product(stop) {
    let a = unary(stop);
    while (peek() && peek() !== stop) {
      if (take('*') || take('\\cdot') || take('\\times')) a = ['mul', a, unary(stop)];
      else if (take('/') || take('\\div')) a = ['div', a, unary(stop)];
      else if (startsAtom(peek())) a = ['mul', a, unary(stop)];
      else break;
    }
    return a;
  }
  function sum(stop) {
    let a = product(stop);
    while (peek() && peek() !== stop) {
      if (take('+')) a = ['add', a, product(stop)];
      else if (take('-')) a = ['sub', a, product(stop)];
      else break;
    }
    return a;
  }
  const tree = sum(null);
  if (i !== tokens.length) throw Error('Unexpected input: ' + peek());
  return tree;
}
"""

EDITOR_JS = PARSER_JS + r"""
function loadScript(url) {
  return new Promise((resolve, reject) => {
    const el = document.createElement('script'); el.src = url;
    el.onload = resolve; el.onerror = () => reject(Error('Could not load MathQuill.'));
    document.head.appendChild(el);
  });
}
export default function({parentElement, data, key, setStateValue}) {
  const root = parentElement.querySelector('.limits-editor');
  const rows = root.querySelector('.rows'), status = root.querySelector('.status');
  const drafts = window.__limitsDrafts ||= new Map();
  const saved = drafts.get(key) || data.initial.slice();
  let disposed = false;
  const fields = [];
  rows.replaceChildren();
  data.labels.forEach((label, j) => {
    const row = document.createElement('div'); row.className = 'entry';
    // This markup is fixed. Student input is only assigned as values/text.
    row.innerHTML = `<strong class="label"></strong><div class="mathfield"></div>
      <div class="keys"></div><label class="plain-label">LaTeX input
      <input class="plain" type="text" spellcheck="false" maxlength="300"></label>`;
    row.querySelector('.label').textContent = label;
    rows.appendChild(row);
    const input = row.querySelector('.plain'); input.value = saved[j];
    const item = {row, input, mq: null, syncing: false}; fields.push(item);
    input.addEventListener('input', () => {
      saved[j] = input.value; drafts.set(key, saved);
      if (item.mq) {
        item.syncing = true; item.mq.latex(input.value); item.syncing = false;
      }
      status.textContent = 'Expression changed. Select Analyze to update the results.';
    });
  });
  root.querySelector('.analyze').onclick = () => {
    try {
      const latex = fields.map(f => f.input.value);
      const asts = latex.map(parseLatex);
      setStateValue('value', {latex, asts});
      status.textContent = 'Expression submitted.';
    } catch (e) { status.textContent = e.message; }
  };
  if (!window.__limitsMathQuill) {
    window.__limitsMathQuill = (async () => {
      const css = document.createElement('link'); css.rel = 'stylesheet';
      css.href = 'https://cdnjs.cloudflare.com/ajax/libs/mathquill/0.10.1/mathquill.min.css';
      document.head.appendChild(css);
      if (!window.jQuery) await loadScript('https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js');
      if (!window.MathQuill) await loadScript('https://cdnjs.cloudflare.com/ajax/libs/mathquill/0.10.1/mathquill.min.js');
      return window.MathQuill.getInterface(2);
    })();
  }
  window.__limitsMathQuill.then(MQ => {
    if (disposed) return;
    fields.forEach((item, j) => {
      const el = item.row.querySelector('.mathfield');
      const initial = item.input.value;
      const mq = MQ.MathField(el, {
        autoCommands: 'pi sqrt', autoOperatorNames: 'sin cos tan ln log exp abs',
        handlers: {edit: field => {
          if (!item.mq || item.syncing) return;
          item.input.value = field.latex(); saved[j] = field.latex(); drafts.set(key, saved);
          status.textContent = 'Select Analyze to update the results.';
        }}
      });
      item.mq = mq; item.syncing = true; mq.latex(initial); item.syncing = false;
      el.querySelector('textarea')?.setAttribute('aria-label', data.labels[j]);
      [['Fraction','\\frac'],['Power','^'],['Square root','\\sqrt'],
       ['sin','\\sin'],['cos','\\cos'],['ln','\\ln'],['π','\\pi']].forEach(([label, cmd]) => {
        const b = document.createElement('button'); b.type = 'button'; b.textContent = label;
        b.onclick = () => { mq.focus(); mq.cmd(cmd); };
        item.row.querySelector('.keys').appendChild(b);
      });
      mq.reflow();
    });
  }).catch(() => {
    if (!disposed) status.textContent = 'MathQuill did not load. You can still enter LaTeX in the text boxes.';
  });
  return () => { disposed = true; fields.forEach(f => { if (f.mq) f.mq.revert(); }); };
}
"""


@st.cache_resource
def get_editor():
    return st.components.v2.component(
        "limits_mathquill",
        html='''<div class="limits-editor"><div class="rows"></div>
          <button class="analyze" type="button">Analyze</button>
          <p class="status" role="status" aria-live="polite">Enter your function, then select Analyze.</p></div>''',
        css="""
        .limits-editor {font-family: sans-serif; color: var(--st-text-color);}
        .limits-editor .entry {margin-bottom: 18px;}
        .limits-editor .label {display: block; margin-bottom: 8px;}
        .limits-editor .mathfield {display: block; min-height: 55px; padding: 12px;
          border: 2px solid #64748b; border-radius: 6px; font-size: 24px;
          background: white; color: #111827; overflow-x: auto;}
        .limits-editor button {padding: 8px 12px; margin: 6px 5px 6px 0;
          border: 1px solid #64748b; border-radius: 6px; cursor: pointer;
          background: #f1f5f9; color: #111827;}
        .limits-editor button:focus-visible, .limits-editor input:focus-visible
          {outline: 3px solid #2563eb; outline-offset: 2px;}
        .limits-editor .analyze {background: #1d4ed8; color: white; font-weight: bold;}
        .limits-editor .plain-label {display: block; font-size: 14px;}
        .limits-editor .plain {display: block; width: 100%; box-sizing: border-box;
          padding: 9px; border: 1px solid #64748b; border-radius: 5px; margin-top: 4px;}
        .limits-editor .status {font-size: 14px; min-height: 20px;}
        """,
        js=EDITOR_JS,
        isolate_styles=False,
    )


def build_expression(tree, depth=0):
    """Validate the browser tree and preserve each subexpression's real domain."""
    if depth > 25 or not isinstance(tree, list) or not tree:
        raise ValueError("Please use a shorter, supported expression.")
    op, *items = tree
    if op == "num":
        if len(items) != 1 or not isinstance(items[0], str) or not re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)", items[0]):
            raise ValueError("Invalid number.")
        if len(items[0]) > 20:
            raise ValueError("Use numbers with at most 20 digits.")
        return sp.Rational(items[0]), sp.S.Reals
    constants = {"x": x, "e": sp.E, "pi": sp.pi}
    if op in constants and not items:
        return constants[op], sp.S.Reals
    arities = {**dict.fromkeys(["add", "sub", "mul", "div", "pow"], 2),
               **dict.fromkeys(["neg", "sqrt", "sin", "cos", "tan", "ln", "log", "exp", "abs"], 1)}
    if op not in arities or len(items) != arities[op]:
        raise ValueError("Unsupported expression. Use x and the listed functions.")
    children = [build_expression(t, depth + 1) for t in items]
    a = [c[0] for c in children]
    domain = sp.Intersection(*(c[1] for c in children))
    if op == "add": expr = sp.Add(*a, evaluate=False)
    elif op == "sub": expr = sp.Add(a[0], sp.Mul(-1, a[1], evaluate=False), evaluate=False)
    elif op == "mul": expr = sp.Mul(*a, evaluate=False)
    elif op == "div":
        expr = sp.Mul(a[0], sp.Pow(a[1], -1, evaluate=False), evaluate=False)
        domain -= sp.solveset(a[1], x, domain=sp.S.Reals)
    elif op == "pow":
        if a[1].is_number and (abs(a[1]) > 100) is sp.S.true:
            raise ValueError("Use exponents between -100 and 100.")
        expr = sp.Pow(*a, evaluate=False)
    elif op == "neg": expr = sp.Mul(-1, a[0], evaluate=False)
    elif op == "sqrt": expr = sp.Pow(a[0], sp.Rational(1, 2), evaluate=False)
    elif op == "log": expr = sp.log(a[0], 10, evaluate=False)
    else:
        fn = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
              "ln": sp.log, "exp": sp.exp, "abs": sp.Abs}[op]
        expr = fn(a[0], evaluate=False)
    try:
        domain = domain.intersect(continuous_domain(expr, x, sp.S.Reals))
        if domain.has(sp.ConditionSet): raise ValueError()
    except (NotImplementedError, ValueError):
        raise ValueError("The real domain could not be determined. Try a simpler expression.")
    return expr, domain


def branch_at(branches, a, boundary, side="+"):
    if boundary is None: return branches[0]
    return branches[0 if a < boundary or (a == boundary and side == "-") else 1]


def finite_real(v):
    return (isinstance(v, sp.Basic) and v.is_number is True
            and not v.has(sp.AccumBounds) and v.is_real is True and v.is_finite is True)


def point_value(branch, a):
    expr, domain = branch
    try:
        if domain.contains(a) is not sp.S.true: return None
        v = sp.simplify(expr.subs(x, a))
        return v if finite_real(v) else None
    except Exception:
        return None


def side_limit(branch, a, side):
    expr, domain = branch
    try:
        interval = sp.Interval.open(a - 1, a) if side == "-" else sp.Interval.open(a, a + 1)
        nearby = domain.intersect(interval)
        available = nearby.closure.contains(a)
        if available is sp.S.false: return "No real domain on this side"
        if available is not sp.S.true: return "Undetermined"
        v = sp.limit(expr, x, a, dir=side)
        if v.has(sp.Limit): return "Undetermined"
        if finite_real(v) or v in (sp.oo, -sp.oo): return v
        return "Does not exist"
    except Exception:
        return "Undetermined"


def same(a, b):
    return isinstance(a, sp.Basic) and isinstance(b, sp.Basic) and (a == b or sp.simplify(a - b) == 0)


def show_value(label, value):
    st.markdown(f"**{label}**")
    if value is None: st.write("Undefined")
    elif isinstance(value, str): st.write(value)
    else: st.latex(sp.latex(value))


def make_graph(branches, a, boundary, lo, hi, left, right, fa, use_3d=False):
    fig = go.Figure()
    cuts = {lo, hi}
    for p in (float(a), float(boundary) if boundary is not None else None):
        if p is not None and lo < p < hi: cuts.add(p)
    for _, domain in branches:
        try:
            edges = domain.boundary.intersect(sp.Interval(lo, hi))
            if isinstance(edges, sp.FiniteSet): cuts.update(float(p) for p in edges)
        except (NotImplementedError, TypeError): pass
    cuts = sorted(cuts)
    for j, (start, end) in enumerate(zip(cuts, cuts[1:])):
        mid = sp.Rational(str((start + end) / 2))
        expr, domain = branch_at(branches, mid, boundary)
        if domain.contains(mid) is sp.S.false: continue
        # Separate traces prevent a line across the approach point, branch boundary, or known pole.
        xx = np.linspace(start, end, max(30, int(700 * (end-start)/(hi-lo))))[1:-1]
        try:
            f = sp.lambdify(x, expr, modules="numpy")
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
