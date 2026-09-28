# -*- coding: utf-8 -*-
"""Derivative tool: import derivative_tool and call derivative_tool.run()."""

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
  const root = parentElement.querySelector('.derivative-editor');
  const rows = root.querySelector('.rows'), status = root.querySelector('.status');
  const drafts = window.__derivativeDrafts ||= new Map();
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
        "derivative_mathquill",
        html='''<div class="derivative-editor"><div class="rows"></div>
          <button class="analyze" type="button">Analyze</button>
          <p class="status" role="status" aria-live="polite">Enter your function, then select Analyze.</p></div>''',
        css="""
        .derivative-editor {font-family: sans-serif; color: var(--st-text-color);}
        .derivative-editor .entry {margin-bottom: 18px;}
        .derivative-editor .label {display: block; margin-bottom: 8px;}
        .derivative-editor .mathfield {display: block; min-height: 55px; padding: 12px;
          border: 2px solid #64748b; border-radius: 6px; font-size: 24px;
          background: white; color: #111827; overflow-x: auto;}
        .derivative-editor button {padding: 8px 12px; margin: 6px 5px 6px 0;
          border: 1px solid #64748b; border-radius: 6px; cursor: pointer;
          background: #f1f5f9; color: #111827;}
        .derivative-editor button:focus-visible, .derivative-editor input:focus-visible
          {outline: 3px solid #2563eb; outline-offset: 2px;}
        .derivative-editor .analyze {background: #1d4ed8; color: white; font-weight: bold;}
        .derivative-editor .plain-label {display: block; font-size: 14px;}
        .derivative-editor .plain {display: block; width: 100%; box-sizing: border-box;
          padding: 9px; border: 1px solid #64748b; border-radius: 5px; margin-top: 4px;}
        .derivative-editor .status {font-size: 14px; min-height: 20px;}
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
        if a[1].has(x):
            if not a[0].has(x):
                if a[0].is_positive is not True:
                    raise ValueError("Use a positive base for a variable exponent.")
            else:
                try:
                    positive_base = sp.solve_univariate_inequality(a[0] > 0, x, relational=False)
                    domain = domain.intersect(positive_base)
                except (NotImplementedError, ValueError):
                    raise ValueError("The positive-base domain could not be determined.")
        expr = sp.Pow(*a, evaluate=False)
    elif op == "neg": expr = sp.Mul(-1, a[0], evaluate=False)
    elif op == "sqrt": expr = sp.Pow(a[0], sp.Rational(1, 2), evaluate=False)
    elif op == "log":
        expr = sp.Mul(sp.log(a[0], evaluate=False), 1 / sp.log(10), evaluate=False)
    else:
        fn = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
              "ln": sp.log, "exp": sp.exp, "abs": sp.Abs}[op]
        expr = fn(a[0], evaluate=False)
    try:
        domain = domain.intersect(continuous_domain(expr, x, sp.S.Reals))
        if domain.has(sp.ConditionSet): raise ValueError()
    except (NotImplementedError, ValueError):
        raise ValueError("The real domain could not be determined. Try a simpler expression.")
    if domain is sp.S.EmptySet:
        raise ValueError("This expression has no supported real domain.")
    return expr, domain



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


def step_by_step_derivation(expr):
    """Return rule explanations and LaTeX without assuming two factors."""
    steps = []

    def text(value):
        steps.append(("text", value))

    def math(value):
        steps.append(("math", value))

    def result(e):
        math(r"\frac{d}{dx}\left[" + sp.latex(e) + r"\right]="
             + sp.latex(sp.simplify(sp.diff(e, x))))

    def visit(e, depth=0):
        if depth > 10 or len(steps) > 100:
            text("Apply the same rules to the remaining inner expression.")
            result(e)
            return
        if not e.has(x):
            text("Constant rule: a quantity independent of x has derivative zero.")
            result(e)
            return
        if e == x:
            text("Identity rule.")
            math(r"\frac{d}{dx}[x]=1")
            return
        if e.is_Add:
            text("Sum/difference rule: differentiate each term separately.")
            for term in e.args:
                visit(term, depth + 1)
            text("Combine the term derivatives.")
            result(e)
            return

        numerator, denominator = sp.fraction(e)
        if denominator.has(x):
            text("Quotient rule. Set u equal to the numerator and v equal to the denominator.")
            math(r"u=" + sp.latex(numerator) + r",\qquad v=" + sp.latex(denominator))
            math(r"\left(\frac{u}{v}\right)'=\frac{u'v-uv'}{v^2},\qquad v\ne0")
            visit(numerator, depth + 1)
            visit(denominator, depth + 1)
            text("Substitute u, v, and their derivatives, then simplify.")
            substituted = sp.Mul(
                sp.Add(
                    sp.Mul(sp.diff(numerator, x), denominator, evaluate=False),
                    sp.Mul(-1, numerator, sp.diff(denominator, x), evaluate=False),
                    evaluate=False,
                ), sp.Pow(denominator, -2, evaluate=False), evaluate=False,
            )
            math(sp.latex(substituted))
            result(e)
            return
        if e.is_Mul:
            constant, dependent = e.as_independent(x, as_Add=False)
            if constant != 1:
                text("Constant multiple rule: keep the constant and differentiate the remaining factor.")
                math(r"\frac{d}{dx}[c\,u(x)]=c\,u'(x),\qquad c=" + sp.latex(constant))
                visit(dependent, depth + 1)
            else:
                u = e.args[0]
                v = sp.Mul(*e.args[1:])
                text("Product rule: group the factors as u and v.")
                math(r"u=" + sp.latex(u) + r",\qquad v=" + sp.latex(v))
                math(r"(uv)'=u'v+uv'")
                visit(u, depth + 1)
                visit(v, depth + 1)
            text("Substitute and simplify.")
            result(e)
            return
        if e.is_Pow:
            base, exponent = e.args
            if not exponent.has(x):
                text("Power rule with the chain rule: multiply by the derivative of the base.")
                math(r"\frac{d}{dx}[u(x)^n]=n\,u(x)^{n-1}u'(x)")
                math(r"u(x)=" + sp.latex(base) + r",\qquad n=" + sp.latex(exponent))
                visit(base, depth + 1)
            elif not base.has(x):
                text("Exponential rule with the chain rule, for a positive constant base b.")
                math(r"\frac{d}{dx}[b^{u(x)}]=b^{u(x)}\ln(b)u'(x)")
                math(r"b=" + sp.latex(base) + r",\qquad u(x)=" + sp.latex(exponent))
                visit(exponent, depth + 1)
            else:
                text("Variable base and exponent: use logarithmic differentiation on the positive-base domain.")
                math(r"y=u(x)^{v(x)},\qquad \ln y=v\ln u")
                math(r"\frac{y'}{y}=v'\ln u+v\frac{u'}{u}")
                math(r"y'=u^v\left(v'\ln u+v\frac{u'}{u}\right)")
                math(r"u=" + sp.latex(base) + r",\qquad v=" + sp.latex(exponent))
                visit(base, depth + 1)
                visit(exponent, depth + 1)
            text("Substitute and simplify.")
            result(e)
            return
        rules = {
            sp.sin: ("Sine and chain rules", r"[\sin u]'=\cos(u)u'"),
            sp.cos: ("Cosine and chain rules", r"[\cos u]'=-\sin(u)u'"),
            sp.tan: ("Tangent and chain rules", r"[\tan u]'=\sec^2(u)u'"),
            sp.exp: ("Exponential and chain rules", r"[e^u]'=e^u u'"),
            sp.log: ("Natural logarithm and chain rules", r"[\ln u]'=\frac{u'}{u},\quad u>0"),
            sp.Abs: ("Absolute value rule away from zeros", r"[|u|]'=\operatorname{sign}(u)u',\quad u\ne0"),
        }
        if e.func in rules:
            title, formula = rules[e.func]
            text(title + ".")
            math(formula)
            math(r"u(x)=" + sp.latex(e.args[0]))
            visit(e.args[0], depth + 1)
            if e.func == sp.Abs:
                text("At a zero of the inner function, check the difference quotient separately.")
        result(e)

    visit(expr)
    return steps


@st.cache_data(show_spinner=False, max_entries=64)
def analyze_function(tree_json):
    expr, domain = build_expression(json.loads(tree_json))
    derivative = sp.simplify(sp.diff(expr, x))
    return expr, domain, derivative


def analyze_point(expr, domain, a):
    # SymPy arguments are not reliably hashable by Streamlit's cache.
    value = point_value((expr, domain), a)
    if value is None:
        return None, "Undefined: f(a) is not real and finite", "Undefined: f(a) is not real and finite", None
    quotient = (expr - value) / (x - a)
    left = side_limit((quotient, domain), a, "-")
    right = side_limit((quotient, domain), a, "+")
    slope = left if same(left, right) and finite_real(left) else None
    return value, left, right, slope


def sample_values(expr, values):
    """Broadcast constants and mask complex or nonfinite values."""
    function = sp.lambdify(x, expr, modules="numpy")
    with np.errstate(all="ignore"):
        yy = np.broadcast_to(np.asarray(function(values), dtype=complex), values.shape)
    return np.where((np.abs(yy.imag) < 1e-10) & np.isfinite(yy.real), yy.real, np.nan)


def plot_curves(expr, derivative, domain, a, lo, hi, value, slope, tangent=False):
    fig = go.Figure()
    interval = sp.Interval(lo, hi)
    cuts = {lo, hi}
    if lo < float(a) < hi:
        cuts.add(float(a))
    derivative_domain = domain
    exact_boundaries = True
    try:
        derivative_domain &= continuous_domain(derivative, x, sp.S.Reals)
    except (NotImplementedError, ValueError):
        # sign() is handled by finding its zeros below.
        exact_boundaries = False
    for d in (domain, derivative_domain):
        try:
            edges = d.boundary.intersect(interval)
            if isinstance(edges, sp.FiniteSet):
                cuts.update(float(p) for p in edges)
            elif edges is not sp.S.EmptySet:
                exact_boundaries = False
        except (NotImplementedError, TypeError, ValueError):
            exact_boundaries = False
    for atom in derivative.atoms(sp.sign, sp.Abs):
        try:
            roots = sp.solveset(atom.args[0], x, domain=interval)
            if isinstance(roots, sp.FiniteSet):
                cuts.update(float(p) for p in roots)
            elif roots is not sp.S.EmptySet:
                exact_boundaries = False
        except (NotImplementedError, TypeError, ValueError):
            exact_boundaries = False
    cuts = sorted(cuts)
    curves = [(expr, domain, "f(x)", "#2563eb")]
    if not tangent:
        curves.append((derivative, derivative_domain, "f′(x)", "#d97706"))
    for function, function_domain, label, color in curves:
        first = True
        for start, end in zip(cuts, cuts[1:]):
            midpoint = sp.Rational(str((start + end) / 2))
            if function_domain.contains(midpoint) is sp.S.false:
                continue
            xx = np.linspace(start, end, max(35, int(900 * (end-start)/(hi-lo))))[1:-1]
            try:
                yy = sample_values(function, xx)
            except Exception:
                continue
            if not np.any(np.isfinite(yy)):
                continue
            fig.add_trace(go.Scatter(
                x=xx, y=yy, mode="lines", name=label, legendgroup=label,
                line=dict(color=color, width=3), showlegend=first, connectgaps=False,
            ))
            first = False
    if value is not None:
        fig.add_trace(go.Scatter(
            x=[float(a)], y=[float(value)], mode="markers", name="(a, f(a))",
            marker=dict(size=9, color="#111827"),
        ))
    if slope is not None and not tangent:
        fig.add_trace(go.Scatter(
            x=[float(a)], y=[float(slope)], mode="markers", name="(a, f′(a))",
            marker=dict(size=9, color="#d97706"),
        ))
    if tangent and slope is not None:
        xx = np.linspace(lo, hi, 100)
        fig.add_trace(go.Scatter(
            x=xx, y=float(slope)*(xx-float(a))+float(value),
            mode="lines", name="Tangent line", line=dict(color="#dc2626", dash="dash", width=2),
        ))
    fig.add_vline(x=float(a), line_dash="dot", line_color="#94a3b8")
    fig.update_layout(
        xaxis=dict(title="x", range=[lo, hi]), yaxis_title="y", height=470,
        margin=dict(l=20, r=20, t=20, b=20), hovermode="x unified",
    )
    return fig, exact_boundaries


def run():
    st.header("𝒅𝒚/𝒅𝒙 Derivative Visualizer")
    st.write("Enter a function, follow its derivative rules, and connect the derivative to tangent and secant slopes.")
    if not hasattr(st.components, "v2"):
        st.error('This editor requires Streamlit 1.51 or newer. Update the Streamlit version in requirements.txt.')
        return
    st.subheader("Enter a function")
    st.caption("Use x as the variable. Type / for a fraction and ^ for a power. Use arrow keys to leave an exponent or denominator.")
    with st.expander("Examples and input help"):
        st.code("x^2+3x+2\n\\frac{x^2+1}{x-1}\n\\sin(x^2)\n2^x\nx^x\n|x|\n\\sqrt{x}\n\\ln(x)", language="latex")
        st.write("Supports fractions, powers, square roots, sin, cos, tan, exp, ln, log, absolute value, e, and π. Trig functions use radians. ln is natural logarithm; log is base 10. For variable exponents, the app uses the positive-base real domain.")
        st.write("Use LaTeX input for pasting. Equations, piecewise expressions, and indexed roots are not supported in this tool.")
    result = get_editor()(
        data={"labels": ["Function f(x)"], "initial": ["x^2+3x+2"]},
        key="derivative_equation", on_value_change=lambda: None,
    )
    if result.value is None:
        st.info("Select Analyze above to begin.")
        return
    try:
        payload = result.value
        if len(json.dumps(payload)) > 16000 or len(payload["asts"]) != 1:
            raise ValueError("Please use a shorter expression.")
        with st.spinner("Finding the derivative..."):
            expr, domain, derivative = analyze_function(json.dumps(payload["asts"][0]))
    except Exception as exc:
        st.error(f"Please check your expression: {exc}")
        return
    st.subheader("Function and symbolic derivative")
    st.latex("f(x)=" + sp.latex(expr))
    st.latex("f'(x)=" + sp.latex(derivative))
    st.latex(r"\text{Original real domain: }" + sp.latex(domain))
    st.caption("The derivative formula applies where the original function is differentiable. The check below uses the definition at your chosen point.")

    with st.expander("Step-by-step derivation", expanded=True):
        try:
            for kind, content in step_by_step_derivation(expr):
                if kind == "math":
                    st.latex(content)
                else:
                    st.write(content)
        except (NotImplementedError, ValueError, TypeError):
            st.info("A detailed rule breakdown is unavailable for this expression. The symbolic derivative is displayed above.")

    st.subheader("Choose a point and viewing range")
    c1, c2, c3 = st.columns(3)
    with c1:
        a = sp.Rational(str(st.number_input("Point a", value=1.0, step=0.1, key="derivative_point")))
    with c2:
        lo = st.number_input("x-axis minimum", value=-5.0, step=0.5, key="derivative_min")
    with c3:
        hi = st.number_input("x-axis maximum", value=5.0, step=0.5, key="derivative_max")
    if lo >= hi:
        st.warning("The x-axis minimum must be smaller than the maximum.")
        return
    if not lo <= float(a) <= hi:
        st.warning("Choose a point a inside the viewing range.")
        return
    with st.spinner("Checking the derivative at your point..."):
        value, left, right, slope = analyze_point(expr, domain, a)
    st.subheader("Does the derivative exist at a?")
    for col, label, val in zip(st.columns(3), ["f(a)", "Slope from the left", "Slope from the right"], [value, left, right]):
        with col:
            show_value(label, val)
    if slope is not None:
        st.success("Both one-sided difference quotients approach the same finite slope. The derivative exists at a.")
        st.latex("f'(" + sp.latex(a) + ")=" + sp.latex(slope))
    elif value is None:
        st.warning("f(a) is undefined or not real and finite, so this function has no derivative at a.")
    elif "Undetermined" in (left, right):
        st.info("The symbolic check is inconclusive. The graph and table are evidence to explore, not a proof of differentiability.")
    else:
        st.warning("There is no finite two-sided derivative at a. A tangent line with a finite slope is not drawn.")
    st.subheader("Graph of the function and its derivative")
    fig, exact_boundaries = plot_curves(expr, derivative, domain, a, lo, hi, value, slope)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Blue: f(x). Orange: f′(x). Graphs use numerical samples; excluded endpoints are omitted.")
    if not exact_boundaries:
        st.caption("Some graph boundaries could not be determined symbolically. Use the derivative check above to assess the selected point.")
    st.subheader("Tangent line at a")
    if slope is not None:
        tangent = sp.simplify(slope*(x-a)+value)
        st.latex(r"y-f(a)=f'(a)(x-a)")
        st.latex("y=" + sp.latex(tangent))
        fig, _ = plot_curves(expr, derivative, domain, a, lo, hi, value, slope, tangent=True)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"At x = {sp.N(a, 6)}, the instantaneous rate of change is {sp.N(slope, 8)} units of f per unit of x.")
    else:
        st.info("A finite tangent slope has not been established at this point.")

    st.subheader("Derivative definition: a table from both sides")
    st.latex(r"f'(a)=\lim_{h\to0}\frac{f(a+h)-f(a)}{h}")
    rows = []
    for side, sign in [("From the left", -1), ("From the right", 1)]:
        for magnitude in [sp.Integer(1), sp.Rational(1, 2), sp.Rational(1, 10), sp.Rational(1, 100), sp.Rational(1, 1000)]:
            h = sign*magnitude
            nearby = point_value((expr, domain), a+h)
            secant = sp.simplify((nearby-value)/h) if nearby is not None and value is not None else None
            rows.append({
                "Approach": side, "h": str(sp.N(h, 6)), "a + h": str(sp.N(a+h, 10)),
                "Secant slope": str(sp.N(secant, 10)) if secant is not None else "Undefined",
            })
    st.table(rows)
    if slope is not None:
        st.latex(r"\lim_{h\to0}\frac{f(a+h)-f(a)}{h}=" + sp.latex(slope))
    st.subheader("Reflection")
    st.text_area(
        "How do the secant slopes relate to f′(a)? What happens when the left and right slopes disagree?",
        key="derivative_reflection",
    )
    st.caption("Your reflection stays in this session; it is not submitted to your teacher.")


if __name__ == "__main__":
    st.set_page_config(page_title="Derivative Visualizer", layout="wide")
    run()
