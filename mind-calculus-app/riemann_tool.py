# -*- coding: utf-8 -*-
"""Riemann tool: import riemann_tool and call riemann_tool.run().

Requires Streamlit >=1.51, NumPy, SymPy, and Plotly.
MathQuill loads in the browser; the LaTeX text box also works without it.
"""

import ast
import math
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
  const s = source.replace(/\*\*/g, '^').replace(/\\(?:left|right)\b/g, '')
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
  const root = parentElement.querySelector('.riemann-editor');
  const rows = root.querySelector('.rows'), status = root.querySelector('.status');
  const drafts = window.__riemannDrafts ||= new Map();
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
        "riemann_mathquill",
        html='''<div class="riemann-editor"><div class="rows"></div>
          <button class="analyze" type="button">Analyze</button>
          <p class="status" role="status" aria-live="polite">Enter your function, then select Analyze.</p></div>''',
        css="""
        .riemann-editor {font-family: sans-serif; color: var(--st-text-color);}
        .riemann-editor .entry {margin-bottom: 18px;}
        .riemann-editor .label {display: block; margin-bottom: 8px;}
        .riemann-editor .mathfield {display: block; min-height: 55px; padding: 12px;
          border: 2px solid #64748b; border-radius: 6px; font-size: 24px;
          background: white; color: #111827; overflow-x: auto;}
        .riemann-editor button {padding: 8px 12px; margin: 6px 5px 6px 0;
          border: 1px solid #64748b; border-radius: 6px; cursor: pointer;
          background: #f1f5f9; color: #111827;}
        .riemann-editor button:focus-visible, .riemann-editor input:focus-visible
          {outline: 3px solid #2563eb; outline-offset: 2px;}
        .riemann-editor .analyze {background: #1d4ed8; color: white; font-weight: bold;}
        .riemann-editor .plain-label {display: block; font-size: 14px;}
        .riemann-editor .plain {display: block; width: 100%; box-sizing: border-box;
          padding: 9px; border: 1px solid #64748b; border-radius: 5px; margin-top: 4px;}
        .riemann-editor .status {font-size: 14px; min-height: 20px;}
        """,
        js=EDITOR_JS,
        isolate_styles=False,
    )


def build_expression(tree, depth=0):
    """Validate the browser tree and preserve each subexpression's real domain."""
    if depth > 25 or not isinstance(tree, list) or not tree:
        raise ValueError("Please use a shorter, supported expression.")
    op, *items = tree
    if not isinstance(op, str):
        raise ValueError("Invalid expression operation.")
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
        a[1] = sp.simplify(a[1])
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
        if not a[1].has(x) and a[1].is_Integer is not True:
            # Use a consistent real convention for numerical noninteger powers.
            condition = a[0] >= 0 if a[1].is_positive is True else a[0] > 0
            try:
                domain = domain.intersect(sp.solve_univariate_inequality(condition, x, relational=False))
            except (NotImplementedError, ValueError):
                raise ValueError("The real domain of this power could not be determined.")
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



def parse_bound(text):
    """Read numbers and a small set of arithmetic expressions without eval."""
    if not text.strip() or len(text) > 120:
        raise ValueError("Use a short finite bound such as -2, pi, or sqrt(2).")
    root = ast.parse(text.replace("^", "**"), mode="eval")
    constants = {"pi": sp.pi, "e": sp.E, "E": sp.E, "oo": sp.oo}
    functions = {
        "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos,
        "exp": sp.exp, "ln": sp.log, "log": lambda v: sp.log(v, 10), "abs": sp.Abs,
    }

    def read(node, depth=0):
        if depth > 12:
            raise ValueError("Use a simpler bound.")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name) and node.id in constants:
            return constants[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = read(node.operand, depth + 1)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = read(node.left, depth + 1), read(node.right, depth + 1)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            if isinstance(node.op, ast.Pow):
                if not finite_real(right) or abs(right) > 100:
                    raise ValueError("Use an exponent between -100 and 100 in a bound.")
                return left ** right
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in functions and len(node.args) == 1 and not node.keywords):
            return functions[node.func.id](read(node.args[0], depth + 1))
        raise ValueError("Allowed bounds use numbers, pi, e, arithmetic, and functions such as sqrt.")

    value = sp.simplify(read(root.body))
    if not finite_real(value):
        raise ValueError("Riemann sums require finite real bounds; infinity is not supported here.")
    return value

METHODS = ["Lower Sum", "Upper Sum", "Left Endpoint", "Right Endpoint", "Midpoint", "Trapezoidal"]


def session_result(name, token, compute):
    """Reuse the last result without asking Streamlit to hash SymPy objects."""
    previous = st.session_state.get(name)
    if previous is None or previous[0] != token:
        previous = (token, compute())
        st.session_state[name] = previous
    return previous[1]


def compile_expression(tree_json):
    expression, domain = build_expression(json.loads(tree_json))
    reduced = sp.simplify(expression)
    return expression, reduced, domain, sp.lambdify(x, reduced, modules="numpy")


def validate_interval(domain, a, b):
    if not finite_real(a) or not finite_real(b) or not bool(a < b):
        raise ValueError("Use finite bounds with a < b. For a reversed integral, negate the result on [b, a].")
    af, bf = float(a), float(b)
    if not math.isfinite(af) or not math.isfinite(bf) or not math.isfinite(bf-af) or af >= bf:
        raise ValueError("These bounds cannot form a usable finite numerical interval.")
    contained = sp.Interval(a, b).is_subset(domain)
    if contained is not True and contained is not sp.S.true:
        raise ValueError("Choose a closed interval where the original expression is real, defined, and continuous. This explorer does not evaluate improper integrals or fill removable holes.")
    return af, bf


def extrema_candidates(expression, a, b):
    """A complete critical-point set only when the symbolic checks succeed."""
    try:
        if sp.count_ops(expression) > 50:
            raise ValueError("Expression is too large for the extrema check.")
        if expression.is_polynomial(x) and sp.degree(expression, x) > 8:
            raise ValueError("Use sampling for this higher-degree polynomial.")
        derivative = sp.simplify(sp.diff(expression, x))
        if derivative == 0:
            return [], True
        if sp.count_ops(derivative) > 100:
            raise ValueError("Derivative is too large for the extrema check.")
        interior = sp.Interval.open(a, b)
        derivative_domain = continuous_domain(derivative, x, sp.S.Reals)
        missing = interior - derivative_domain
        if missing is sp.S.EmptySet:
            corners = []
        elif isinstance(missing, sp.FiniteSet):
            corners = list(missing)
        else:
            raise ValueError("The nondifferentiable points could not be enumerated.")
        roots = sp.solveset(derivative, x, domain=interior)
        if roots is sp.S.EmptySet:
            stationary = []
        elif isinstance(roots, sp.FiniteSet):
            stationary = list(roots)
        else:
            raise ValueError("The stationary points could not be enumerated.")
        candidates = stationary + corners
        if len(candidates) > 200 or any(not finite_real(p) for p in candidates):
            raise ValueError("Too many or unresolved critical points.")
        points = sorted({float(sp.N(p, 17)) for p in candidates})
        if not all(math.isfinite(p) for p in points):
            raise ValueError("Critical points could not be evaluated.")
        return points, True
    except Exception:
        # Failure to solve is not proof that there are no interior extrema.
        return [], False


def real_values(function, points):
    """Broadcast constant functions; reject every undefined/complex sample."""
    points = np.asarray(points, dtype=float)
    try:
        with np.errstate(all="ignore"):
            raw = np.broadcast_to(np.asarray(function(points), dtype=complex), points.shape)
    except Exception:
        values = []
        for p in points.ravel():
            try:
                with np.errstate(all="ignore"):
                    values.append(complex(function(float(p))))
            except Exception:
                values.append(complex(float("nan"), 0))
        raw = np.asarray(values, dtype=complex).reshape(points.shape)
    if np.any(~np.isfinite(raw.real)) or np.any(~np.isfinite(raw.imag)) or np.any(raw.imag != 0):
        raise ValueError("A sample was undefined, complex, or too large to evaluate. Try a smaller interval within the function's real domain.")
    return np.asarray(raw.real, dtype=float)


def riemann_data(function, a, b, n, samples=50, critical_points=()):
    """Numerical quadrature; symbolic domain checking happens before this call."""
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(b-a) and a < b):
        raise ValueError("Use finite bounds with a < b.")
    if type(n) is not int or not 1 <= n <= 200:
        raise ValueError("Use 1–200 subintervals.")
    if type(samples) is not int or not 2 <= samples <= 201:
        raise ValueError("Use 2–201 samples per subinterval.")
    edges = np.linspace(a, b, n+1)
    widths = np.diff(edges)
    if np.any(widths <= 0):
        raise ValueError("The subintervals are too narrow for floating-point arithmetic.")
    mids = edges[:-1] + widths/2
    endpoints = real_values(function, edges)
    midpoint_heights = real_values(function, mids)
    grid = edges[:-1, None] + widths[:, None]*np.linspace(0, 1, samples)[None, :]
    grid[:, 0], grid[:, -1] = edges[:-1], edges[1:]
    values = real_values(function, grid)
    low_index, high_index = np.argmin(values, axis=1), np.argmax(values, axis=1)
    indices = np.arange(n)
    low, high = values[indices, low_index].copy(), values[indices, high_index].copy()
    low_x, high_x = grid[indices, low_index].copy(), grid[indices, high_index].copy()
    for point in critical_points:
        if not math.isfinite(point) or not a < point < b:
            continue
        j = min(n-1, max(0, int(np.searchsorted(edges, point, side="right"))-1))
        height = float(real_values(function, np.asarray(point)))
        if height < low[j]: low[j], low_x[j] = height, point
        if height > high[j]: high[j], high_x[j] = height, point
    heights = {"Lower Sum": low, "Upper Sum": high, "Left Endpoint": endpoints[:-1],
               "Right Endpoint": endpoints[1:], "Midpoint": midpoint_heights}
    locations = {"Lower Sum": low_x, "Upper Sum": high_x, "Left Endpoint": edges[:-1],
                 "Right Endpoint": edges[1:], "Midpoint": mids}
    with np.errstate(all="ignore"):
        contributions = {method: h*widths for method, h in heights.items()}
        contributions["Trapezoidal"] = (endpoints[:-1]/2 + endpoints[1:]/2)*widths
    if any(not np.all(np.isfinite(v)) for v in contributions.values()):
        raise ValueError("An interval contribution overflowed. Use smaller bounds or values.")
    try:
        totals = {method: math.fsum(v) for method, v in contributions.items()}
    except OverflowError:
        raise ValueError("The sum overflowed. Use smaller bounds or values.")
    if not all(math.isfinite(v) for v in totals.values()):
        raise ValueError("The sum is not finite.")
    return dict(edges=edges, widths=widths, endpoints=endpoints, heights=heights,
                locations=locations, contributions=contributions, totals=totals)


def reference_integral(expression, a, b):
    try:
        value = sp.integrate(expression, (x, a, b))
        if value.has(sp.Integral) or not finite_real(value):
            return None, None
        number = float(sp.N(value, 17))
        if not math.isfinite(number):
            return None, None
        return value, number
    except Exception:
        return None, None


def method_label(method, verified):
    if method in ("Lower Sum", "Upper Sum") and not verified:
        return "Sampled " + method.lower() + " estimate"
    return method


def signed_polygons(left, right, h_left, h_right):
    """Split an axis-crossing trapezoid into two simple signed polygons."""
    if h_left == 0 and h_right == 0:
        return []
    crossing = (h_left < 0 < h_right) or (h_right < 0 < h_left)
    if crossing:
        scale = max(abs(h_left), abs(h_right))
        fraction = (abs(h_left)/scale)/(abs(h_left)/scale+abs(h_right)/scale)
        middle = left + fraction*(right-left)
        return signed_polygons(left, middle, h_left, 0) + signed_polygons(middle, right, 0, h_right)
    positive = h_left >= 0 and h_right >= 0
    return [(positive, [left, left, right, right, left], [0, h_left, h_right, 0, 0])]


def make_plot(function, data, method, verified):
    edges = data["edges"]
    a, b = float(edges[0]), float(edges[-1])
    fig = go.Figure()
    collections = {True: {"x": [], "y": [], "custom": []}, False: {"x": [], "y": [], "custom": []}}
    for j, (left, right) in enumerate(zip(edges, edges[1:])):
        if method == "Trapezoidal":
            h_left, h_right = data["endpoints"][j:j+2]
        else:
            h_left = h_right = data["heights"][method][j]
        for positive, xx, yy in signed_polygons(left, right, h_left, h_right):
            item = collections[positive]
            item["x"].extend([*xx, None]); item["y"].extend([*yy, None])
            info = [j+1, float(data["contributions"][method][j])]
            item["custom"].extend([info]*len(xx)+[[None, None]])
    for positive, item in collections.items():
        if not item["x"]: continue
        fig.add_trace(go.Scatter(
            x=item["x"], y=item["y"], customdata=item["custom"], mode="lines", fill="toself",
            name="Positive contributions" if positive else "Negative contributions",
            line=dict(color="#15803d" if positive else "#c2410c", width=1),
            fillcolor="rgba(22,163,74,0.23)" if positive else "rgba(234,88,12,0.25)",
            hovertemplate="Subinterval %{customdata[0]}<br>Signed contribution: %{customdata[1]:.8g}<extra></extra>",
            connectgaps=False,
        ))
    xx = np.unique(np.concatenate([np.linspace(a, b, 1201), edges]))
    yy = real_values(function, xx)
    fig.add_trace(go.Scatter(x=xx, y=yy, name="f(x)", mode="lines", line=dict(color="#111827", width=3)))
    if method == "Trapezoidal":
        sample_x, sample_y = edges, data["endpoints"]
    else:
        sample_x, sample_y = data["locations"][method], data["heights"][method]
    fig.add_trace(go.Scatter(x=sample_x, y=sample_y, mode="markers", name="Evaluation points",
                            marker=dict(size=6, color="#1d4ed8")))
    fig.add_hline(y=0, line_color="#64748b", line_width=1)
    fig.update_layout(title=method_label(method, verified), height=510,
                      xaxis=dict(title="x", range=[a, b]), yaxis=dict(title="f(x)", rangemode="tozero"),
                      margin=dict(l=20, r=20, t=45, b=20), legend=dict(orientation="h"))
    return fig


def number_latex(number, digits=8):
    return sp.latex(sp.Float(float(number), digits))


def table_rows(data, method):
    rows = []
    for j, width in enumerate(data["widths"]):
        row = {"i": j+1, "Left endpoint": float(data["edges"][j]),
               "Right endpoint": float(data["edges"][j+1]), "Width": float(width)}
        if method == "Trapezoidal":
            row["f(left)"] = float(data["endpoints"][j])
            row["f(right)"] = float(data["endpoints"][j+1])
        else:
            row["Evaluation point"] = float(data["locations"][method][j])
            row["Height"] = float(data["heights"][method][j])
        row["Signed contribution"] = float(data["contributions"][method][j])
        rows.append(row)
    return rows


def run():
    st.header("∑ Riemann Sum Explorer")
    st.write("Explore how rectangles and trapezoids approximate a signed definite integral.")
    if not hasattr(st.components, "v2"):
        st.error("This editor requires Streamlit 1.51 or newer. Update Streamlit in requirements.txt.")
        return
    st.caption("Type / for a fraction and ^ for a power. Select Analyze after editing the function.")
    with st.expander("Examples and input help"):
        st.code("x^2\nx**2\n3\n-x^2\n\\sin(x)\n1-(x-0.3)^2\n\\frac{1}{1+x^2}\n\\sqrt{x}", language="latex")
        st.write("Supports x, arithmetic, implicit multiplication such as 2x, square roots, sin, cos, tan, exp, ln, log, absolute value, e, and π. Trig uses radians. ln is natural log; log is base 10. Noninteger powers use nonnegative bases (positive bases when required). Indexed roots and piecewise input are not supported in this tool.")
        st.write("Try x² on [0, 2]; sin(x) on [0, pi]; or the constant 3 to compare all methods. For an interior maximum, try 1 − (x − 0.3)² on [0, 1] with one subinterval.")
    result = get_editor()(data={"labels": ["Function f(x)"], "initial": ["x^2"]},
                          key="riemann_equation", on_value_change=lambda: None)
    if result.value is None:
        st.info("Select Analyze above to begin.")
        return
    try:
        payload = result.value
        if len(json.dumps(payload)) > 16000 or len(payload["asts"]) != 1:
            raise ValueError("Please use a shorter expression.")
        token = json.dumps(payload["asts"][0])
        expression, reduced, domain, function = session_result("riemann_expression", token, lambda: compile_expression(token))
    except Exception as exc:
        st.error(f"Please check the function: {exc}")
        return
    st.latex("f(x)=" + sp.latex(expression))
    c1, c2, c3 = st.columns(3)
    with c1: a_text = st.text_input("Interval start a", "0", key="riemann_a")
    with c2: b_text = st.text_input("Interval end b", "2", key="riemann_b")
    with c3: n = st.slider("Subintervals n", 1, 200, 10, key="riemann_n")
    st.caption("Bounds accept finite expressions such as -2, pi, and sqrt(2). Use a < b.")
    method = st.selectbox("Approximation method", METHODS, key="riemann_method")
    with st.expander("Sampling settings"):
        samples = st.slider("Samples per subinterval for lower/upper estimates", 2, 201, 51, key="riemann_samples")
        st.caption("Endpoints are included. When SymPy resolves all critical points, those are included too. Sampling alone can miss peaks and valleys, even at the highest setting.")
    try:
        a, b = parse_bound(a_text), parse_bound(b_text)
        interval_token = (token, a_text, b_text)
        af, bf = session_result("riemann_interval", interval_token, lambda: validate_interval(domain, a, b))
        critical, verified = session_result("riemann_extrema", interval_token, lambda: extrema_candidates(reduced, a, b))
        data = riemann_data(function, af, bf, int(n), int(samples), critical)
    except Exception as exc:
        st.warning(str(exc))
        return
    st.latex(r"\Delta x=\frac{b-a}{n}=\frac{" + sp.latex(b-a) + "}{" + str(n) + r"}\approx " + number_latex((bf-af)/n))
    if verified:
        st.caption("Lower/upper extrema use endpoints and the complete critical-point set found by SymPy. Displayed heights and sums are evaluated numerically, with rounding.")
    else:
        st.info("Lower and upper values are sampled estimates. The symbolic check did not establish every critical point, so these estimates are not guaranteed bounds on the integral.")
    label = method_label(method, verified)
    total = data["totals"][method]
    st.metric(label, f"{total:.10g}")
    try:
        st.plotly_chart(make_plot(function, data, method, verified), use_container_width=True)
    except Exception as exc:
        st.warning(f"The graph could not be drawn: {exc}")
    st.caption("Green pieces contribute positively; orange pieces contribute negatively. A lower sum is based on smaller function values, including when those values are negative.")
    st.subheader("How the sum is constructed")
    if method == "Trapezoidal":
        st.latex(r"T_n=\sum_{i=1}^{n}\frac{f(x_{i-1})+f(x_i)}{2}\,\Delta x")
        st.write("Each top edge connects two endpoint heights. The trapezoidal rule uses both endpoints of each subinterval.")
    else:
        st.latex(r"S_n=\sum_{i=1}^{n}f(x_i^*)\,\Delta x")
        descriptions = {
            "Lower Sum": "Use the minimum on each subinterval when the extrema are resolved; otherwise use the smallest sampled value.",
            "Upper Sum": "Use the maximum on each subinterval when the extrema are resolved; otherwise use the largest sampled value.",
            "Left Endpoint": "Choose the left endpoint of each subinterval.",
            "Right Endpoint": "Choose the right endpoint of each subinterval.",
            "Midpoint": "Choose the midpoint of each subinterval.",
        }
        st.write(descriptions[method])
    rows = table_rows(data, method)
    st.dataframe(rows, use_container_width=True, hide_index=True)
    j = int(st.number_input("Inspect subinterval i", min_value=1, max_value=int(n), value=1, step=1,
                            key=f"riemann_inspect_{n}"))-1
    st.latex(r"[x_{i-1},x_i]=[" + number_latex(data["edges"][j]) + "," + number_latex(data["edges"][j+1]) + "]")
    if method == "Trapezoidal":
        st.latex(r"A_i=\frac{(" + number_latex(data["endpoints"][j]) + ")+(" + number_latex(data["endpoints"][j+1])
                 + r")}{2}\left(" + number_latex(data["widths"][j]) + r"\right)\approx " + number_latex(data["contributions"][method][j]))
    else:
        st.latex(r"x_i^*\approx " + number_latex(data["locations"][method][j]))
        st.latex(r"A_i=f(x_i^*)\Delta x\approx \left(" + number_latex(data["heights"][method][j])
                 + r"\right)\left(" + number_latex(data["widths"][j]) + r"\right)=" + number_latex(data["contributions"][method][j]))
    st.latex(r"\sum_{i=1}^{" + str(n) + r"} A_i\approx " + number_latex(total, 12))
    st.subheader("Compare the methods")
    reference, exact_number = None, None
    if st.checkbox("Compare with the symbolic definite integral", value=True, key="riemann_reference"):
        with st.spinner("Computing the definite integral..."):
            reference, exact_number = session_result("riemann_reference_result", interval_token, lambda: reference_integral(reduced, a, b))
        if reference is not None:
            st.latex(r"I=\int_{" + sp.latex(a) + "}^{" + sp.latex(b) + "}" + sp.latex(expression) + r"\,dx=" + sp.latex(reference))
            st.caption(f"Signed integral: approximately {exact_number:.12g}. This is net accumulation, not total geometric area.")
        else:
            st.info("SymPy did not return a finite resolved reference value. The sums are available, but an error relative to the integral cannot be reported.")
    comparison = []
    for candidate in METHODS:
        value = data["totals"][candidate]
        row = {"Method": method_label(candidate, verified), "Sum": value}
        if exact_number is not None:
            error = abs(value-exact_number)
            row["Absolute error (approx.)"] = error
            if exact_number != 0:
                relative = error/abs(exact_number)
                row["Relative error"] = f"{relative:.6%}" if math.isfinite(relative) else "Too large to represent"
            else:
                row["Relative error"] = "Undefined (reference is zero)"
        comparison.append(row)
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    if exact_number is not None:
        st.latex(r"\text{Absolute error}=|S_n-I|,\qquad\text{Relative error}=\frac{|S_n-I|}{|I|}\quad(I\ne0)")
        st.caption("Errors use the numerical value of the symbolic reference. When the integral is close to zero, relative error can be large even if absolute error is small.")
    st.subheader("Reflection")
    st.text_area("What changes as n increases? When do lower and upper sums differ from left and right sums? How do negative contributions affect the total?", key="riemann_reflection")
    st.caption("Your reflection stays in this session; it is not submitted to your teacher.")


if __name__ == "__main__":
    st.set_page_config(page_title="Riemann Sum Explorer", layout="wide")
    run()
