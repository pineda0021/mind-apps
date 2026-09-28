# -*- coding: utf-8 -*-
"""Solid of revolution tool: import solid_volume_tool and call solid_volume_tool.run().

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
  const pattern = /\s+|\\[a-zA-Z]+|(?:\d+(?:\.\d*)?|\.\d+)|[xye+\-*/^(){}|]/y;
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
    return t && (/^[\d.]/.test(t) || ['x','y','e','(','{','|','\\pi','\\frac','\\sqrt'].includes(t)
      || funcs.some(f => t === '\\' + f));
  }
  function group() { expect('{'); const v = sum('}'); expect('}'); return v; }
  function atom(stop) {
    if (++depth > 25) throw Error('Please use a shorter expression.');
    let v, t = tokens[i++];
    if (t && /^[\d.]/.test(t)) v = ['num', t];
    else if (t === 'x' || t === 'y' || t === 'e' || t === '\\pi') v = [t === '\\pi' ? 'pi' : t];
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
    } else throw Error('Expected a number, the selected variable, or a supported function.');
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
  const root = parentElement.querySelector('.solid_volume-editor');
  const rows = root.querySelector('.rows'), status = root.querySelector('.status');
  const drafts = window.__solid_volumeDrafts ||= new Map();
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
        "solid_volume_mathquill",
        html='''<div class="solid_volume-editor"><div class="rows"></div>
          <button class="analyze" type="button">Analyze</button>
          <p class="status" role="status" aria-live="polite">Enter your function, then select Analyze.</p></div>''',
        css="""
        .solid_volume-editor {font-family: sans-serif; color: var(--st-text-color);}
        .solid_volume-editor .entry {margin-bottom: 18px;}
        .solid_volume-editor .label {display: block; margin-bottom: 8px;}
        .solid_volume-editor .mathfield {display: block; min-height: 55px; padding: 12px;
          border: 2px solid #64748b; border-radius: 6px; font-size: 24px;
          background: white; color: #111827; overflow-x: auto;}
        .solid_volume-editor button {padding: 8px 12px; margin: 6px 5px 6px 0;
          border: 1px solid #64748b; border-radius: 6px; cursor: pointer;
          background: #f1f5f9; color: #111827;}
        .solid_volume-editor button:focus-visible, .solid_volume-editor input:focus-visible
          {outline: 3px solid #2563eb; outline-offset: 2px;}
        .solid_volume-editor .analyze {background: #1d4ed8; color: white; font-weight: bold;}
        .solid_volume-editor .plain-label {display: block; font-size: 14px;}
        .solid_volume-editor .plain {display: block; width: 100%; box-sizing: border-box;
          padding: 9px; border: 1px solid #64748b; border-radius: 5px; margin-top: 4px;}
        .solid_volume-editor .status {font-size: 14px; min-height: 20px;}
        """,
        js=EDITOR_JS,
        isolate_styles=False,
    )


def build_expression(tree, variable_name="x", depth=0):
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
    constants = {variable_name: x, "e": sp.E, "pi": sp.pi}
    if op in constants and not items:
        return constants[op], sp.S.Reals
    arities = {**dict.fromkeys(["add", "sub", "mul", "div", "pow"], 2),
               **dict.fromkeys(["neg", "sqrt", "sin", "cos", "tan", "ln", "log", "exp", "abs"], 1)}
    if op not in arities or len(items) != arities[op]:
        raise ValueError(f"Use only {variable_name} and the listed functions. Change the input variable when switching methods or axes.")
    children = [build_expression(t, variable_name, depth + 1) for t in items]
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
        raise ValueError("This explorer requires finite real bounds and axis offsets.")
    return value



def session_result(name, token, compute):
    """Reuse the last result without asking Streamlit to hash SymPy objects."""
    previous = st.session_state.get(name)
    if previous is None or previous[0] != token:
        previous = (token, compute())
        st.session_state[name] = previous
    return previous[1]


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

r = sp.Symbol("r", nonnegative=True)


def display_expr(expr, variable_name):
    return sp.latex(expr.xreplace({x: sp.Symbol(variable_name, real=True)}))


def compile_functions(tree_json, variable_name):
    trees = json.loads(tree_json)
    f, fd = build_expression(trees[0], variable_name)
    g, gd = build_expression(trees[1], variable_name)
    return f, g, fd.intersect(gd), sp.lambdify(x, sp.simplify(f), "numpy"), sp.lambdify(x, sp.simplify(g), "numpy")


def validate_region(domain, a, b, c):
    if not all(finite_real(p) for p in (a, b, c)):
        raise ValueError("Use finite real bounds and a finite real axis offset.")
    if bool(a > b): a, b = b, a
    af, bf, cf = float(a), float(b), float(c)
    if not all(math.isfinite(p) for p in (af, bf, cf, bf-af, af-cf, bf-cf)):
        raise ValueError("The coordinates are too large for numerical evaluation.")
    if a != b and af >= bf:
        raise ValueError("The bounds are too close to distinguish numerically.")
    contained = sp.Interval(a, b).is_subset(domain)
    if contained is not True and contained is not sp.S.true:
        raise ValueError("Both original functions must be real, defined, and continuous throughout the closed interval. Choose different bounds. Infinite bounds, improper integrals, and intervals with holes need a separate analysis.")
    return a, b, af, bf, cf


def washer_radii(f, g, positions, c):
    first, second = real_values(f, positions), real_values(g, positions)
    low, high = np.minimum(first, second), np.maximum(first, second)
    with np.errstate(all="ignore"):
        outer = np.maximum(np.abs(low-c), np.abs(high-c))
        inner = np.where((low <= c) & (c <= high), 0, np.minimum(np.abs(low-c), np.abs(high-c)))
    if not np.all(np.isfinite(outer)):
        raise ValueError("A radius is too large to evaluate.")
    return inner, outer


def washer_density(f, g, positions, c):
    inner, outer = washer_radii(f, g, positions, c)
    with np.errstate(all="ignore"):
        # Factored form avoids subtracting two nearly equal squared radii.
        return np.pi*(outer-inner)*(outer+inner)


def shell_bands(f, g, radii, a, b, c):
    """At each radius, return the original axial intervals on both sides."""
    radii = np.asarray(radii, dtype=float)
    bands = []
    for direction in (-1, 1):
        start, end = (max(0.0, c-b), c-a) if direction == -1 else (max(0.0, a-c), b-c)
        active = (radii >= start) & (radii <= end) & (end > start)
        low, high = np.zeros(radii.shape), np.zeros(radii.shape)
        if np.any(active):
            positions = np.clip(c + direction*radii[active], a, b)
            first, second = real_values(f, positions), real_values(g, positions)
            low[active], high[active] = np.minimum(first, second), np.maximum(first, second)
        bands.append((low, high, active))
    return bands


def shell_height(f, g, radii, a, b, c):
    (l1, h1, valid1), (l2, h2, valid2) = shell_bands(f, g, radii, a, b, c)
    overlap = np.where(valid1 & valid2, np.maximum(0, np.minimum(h1, h2)-np.maximum(l1, l2)), 0)
    return np.maximum(0, (h1-l1)+(h2-l2)-overlap)


def shell_density(f, g, radii, a, b, c):
    with np.errstate(all="ignore"):
        return 2*np.pi*np.asarray(radii)*shell_height(f, g, radii, a, b, c)


def merge_intervals(intervals):
    """Union axial intervals so a rotated region is counted only once."""
    merged = []
    for low, high in sorted(intervals):
        if high <= low: continue
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(high, merged[-1][1]))
        else:
            merged.append((low, high))
    return merged


def radial_intervals_at(f, g, radius, a, b, c):
    bands = shell_bands(f, g, np.asarray([radius]), a, b, c)
    return merge_intervals([(float(low[0]), float(high[0])) for low, high, active in bands if active[0]])


def zero_cuts(expressions, variable, a, b):
    cuts = {a, b}
    complete = True
    for expression in expressions:
        try:
            expression = sp.simplify(expression)
            if expression == 0 or not expression.has(variable): continue
            if sp.count_ops(expression) > 40:
                raise ValueError()
            if expression.is_polynomial(variable) and sp.degree(expression, variable) > 8:
                raise ValueError()
            roots = sp.solveset(expression, variable, domain=sp.Interval.open(a, b))
            if roots is sp.S.EmptySet: continue
            if not isinstance(roots, sp.FiniteSet) or len(roots) > 80:
                raise ValueError()
            for point in roots:
                if not finite_real(point): raise ValueError()
                cuts.add(point)
        except Exception:
            complete = False
    if len(cuts) > 100:
        return [a, b], False
    return sorted(cuts, key=lambda p: float(sp.N(p, 17))), complete


def ordered_at(first, second, variable, midpoint):
    difference = sp.simplify((first-second).subs(variable, midpoint))
    if difference.is_nonpositive is True: return first, second
    if difference.is_nonnegative is True: return second, first
    raise ValueError("The ordering could not be established symbolically.")


def washer_pieces(f, g, a, b, c):
    uf, ug = sp.simplify(f-c), sp.simplify(g-c)
    cuts, complete = zero_cuts([uf, ug, uf-ug, uf+ug], x, a, b)
    outer2 = sp.Max(uf**2, ug**2)
    general = sp.pi*sp.Piecewise((outer2, uf*ug <= 0), (sp.Abs(uf**2-ug**2), True))
    pieces = []
    for left, right in zip(cuts, cuts[1:]):
        density = general
        if complete:
            try:
                midpoint = (left+right)/2
                inner2, outer2 = ordered_at(uf**2, ug**2, x, midpoint)
                product = sp.simplify((uf*ug).subs(x, midpoint))
                if product.is_nonpositive is True: inner2 = sp.S.Zero
                elif product.is_positive is not True: raise ValueError()
                density = sp.simplify(sp.pi*(outer2-inner2))
            except Exception:
                pass
        pieces.append((left, right, density))
    return pieces


def shell_pieces(f, g, a, b, c):
    """Integrate in radius, splitting where the two source sides start/end."""
    sources = []
    for direction in (-1, 1):
        start, end = (sp.Max(0, c-b), c-a) if direction == -1 else (sp.Max(0, a-c), b-c)
        if bool(end > start):
            sources.append((start, end, sp.simplify(f.subs(x, c+direction*r)), sp.simplify(g.subs(x, c+direction*r))))
    boundaries = sorted({p for item in sources for p in item[:2]}, key=lambda p: float(sp.N(p, 17)))
    pieces = []
    for left, right in zip(boundaries, boundaries[1:]):
        midpoint = (left+right)/2
        active = [(first, second) for start, end, first, second in sources if bool(start < midpoint) and bool(midpoint < end)]
        if not active: continue
        endpoints = [value for pair in active for value in pair]
        switches = [endpoints[i]-endpoints[j] for i in range(len(endpoints)) for j in range(i)]
        cuts, complete = zero_cuts(switches, r, left, right)
        for low_radius, high_radius in zip(cuts, cuts[1:]):
            middle = (low_radius+high_radius)/2
            intervals = [(sp.Min(*pair), sp.Max(*pair)) for pair in active]
            if complete:
                try: intervals = [ordered_at(*pair, r, middle) for pair in active]
                except Exception: pass
            height = sum((hi-lo for lo, hi in intervals), sp.S.Zero)
            if len(intervals) == 2:
                (l1, u1), (l2, u2) = intervals
                overlap = sp.Max(0, sp.Min(u1, u2)-sp.Max(l1, l2))
                if complete:
                    try:
                        _, lo = ordered_at(l1, l2, r, middle)
                        hi, _ = ordered_at(u1, u2, r, middle)
                        _, overlap = ordered_at(sp.S.Zero, hi-lo, r, middle)
                    except Exception: pass
                height -= overlap
            pieces.append((low_radius, high_radius, sp.simplify(2*sp.pi*r*height)))
    return pieces


def numerical_volume(density, breaks, tolerance=1e-6, max_panels=1024):
    """Refine 12-point Gauss quadrature; stabilization is not a proof of error."""
    breaks = np.asarray(sorted(set(float(p) for p in breaks)), dtype=float)
    if len(breaks) < 2 or np.any(~np.isfinite(breaks)) or np.any(np.diff(breaks) <= 0):
        raise ValueError("The integration intervals must be finite and increasing.")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Use a positive finite numerical tolerance.")
    nodes, weights = np.polynomial.legendre.leggauss(12)
    previous, stable, change, evaluations = None, 0, math.inf, 0
    panels = 1
    while panels <= max_panels:
        subtotal = []
        for left, right in zip(breaks, breaks[1:]):
            edges = np.linspace(left, right, panels+1)
            half = np.diff(edges)/2
            if np.any(half <= 0): raise ValueError("The subdivisions are too narrow to represent numerically.")
            centers = edges[:-1]+half
            points = centers[:, None]+half[:, None]*nodes[None, :]
            values = real_values(density, points)
            if np.any(values < 0): raise ValueError("The volume density unexpectedly became negative.")
            with np.errstate(all="ignore"):
                parts = half*np.sum(values*weights[None, :], axis=1)
            if np.any(~np.isfinite(parts)): raise ValueError("The numerical volume overflowed.")
            subtotal.extend(parts)
            evaluations += points.size
        current = math.fsum(subtotal)
        if not math.isfinite(current): raise ValueError("The numerical volume is not finite.")
        if previous is not None:
            change = abs(current-previous)
            stable = stable+1 if change <= tolerance*max(1.0, abs(current)) else 0
            if stable >= 2:
                return dict(volume=current, change=change, stable=True, evaluations=evaluations)
        previous = current
        panels *= 2
    return dict(volume=current, change=change, stable=False, evaluations=evaluations)


def exact_volume(pieces, variable):
    values = []
    try:
        if len(pieces) > 30: return None
        for left, right, density in pieces:
            if sp.count_ops(density) > 80: return None
            value = sp.integrate(density.rewrite(sp.Piecewise), (variable, left, right))
            if value.has(sp.Integral) or not finite_real(value) or value.is_nonnegative is not True:
                return None
            values.append(value)
        total = sp.simplify(sum(values, sp.S.Zero))
        return total if finite_real(total) and total.is_nonnegative is True else None
    except Exception:
        return None


def make_slices(f, g, method, a, b, c, breaks, count):
    edges = np.unique(np.concatenate([np.linspace(breaks[0], breaks[-1], count+1), np.asarray(breaks, dtype=float)]))
    if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("The slice widths cannot be represented numerically.")
    slices = []
    for left, right in zip(edges, edges[1:]):
        middle = left+(right-left)/2
        if method == "Disk/Washer":
            inner, outer = washer_radii(f, g, np.asarray([middle]), c)
            bands = [(float(left), float(right))]
            inner, outer = float(inner[0]), float(outer[0])
            height = right-left
        else:
            inner, outer = float(left), float(right)
            bands = radial_intervals_at(f, g, middle, a, b, c)
            height = math.fsum(hi-lo for lo, hi in bands)
        with np.errstate(all="ignore"):
            volume = np.pi*(outer-inner)*(outer+inner)*height
        if not math.isfinite(volume): raise ValueError("A slice volume overflowed.")
        slices.append(dict(left=float(left), right=float(right), midpoint=float(middle),
                           inner=inner, outer=outer, bands=bands, height=float(height), volume=float(volume)))
    return slices


def annular_mesh(axial_low, axial_high, inner, outer, axis, offset, segments=36):
    """A closed annular cylinder in physical x,y,z coordinates."""
    vertices, triangles = [], []
    for axial in (axial_low, axial_high):
        for radius in (inner, outer):
            for theta in np.linspace(0, 2*np.pi, segments, endpoint=False):
                transverse = offset+radius*math.cos(theta)
                z = radius*math.sin(theta)
                vertices.append((axial, transverse, z) if axis == "Horizontal" else (transverse, axial, z))
    def quad(a, b, c, d):
        triangles.extend([(a, b, c), (a, c, d)])
    for j in range(segments):
        k = (j+1) % segments
        quad(segments+j, segments+k, 3*segments+k, 3*segments+j)
        if inner > 0: quad(j, 2*segments+j, 2*segments+k, k)
        quad(j, k, segments+k, segments+j)
        quad(2*segments+j, 3*segments+j, 3*segments+k, 2*segments+k)
    return vertices, triangles


def make_3d(slices, axis, c, selected):
    fig = go.Figure()
    for highlight in (False, True):
        vertices, triangles = [], []
        for j, item in enumerate(slices):
            if (j == selected) != highlight: continue
            if item["outer"] <= item["inner"]: continue
            for low, high in item["bands"]:
                if high <= low: continue
                points, faces = annular_mesh(low, high, item["inner"], item["outer"], axis, c)
                offset = len(vertices)
                vertices.extend(points)
                triangles.extend(tuple(index+offset for index in face) for face in faces)
        if vertices:
            vv, tt = np.asarray(vertices), np.asarray(triangles)
            fig.add_trace(go.Mesh3d(x=vv[:, 0], y=vv[:, 1], z=vv[:, 2], i=tt[:, 0], j=tt[:, 1], k=tt[:, 2],
                                   name="Selected slice" if highlight else "Other slices", showlegend=True,
                                   color="#f97316" if highlight else "#3b82f6", opacity=0.9 if highlight else 0.35,
                                   flatshading=False, hoverinfo="name"))
    axial_values = [p for item in slices for interval in item["bands"] for p in interval]
    if axial_values:
        low, high = min(axial_values), max(axial_values)
        xx, yy = ([low, high], [c, c]) if axis == "Horizontal" else ([c, c], [low, high])
        fig.add_trace(go.Scatter3d(x=xx, y=yy, z=[0, 0], mode="lines", name="Rotation axis",
                                  line=dict(color="#111827", width=5, dash="dash")))
    fig.update_layout(height=570, margin=dict(l=0, r=0, t=10, b=0),
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"))
    return fig


def make_region_plot(f, g, variable_name, a, b, c, axis, method, selected):
    coordinates = np.linspace(a, b, 900)
    first, second = real_values(f, coordinates), real_values(g, coordinates)
    low, high = np.minimum(first, second), np.maximum(first, second)
    polygon_t = np.concatenate([coordinates, coordinates[::-1]])
    polygon_v = np.concatenate([low, high[::-1]])
    def xy(t, v): return (t, v) if variable_name == "x" else (v, t)
    xx, yy = xy(polygon_t, polygon_v)
    fig = go.Figure(go.Scatter(x=xx, y=yy, fill="toself", mode="lines", name="Region",
                               line=dict(width=0), fillcolor="rgba(59,130,246,0.20)", hoverinfo="skip"))
    for values, label, color in [(first, "First boundary", "#2563eb"), (second, "Second boundary", "#9333ea")]:
        xx, yy = xy(coordinates, values)
        fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name=label, line=dict(color=color, width=3)))
    positions = [selected["midpoint"]] if method == "Disk/Washer" else [c-selected["midpoint"], c+selected["midpoint"]]
    for position in positions:
        if a <= position <= b:
            endpoints = [float(real_values(f, np.asarray(position))), float(real_values(g, np.asarray(position)))]
            xx, yy = xy([position, position], endpoints)
            fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="Source slice",
                                    line=dict(color="#f97316", width=6), showlegend=False))
    if axis == "Horizontal": fig.add_hline(y=c, line_dash="dash", line_color="#111827", annotation_text="Rotation axis")
    else: fig.add_vline(x=c, line_dash="dash", line_color="#111827", annotation_text="Rotation axis")
    fig.update_layout(height=470, xaxis_title="x", yaxis_title="y", margin=dict(l=15, r=15, t=15, b=15))
    return fig


def number_latex(value, digits=10):
    return sp.latex(sp.Float(float(value), digits))


def run():
    st.header("🧊 Solids of Revolution Explorer")
    st.write("Rotate the region between two curves. Explore radii, washers, shells, and how each slice contributes to volume.")
    if not hasattr(st.components, "v2"):
        st.error("This editor requires Streamlit 1.51 or newer. Update Streamlit in requirements.txt.")
        return
    left, right = st.columns(2)
    with left:
        method = st.selectbox("Method", ["Disk/Washer", "Shell"], key="solid_method")
    with right:
        axis = st.selectbox("Axis direction", ["Horizontal", "Vertical"],
                            format_func=lambda value: "Horizontal line y = c" if value == "Horizontal" else "Vertical line x = c",
                            key="solid_axis")
    variable_name = "x" if ((method == "Disk/Washer") == (axis == "Horizontal")) else "y"
    dependent = "y" if variable_name == "x" else "x"
    st.info(f"Enter the boundaries as {dependent} = f({variable_name}) and {dependent} = g({variable_name}). The functions may cross; their order is handled automatically.")
    with st.expander("Examples and supported scenarios"):
        st.dataframe([
            {"Scenario": "Washer", "Method": "Disk/Washer", "Axis": "y = 0", "Boundaries": "y = x and y = x²", "Bounds": "0 to 1", "Volume": "2π/15"},
            {"Scenario": "Same region, different axis", "Method": "Shell", "Axis": "x = 0", "Boundaries": "y = x and y = x²", "Bounds": "0 to 1", "Volume": "π/6"},
            {"Scenario": "Horizontal input", "Method": "Disk/Washer", "Axis": "x = 0", "Boundaries": "x = y and x = y²", "Bounds": "0 to 1", "Volume": "2π/15"},
            {"Scenario": "Horizontal shells", "Method": "Shell", "Axis": "y = 0", "Boundaries": "x = y and x = y²", "Bounds": "0 to 1", "Volume": "π/6"},
            {"Scenario": "Shifted axis", "Method": "Disk/Washer", "Axis": "y = 2", "Boundaries": "y = x and y = x²", "Bounds": "0 to 1", "Volume": "8π/15"},
            {"Scenario": "Axis inside the region", "Method": "Disk/Washer", "Axis": "y = 0", "Boundaries": "y = 1 and y = -1", "Bounds": "0 to 2", "Volume": "2π"},
            {"Scenario": "Overlapping shells", "Method": "Shell", "Axis": "x = 0", "Boundaries": "y = 1 and y = 0", "Bounds": "-1 to 1", "Volume": "π (not 2π)"},
            {"Scenario": "Sphere", "Method": "Disk/Washer", "Axis": "y = 0", "Boundaries": "y = sqrt(1-x²) and y = -sqrt(1-x²)", "Bounds": "-1 to 1", "Volume": "4π/3"},
        ], use_container_width=True, hide_index=True)
        st.write("Both boundary functions must be continuous and real on the finite closed interval. Negative coordinates, crossing curves, shifted axes, and reversed bound entry are supported. Infinite bounds, improper volumes, implicit or parametric regions, and oblique axes are outside this tool's scope.")
        st.write("Changing methods may require rewriting the boundaries in the other variable. This app does not automatically invert a curve or choose among multiple inverse branches.")
        st.code("x^2\nx**2\n\\sqrt{1-x^2}\n\\sin(x)\n\\frac{1}{1+x^2}", language="latex")
        st.write("Use the selected variable x or y. Supports fractions, powers, square roots, absolute value, sin, cos, tan, exp, ln, log, e, and π. Trig uses radians; ln is natural log and log is base 10. Noninteger powers use nonnegative bases, or positive bases when required. Indexed roots and piecewise input are not supported.")
    st.caption("Type / for a fraction and ^ for a power. Select Analyze after changing either function. When y is required, enter y rather than x.")
    result = get_editor()(data={"labels": [f"First boundary: {dependent} = f({variable_name})", f"Second boundary: {dependent} = g({variable_name})"],
                               "initial": [variable_name, variable_name+"^2"]},
                          key=f"solid_equations_{variable_name}", on_value_change=lambda: None)
    if result.value is None:
        st.info("Select Analyze above to begin.")
        return
    try:
        payload = result.value
        if len(json.dumps(payload)) > 20000 or len(payload["asts"]) != 2:
            raise ValueError("Enter two shorter supported expressions.")
        token = json.dumps(payload["asts"])
        f, g, domain, f_num, g_num = session_result("solid_functions", (token, variable_name), lambda: compile_functions(token, variable_name))
    except Exception as exc:
        st.error(f"Please check the boundaries: {exc}")
        return
    st.latex(dependent + "=" + display_expr(f, variable_name) + r",\qquad " + dependent + "=" + display_expr(g, variable_name))
    col1, col2, col3 = st.columns(3)
    with col1: a_text = st.text_input(f"Start {variable_name} = a", "0", key="solid_a")
    with col2: b_text = st.text_input(f"End {variable_name} = b", "1", key="solid_b")
    with col3: c_text = st.text_input("Axis offset c", "0", key="solid_c")
    st.caption("Bounds and the axis offset accept finite expressions such as -2, pi, and sqrt(2). Volume is nonnegative; reversed bounds are reordered.")
    try:
        entered_a, entered_b, c = parse_bound(a_text), parse_bound(b_text), parse_bound(c_text)
        region_token = (token, variable_name, a_text, b_text, c_text)
        a, b, af, bf, cf = session_result("solid_region", region_token, lambda: validate_region(domain, entered_a, entered_b, c))
    except Exception as exc:
        st.warning(str(exc))
        return
    st.latex(("y" if axis == "Horizontal" else "x") + "=" + sp.latex(c))
    if bool(entered_a > entered_b): st.info("The bounds were reordered to describe the same geometric region. Volume does not change sign.")
    if a == b:
        st.success("The interval has zero width, so the volume is zero.")
        st.latex("V=0")
        return
    with st.expander("Display and numerical settings"):
        count = st.slider("Base number of displayed slices", 5, 100, 24, key="solid_slices")
        tolerance = st.select_slider("Numerical refinement tolerance", options=[1e-4, 1e-6, 1e-8, 1e-10], value=1e-6,
                                    format_func=lambda value: f"{value:.0e}", key="solid_tolerance")
        st.caption("Extra slice boundaries may be inserted at known changes in geometry. Displayed slices use midpoint values. Numerical refinement compares successive estimates; this is not a certified error bound.")
    problem_token = (region_token, method)
    try:
        pieces = session_result("solid_pieces", problem_token,
                                lambda: washer_pieces(f, g, a, b, c) if method == "Disk/Washer" else shell_pieces(f, g, a, b, c))
        breaks = sorted({float(sp.N(p, 17)) for left, right, _ in pieces for p in (left, right)})
        density = (lambda points: washer_density(f_num, g_num, points, cf)) if method == "Disk/Washer" else (
            lambda radii: shell_density(f_num, g_num, radii, af, bf, cf))
        with st.spinner("Computing the volume..."):
            numerical = session_result("solid_numerical", (problem_token, tolerance), lambda: numerical_volume(density, breaks, tolerance))
        slices = make_slices(f_num, g_num, method, af, bf, cf, breaks, int(count))
    except Exception as exc:
        st.error(f"The volume could not be evaluated: {exc}")
        return
    st.subheader("Volume")
    st.latex(r"V\approx " + number_latex(numerical["volume"], 12) + r"\quad\text{cubic units}")
    if numerical["stable"]:
        st.caption(f"The numerical estimate stabilized. Last refinement change: {numerical['change']:.3g}. This is a numerical comparison, not a guaranteed error bound.")
    else:
        st.warning(f"The numerical estimate did not stabilize at the requested tolerance. Last change: {numerical['change']:.3g}. Check the symbolic result and try a simpler or shorter interval.")
    if st.checkbox("Attempt an exact symbolic volume", value=True, key="solid_exact"):
        with st.spinner("Checking for a symbolic result..."):
            exact = session_result("solid_exact_result", problem_token, lambda: exact_volume(pieces, x if method == "Disk/Washer" else r))
        if exact is not None:
            st.latex("V=" + sp.latex(exact))
            st.caption("Symbolic result for the region and axis shown above.")
            try:
                exact_float = float(sp.N(exact, 17))
                disagreement = abs(exact_float-numerical["volume"])
                if disagreement > 10*tolerance*max(1.0, abs(exact_float)):
                    st.warning("The symbolic and numerical results differ beyond the requested tolerance. Review the setup before using the numerical estimate.")
            except (TypeError, ValueError, OverflowError):
                pass
        else:
            st.info("A finite closed symbolic result was not obtained. The integral setup below remains valid; the displayed volume is numerical.")
    st.subheader("Step-by-step setup")
    st.write("1. Identify the region between the two boundaries and measure distances from the rotation axis.")
    if method == "Disk/Washer":
        st.write("2. Use slices perpendicular to the axis. R is the farthest boundary distance. If a slice crosses the axis, its inner radius is zero.")
        t = variable_name
        st.latex(r"R("+t+r")=\max\{|f("+t+r")-c|,|g("+t+r")-c|\}")
        st.latex(r"r_{\mathrm{in}}("+t+r")=\begin{cases}0,&(f("+t+r")-c)(g("+t+r")-c)\le0,\\"
                 + r"\min\{|f("+t+r")-c|,|g("+t+r")-c|\},&\text{otherwise.}\end{cases}")
        st.write("3. Subtract the inner disk area from the outer disk area and integrate.")
        st.latex(r"V=\pi\int_{"+sp.latex(a)+"}^{"+sp.latex(b)+r"}\left[R("+t+r")^2-r_{\mathrm{in}}("+t+r")^2\right]\,d"+t)
        integral_variable = x
    else:
        st.write("2. Use slices parallel to the axis. Integrate over the nonnegative radius r.")
        st.latex(variable_name+r"=c-r\quad\text{or}\quad "+variable_name+r"=c+r")
        st.write("At each radius, retain only source positions inside the original interval. Each contributes an interval along the rotation axis. Combine those intervals before calculating their total length H(r).")
        st.latex(r"H(r)=|I_-(r)\cup I_+(r)|")
        st.latex(r"H=h_-+h_+-\max\{0,\min(U_-,U_+)-\max(L_-,L_+)\}")
        st.caption("The overlap formula applies when both source intervals exist. If only one exists, H is its length. Disjoint intervals contribute both lengths.")
        st.write("3. Multiply the shell circumference by this combined height and integrate in radius.")
        st.latex(r"V=2\pi\int r\,H(r)\,dr")
        if af < cf < bf:
            st.info("The source interval crosses the axis. Adding 2π|"+variable_name+" − c|·|f − g| over both sides can double-count volume. This calculation subtracts the overlap at each radius.")
        integral_variable = r
    st.write("4. Add the contributions from the intervals below. Known changes in geometry are split automatically; unresolved changes remain in the formula through absolute values, minima, maxima, or piecewise expressions.")
    with st.expander("Integral for each piece", expanded=len(pieces) <= 4):
        for j, (left, right, expression) in enumerate(pieces):
            st.latex("V_{"+str(j+1)+"}="+display_expr(sp.Integral(expression, (integral_variable, left, right)), variable_name))
        if len(pieces) > 1:
            st.latex("V="+"+".join("V_{"+str(j+1)+"}" for j in range(len(pieces))))
    st.subheader("Inspect the region and a slice")
    signature = (problem_token, int(count), len(slices))
    if st.session_state.get("solid_slice_signature") != signature:
        st.session_state["solid_slice_signature"] = signature
        st.session_state["solid_selected_slice"] = 1
    selected = int(st.slider("Selected slice", 1, len(slices), key="solid_selected_slice"))-1 if len(slices) > 1 else 0
    item = slices[selected]
    try:
        st.plotly_chart(make_region_plot(f_num, g_num, variable_name, af, bf, cf, axis, method, item), use_container_width=True)
    except Exception as exc:
        st.info(f"The region plot could not be drawn: {exc}")
    if method == "Disk/Washer":
        st.latex(r"\Delta V\approx\pi\left[("+number_latex(item["outer"])+r")^2-("+number_latex(item["inner"])
                 +r")^2\right]("+number_latex(item["right"]-item["left"])+r")="+number_latex(item["volume"]))
    else:
        st.latex(r"\Delta V\approx2\pi("+number_latex(item["midpoint"])+")("+number_latex(item["height"])
                 +")("+number_latex(item["right"]-item["left"])+r")="+number_latex(item["volume"]))
        st.write("Axial interval(s) for this radius: "+(", ".join(f"[{lo:.7g}, {hi:.7g}]" for lo, hi in item["bands"]) or "none"))
    if st.checkbox("Show the 3D model", value=True, key="solid_show_3d"):
        try:
            st.plotly_chart(make_3d(slices, axis, cf, selected), use_container_width=True)
            st.caption("Drag to rotate; scroll to zoom. Orange marks the selected slice. The mesh approximates circular faces with polygons and uses the midpoint height/radii of each displayed slice; it is not an exact boundary model.")
        except Exception as exc:
            st.info(f"The 3D model could not be drawn: {exc}")
    slice_total = math.fsum(s["volume"] for s in slices)
    st.write(f"Sum of displayed slice volumes: {slice_total:.10g} cubic units.")
    st.caption("This midpoint approximation can differ from the refined volume above. Increasing displayed slices improves the illustration but does not change the separately refined calculation.")
    with st.expander("Slice table"):
        rows = []
        for j, row in enumerate(slices):
            record = {"Slice": j+1, "Start": row["left"], "End": row["right"], "Midpoint": row["midpoint"], "Volume": row["volume"]}
            if method == "Disk/Washer":
                record.update({"Inner radius": row["inner"], "Outer radius": row["outer"]})
            else:
                record.update({"Combined height": row["height"], "Axial intervals": len(row["bands"])})
            rows.append(record)
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("Start/end are values of the integration variable for washers and radial distances for shells.")
    st.subheader("Reflection")
    st.text_area("How did moving the axis change the volume? When did a washer become a disk? Why can shells on opposite sides overlap?", key="solid_reflection")
    st.caption("Your reflection stays in this session; it is not submitted to your teacher.")


if __name__ == "__main__":
    st.set_page_config(page_title="Solids of Revolution Explorer", layout="wide")
    run()
