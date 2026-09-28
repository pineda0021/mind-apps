# -*- coding: utf-8 -*-
"""Antiderivative tool: import antiderivative_tool and call antiderivative_tool.run()."""

import ast
import json
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from sympy.calculus.util import continuous_domain
from sympy.integrals.manualintegrate import integral_steps

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
  const root = parentElement.querySelector('.antiderivative-editor');
  const rows = root.querySelector('.rows'), status = root.querySelector('.status');
  const drafts = window.__antiderivativeDrafts ||= new Map();
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
        "antiderivative_mathquill",
        html='''<div class="antiderivative-editor"><div class="rows"></div>
          <button class="analyze" type="button">Analyze</button>
          <p class="status" role="status" aria-live="polite">Enter your function, then select Analyze.</p></div>''',
        css="""
        .antiderivative-editor {font-family: sans-serif; color: var(--st-text-color);}
        .antiderivative-editor .entry {margin-bottom: 18px;}
        .antiderivative-editor .label {display: block; margin-bottom: 8px;}
        .antiderivative-editor .mathfield {display: block; min-height: 55px; padding: 12px;
          border: 2px solid #64748b; border-radius: 6px; font-size: 24px;
          background: white; color: #111827; overflow-x: auto;}
        .antiderivative-editor button {padding: 8px 12px; margin: 6px 5px 6px 0;
          border: 1px solid #64748b; border-radius: 6px; cursor: pointer;
          background: #f1f5f9; color: #111827;}
        .antiderivative-editor button:focus-visible, .antiderivative-editor input:focus-visible
          {outline: 3px solid #2563eb; outline-offset: 2px;}
        .antiderivative-editor .analyze {background: #1d4ed8; color: white; font-weight: bold;}
        .antiderivative-editor .plain-label {display: block; font-size: 14px;}
        .antiderivative-editor .plain {display: block; width: 100%; box-sizing: border-box;
          padding: 9px; border: 1px solid #64748b; border-radius: 5px; margin-top: 4px;}
        .antiderivative-editor .status {font-size: 14px; min-height: 20px;}
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


def parse_bound(text):
    """Read numbers and a small set of arithmetic expressions without eval."""
    if not text.strip() or len(text) > 120:
        raise ValueError("Use a short bound such as -2, pi, sqrt(2), or oo.")
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
        raise ValueError("Allowed bounds use numbers, pi, e, oo, arithmetic, and functions such as sqrt.")

    value = sp.simplify(read(root.body))
    if not finite_real(value) and value not in (sp.oo, -sp.oo):
        raise ValueError("A bound must be a real number, oo, or -oo.")
    return value


def real_primitive(result, variable=x):
    """Take a real primitive on each real interval; log(x) becomes log|x|."""
    if result.has(sp.Integral):
        return result
    if variable.is_real is not True:
        result = result.xreplace({variable: sp.Symbol(variable.name, real=True)})
    return sp.simplify(sp.re(result))


def integrate_function(tree_json):
    expr, domain = build_expression(json.loads(tree_json))
    reduced = sp.simplify(expr)
    rule = None
    try:
        rule = integral_steps(reduced, x)
        result = rule.eval()
    except (NotImplementedError, ValueError, TypeError, AttributeError):
        result = sp.Integral(reduced, x)
    if result.has(sp.Integral):
        result = sp.integrate(reduced, x)
    primitive = real_primitive(result)
    return expr, domain, primitive, rule


def session_result(name, token, compute):
    """Keep the last result in this session, without hashing SymPy objects."""
    previous = st.session_state.get(name)
    if previous is None or previous[0] != token:
        previous = (token, compute())
        st.session_state[name] = previous
    return previous[1]


def integration_steps(rule):
    """Translate the manual integrator's actual rule tree into student steps."""
    steps = []

    def text(s): steps.append(("text", s))
    def math(s): steps.append(("math", s))

    def conclusion(node):
        result = real_primitive(node.eval(), node.variable)
        if not result.has(sp.Integral):
            math(sp.latex(sp.Integral(node.integrand, node.variable)) + "=" + sp.latex(result))

    def visit(node, depth=0):
        if node is None:
            return
        if depth > 12 or len(steps) > 100:
            text("The remaining steps are summarized by the result below.")
            conclusion(node)
            return
        kind = type(node).__name__
        if kind == "AlternativeRule":
            choices = node.alternatives
            chosen = next((r for r in choices if not r.contains_dont_know()), choices[0])
            visit(chosen, depth + 1)
            return
        if kind == "AddRule":
            text("Sum rule: integrate each term separately.")
            for child in node.substeps:
                visit(child, depth + 1)
            text("Combine the antiderivatives.")
        elif kind == "ConstantTimesRule":
            text("Constant multiple rule: move the constant outside the integral.")
            math(sp.latex(sp.Integral(node.integrand, node.variable)) + "="
                 + sp.latex(node.constant) + r"\left(" + sp.latex(sp.Integral(node.other, node.variable)) + r"\right)")
            visit(node.substep, depth + 1)
        elif kind in ("RewriteRule", "CompleteSquareRule"):
            text("Rewrite the integrand into a form that is easier to integrate.")
            math(sp.latex(node.integrand) + "=" + sp.latex(node.rewritten))
            visit(node.substep, depth + 1)
        elif kind == "URule":
            text("Substitution: change both the expression and the differential.")
            math(sp.latex(node.u_var) + "=" + sp.latex(node.u_func))
            math("d" + sp.latex(node.u_var) + "=" + sp.latex(sp.diff(node.u_func, node.variable))
                 + r"\,d" + sp.latex(node.variable))
            math(sp.latex(sp.Integral(node.integrand, node.variable)) + "="
                 + sp.latex(sp.Integral(node.substep.integrand, node.u_var)))
            visit(node.substep, depth + 1)
            text("Substitute the original expression back in.")
        elif kind == "PartsRule" and node.second_step is not None:
            text("Integration by parts.")
            math(r"\int u\,dv=uv-\int v\,du")
            math("u=" + sp.latex(node.u) + r",\qquad dv=" + sp.latex(node.dv) + r"\,d" + sp.latex(node.variable))
            visit(node.v_step, depth + 1)
            v = node.v_step.eval()
            math("du=" + sp.latex(sp.diff(node.u, node.variable)) + r"\,d" + sp.latex(node.variable)
                 + r",\qquad v=" + sp.latex(real_primitive(v, node.variable)))
            math(sp.latex(sp.Integral(node.integrand, node.variable)) + "="
                 + sp.latex(node.u*v) + "-" + sp.latex(sp.Integral(v*sp.diff(node.u, node.variable), node.variable)))
            visit(node.second_step, depth + 1)
        elif kind == "ConstantRule":
            text("Constant rule: integrate a constant by multiplying it by the variable.")
        elif kind == "PowerRule":
            text("Power rule: add one to the exponent, then divide by the new exponent.")
            math(r"\int t^n\,dt=\frac{t^{n+1}}{n+1},\qquad n\ne-1")
        elif kind == "ReciprocalRule":
            text("Logarithm rule, on an interval that does not cross zero.")
            math(r"\int\frac{1}{t}\,dt=\ln|t|")
        elif kind in ("SinRule", "CosRule", "ExpRule"):
            text("Apply the standard trigonometric or exponential antiderivative rule.")
        elif kind == "DontKnowRule":
            text("The step engine did not find a method for this part. The general integrator may still find a result.")
            return
        else:
            text("This part uses a symbolic integration rule; detailed intermediate steps are not available here.")
        conclusion(node)

    visit(rule)
    text("Add one arbitrary constant C to the final answer. Separate domain intervals may have different constants.")
    return steps


def ordered_bounds(a, b):
    if a == b:
        if a in (sp.oo, -sp.oo):
            raise ValueError("Equal infinite bounds do not define an integration interval.")
        return a, b, 1
    if bool(a < b):
        return a, b, 1
    return b, a, -1


def definite_integral(expr, domain, a, b):
    """Split at every known excluded interior point and check each part."""
    lo, hi, orientation = ordered_bounds(a, b)
    if lo == hi:
        return {"status": "finite", "value": sp.S.Zero, "parts": [], "proper": True}
    interior = sp.Interval.open(lo, hi)
    missing = interior - domain
    if missing is sp.S.EmptySet:
        breaks = []
    elif isinstance(missing, sp.FiniteSet):
        breaks = sorted(missing, key=lambda p: float(sp.N(p)))
    else:
        raise ValueError("The interval includes a non-real region or discontinuities this tool cannot fully resolve. Choose bounds within one real interval, or with finitely many isolated singularities.")
    proper = bool(finite_real(lo) and finite_real(hi)
                  and sp.Interval(lo, hi).is_subset(domain) is True)
    points = [lo, *breaks, hi]
    if lo == -sp.oo and hi == sp.oo and not breaks:
        points = [lo, sp.S.Zero, hi]
    parts = []
    status = "finite"
    for start, end in zip(points, points[1:]):
        value = sp.integrate(expr, (x, start, end))
        if value.has(sp.Integral):
            state = "unresolved"
        elif value in (sp.oo, -sp.oo, sp.zoo, sp.nan) or value.has(sp.AccumBounds):
            state = "divergent"
        elif finite_real(value):
            state = "finite"
        else:
            state = "unresolved"
        parts.append((start, end, value, state))
        if state == "divergent":
            status = "divergent"
        elif state == "unresolved" and status == "finite":
            status = "unresolved"
    # Never add divergent parts: -infinity + infinity is not convergence.
    total = sp.simplify(orientation*sum((p[2] for p in parts), sp.S.Zero)) if status == "finite" else None
    return {"status": status, "value": total, "parts": parts, "proper": proper}


def sample_values(expr, values):
    """Handle scalar constants, special functions, and real-domain gaps."""
    try:
        function = sp.lambdify(x, expr, modules="numpy")
        with np.errstate(all="ignore"):
            yy = np.broadcast_to(np.asarray(function(values), dtype=complex), values.shape)
    except Exception:
        function = sp.lambdify(x, expr, modules="mpmath")
        result = []
        for p in values:
            try: result.append(complex(function(float(p))))
            except Exception: result.append(complex(float("nan"), 0))
        yy = np.asarray(result)
    return np.where((np.abs(yy.imag) < 1e-9) & np.isfinite(yy.real), yy.real, np.nan)


def plot_functions(expr, domain, lo, hi, primitive=None, constant=0, shade=None):
    fig = go.Figure()
    cuts = {lo, hi}
    complete = True
    try:
        boundaries = domain.boundary.intersect(sp.Interval(lo, hi))
        if isinstance(boundaries, sp.FiniteSet): cuts.update(float(p) for p in boundaries)
        elif boundaries is not sp.S.EmptySet: complete = False
    except (NotImplementedError, TypeError, ValueError):
        complete = False
    if shade is not None:
        lower, upper, _ = ordered_bounds(*shade)
        for p in (lower, upper):
            if finite_real(p) and lo < float(p) < hi: cuts.add(float(p))
    seen = set()
    cuts = sorted(cuts)
    for start, end in zip(cuts, cuts[1:]):
        mid = sp.Rational(str((start+end)/2))
        if domain.contains(mid) is sp.S.false: continue
        xx = np.linspace(start, end, max(30, int(650*(end-start)/(hi-lo))))[1:-1]
        try:
            yy = sample_values(expr, xx)
        except Exception:
            complete = False
            continue
        for label, function, color in [("f(x)", expr, "#2563eb"), ("F(x) + C", primitive, "#d97706")]:
            if function is None or function.has(sp.Integral): continue
            try:
                values = yy if label == "f(x)" else sample_values(function + constant, xx)
            except Exception:
                complete = False
                continue
            fig.add_trace(go.Scatter(x=xx, y=values, mode="lines", name=label, legendgroup=label,
                                    showlegend=label not in seen, line=dict(color=color, width=3), connectgaps=False))
            seen.add(label)
        if shade is not None and bool(lower <= mid) and bool(mid <= upper):
            for label, values, color in [
                ("Above the x-axis", np.maximum(yy, 0), "rgba(22,163,74,0.25)"),
                ("Below the x-axis", np.minimum(yy, 0), "rgba(234,88,12,0.25)"),
            ]:
                fig.add_trace(go.Scatter(x=xx, y=values, mode="lines", line=dict(width=0),
                                        fill="tozeroy", fillcolor=color, name=label, legendgroup=label,
                                        showlegend=label not in seen, connectgaps=False, hoverinfo="skip"))
                seen.add(label)
    fig.update_layout(xaxis=dict(title="x", range=[lo, hi]), yaxis_title="y", height=470,
                      margin=dict(l=20, r=20, t=20, b=20))
    return fig, complete


def run():
    st.header("∫ Antiderivative Visualizer")
    st.write("Explore antiderivative rules, the constant of integration, and signed definite integrals.")
    if not hasattr(st.components, "v2"):
        st.error("This editor requires Streamlit 1.51 or newer. Update Streamlit in requirements.txt.")
        return
    st.subheader("Enter a function")
    st.caption("Type / for a fraction and ^ for a power. Use arrow keys to leave an exponent or denominator. Select Analyze after editing.")
    with st.expander("Examples and input help"):
        st.code("x^2+1\n2x(x^2+1)^3\nx e^x\n\\frac{1}{x}\n\\sin(x)\n\\frac{1}{\\sqrt{x}}", language="latex")
        st.write("Use x as the variable. Supports fractions, powers, square roots, sin, cos, tan, exp, ln, log, absolute value, e, and π. Trig uses radians. ln is natural log; log is base 10. Variable exponents use the positive-base real domain. Equations, indexed roots, and piecewise input are not supported.")
    result = get_editor()(data={"labels": ["Integrand f(x)"], "initial": ["x^2+1"]},
                          key="antiderivative_equation", on_value_change=lambda: None)
    if result.value is None:
        st.info("Select Analyze above to begin.")
        return
    try:
        payload = result.value
        if len(json.dumps(payload)) > 16000 or len(payload["asts"]) != 1:
            raise ValueError("Please use a shorter expression.")
        token = json.dumps(payload["asts"][0])
        with st.spinner("Finding an antiderivative..."):
            expr, domain, primitive, rule = session_result("anti_symbolic", token, lambda: integrate_function(token))
    except Exception as exc:
        st.error(f"Please check your expression: {exc}")
        return
    st.subheader("Symbolic antiderivative")
    st.latex("f(x)=" + sp.latex(expr))
    st.latex(r"\text{Original real domain: }" + sp.latex(domain))
    resolved = not primitive.has(sp.Integral)
    if resolved:
        st.latex(sp.latex(sp.Integral(expr, x)) + "=" + sp.latex(primitive) + "+C")
        st.caption("Antiderivatives apply on intervals of the original domain. Separate intervals may have different constants. Real logarithmic antiderivatives use absolute values where needed.")
    else:
        st.latex(sp.latex(primitive))
        st.info("SymPy left an integral unevaluated. This does not prove that no antiderivative exists. You can still explore the function and definite integral below.")
    with st.expander("Step-by-step integration", expanded=True):
        if rule is None:
            st.write("A detailed rule breakdown is unavailable for this expression.")
        else:
            try:
                for kind, content in integration_steps(rule):
                    if kind == "math": st.latex(content)
                    else: st.write(content)
            except (NotImplementedError, ValueError, TypeError, AttributeError):
                st.info("The step engine could not display every intermediate step for this expression.")
    if resolved:
        with st.expander("Check by differentiating"):
            try:
                derivative = sp.simplify(sp.diff(primitive, x))
                st.latex(r"\frac{d}{dx}[F(x)]=" + sp.latex(derivative))
                residual = sp.simplify(derivative - expr)
                if residual == 0:
                    st.success("Differentiating F gives the original integrand.")
                else:
                    st.write("Review this derivative on the original domain. Absolute values and piecewise expressions may prevent a single global simplification.")
            except (NotImplementedError, ValueError, TypeError):
                st.info("The symbolic derivative check could not be completed for this expression.")
    st.subheader("Graph and constant of integration")
    c1, c2, c3 = st.columns(3)
    with c1: lo = st.number_input("x-axis minimum", value=-5.0, step=0.5, key="anti_min")
    with c2: hi = st.number_input("x-axis maximum", value=5.0, step=0.5, key="anti_max")
    with c3: constant = st.number_input("Constant C", value=0.0, step=1.0, key="anti_constant")
    if lo >= hi:
        st.warning("The x-axis minimum must be smaller than the maximum.")
        return
    fig, complete = plot_functions(expr, domain, lo, hi, primitive=primitive if resolved else None,
                                   constant=sp.Rational(str(constant)))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Blue: f(x). Orange: F(x) + C. Changing C shifts the antiderivative vertically without changing its derivative.")
    if not complete:
        st.caption("Some curve values or boundaries could not be resolved. The graph is a numerical illustration.")
    st.subheader("Definite integral and signed accumulation")
    c1, c2 = st.columns(2)
    with c1: a_text = st.text_input("Lower bound a", "-2", key="anti_a")
    with c2: b_text = st.text_input("Upper bound b", "2", key="anti_b")
    st.caption("Bounds accept numbers and expressions such as pi, sqrt(2), -oo, and oo. Reversing the bounds reverses the sign of a convergent integral.")
    try:
        a, b = parse_bound(a_text), parse_bound(b_text)
        with st.spinner("Checking the definite integral..."):
            answer = session_result("anti_definite", (token, a_text, b_text), lambda: definite_integral(expr, domain, a, b))
    except Exception as exc:
        st.warning(f"Could not evaluate these bounds: {exc}")
        return
    st.latex(sp.latex(sp.Integral(expr, (x, a, b))))
    if answer["status"] == "finite":
        st.latex("=" + sp.latex(answer["value"]))
        st.write("Approximate value:", str(sp.N(answer["value"], 10)))
    elif answer["status"] == "divergent":
        st.error("The improper integral diverges. Divergent pieces cannot be canceled to obtain a finite answer.")
    else:
        st.info("Symbolic evaluation or convergence is unresolved. The graph alone cannot establish convergence.")
    with st.expander("Definite-integral reasoning", expanded=True):
        if a == b:
            st.write("Equal finite bounds give zero signed accumulation.")
        elif answer["proper"]:
            st.write("The original integrand is continuous throughout the closed interval.")
            if resolved and answer["status"] == "finite":
                try:
                    low, high, _ = ordered_bounds(a, b)
                    primitive_domain = continuous_domain(primitive, x, sp.S.Reals)
                    valid = sp.Interval(low, high).is_subset(primitive_domain) is True
                    Fa, Fb = sp.simplify(primitive.subs(x, a)), sp.simplify(primitive.subs(x, b))
                    agrees = sp.simplify(Fb-Fa-answer["value"]) == 0
                    if valid and finite_real(Fa) and finite_real(Fb) and agrees:
                        st.latex(r"\int_a^b f(x)\,dx=F(b)-F(a)")
                        st.latex("=" + sp.latex(Fb) + r"-\left(" + sp.latex(Fa) + r"\right)=" + sp.latex(answer["value"]))
                        st.write("The constant C cancels when subtracting endpoint values.")
                    else:
                        st.write("A direct symbolic definite integral was used; simple endpoint substitution into this primitive was not verified.")
                except (NotImplementedError, ValueError, TypeError):
                    st.write("The definite integral was computed directly; an endpoint formula was not verified.")
        else:
            st.write("This is treated as an improper integral. Split at excluded interior points, and require every one-sided piece to converge separately.")
            st.latex(r"\int_c^b f(x)\,dx=\lim_{t\to c^+}\int_t^b f(x)\,dx")
            st.write("Use the corresponding one-sided limit at each singular or infinite endpoint. No Cauchy principal value is substituted for convergence.")
        for start, end, value, state in answer["parts"]:
            st.latex(sp.latex(sp.Integral(expr, (x, start, end))) + "=" + sp.latex(value))
            if state != "finite": st.write("This piece is " + state + ".")
        if a > b: st.write("The original bounds are reversed, so negate the sum of the pieces above.")
    fig, _ = plot_functions(expr, domain, lo, hi, shade=(a, b))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green shading is above the x-axis; orange shading is below it. Shading is clipped to the viewing window and does not establish convergence.")
    if st.checkbox("Also calculate total geometric area", key="anti_total_area"):
        try:
            low, high, _ = ordered_bounds(a, b)
            area = session_result("anti_area", (token, a_text, b_text), lambda: definite_integral(sp.Abs(expr), domain, low, high))
            st.latex(r"\text{Total area}=\int_{\min(a,b)}^{\max(a,b)}|f(x)|\,dx")
            if area["status"] == "finite": st.latex("=" + sp.latex(area["value"]))
            elif area["status"] == "divergent": st.warning("The total geometric area is infinite.")
            else: st.info("The total area could not be resolved symbolically.")
        except Exception as exc:
            st.info(f"Total area could not be evaluated: {exc}")
    st.subheader("Reflection")
    st.text_area("What changes when C changes? How does signed accumulation differ from total geometric area?",
                 key="anti_reflection")
    st.caption("Your reflection stays in this session; it is not submitted to your teacher.")


if __name__ == "__main__":
    st.set_page_config(page_title="Antiderivative Visualizer", layout="wide")
    run()
