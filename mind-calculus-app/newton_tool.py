# -*- coding: utf-8 -*-
"""Newton tool: import newton_tool and call newton_tool.run()."""

import json
import math
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

x = sp.Symbol("x", real=True)

# A recursive-descent JavaScript parser: LaTeX -> a small expression tree.
# No JavaScript eval(), Python eval(), or sympify(student_text) is used.
PARSER_JS = r"""
function parseLatex(source) {
  if (!source.trim() || source.length > 300)
    throw Error('Enter an expression with 1–300 characters.');
  const s = source.replace(/\*\*/g, '^').replace(/\\(?:left|right)\b/g, '')
    .replace(/\\[,!;: ]/g, ' ')
    .replace(/\\operatorname\{(sin|cos|tan|ln|log|exp|abs|cbrt)\}/g, '\\$1');
  const tokens = [];
  const pattern = /\s+|\\[a-zA-Z]+|(?:\d+(?:\.\d*)?|\.\d+)|[xe+\-*/^(){}|\[\]]/y;
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
  const funcs = ['sin', 'cos', 'tan', 'ln', 'log', 'exp', 'abs', 'cbrt'];
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
    else if (t === '\\sqrt') {
      let kind = 'sqrt';
      if (take('[')) {
        const index = tokens[i++]; expect(']');
        if (!['2', '3'].includes(index)) throw Error('Only square and cube roots are supported.');
        kind = index === '3' ? 'cbrt' : 'sqrt';
      }
      v = [kind, group()];
    }
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
  const root = parentElement.querySelector('.newton-editor');
  const rows = root.querySelector('.rows'), status = root.querySelector('.status');
  const drafts = window.__newtonDrafts ||= new Map();
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
      [['Fraction','\\frac'],['Power','^'],['Square root','\\sqrt'],['Cube root','cube-root'],
       ['sin','\\sin'],['cos','\\cos'],['ln','\\ln'],['π','\\pi']].forEach(([label, cmd]) => {
        const b = document.createElement('button'); b.type = 'button'; b.textContent = label;
        b.onclick = () => {
          mq.focus();
          if (cmd === 'cube-root') { mq.write('\\sqrt[3]{}'); mq.keystroke('Left'); }
          else mq.cmd(cmd);
        };
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
        "newton_mathquill",
        html='''<div class="newton-editor"><div class="rows"></div>
          <button class="analyze" type="button">Analyze</button>
          <p class="status" role="status" aria-live="polite">Enter your function, then select Analyze.</p></div>''',
        css="""
        .newton-editor {font-family: sans-serif; color: var(--st-text-color);}
        .newton-editor .entry {margin-bottom: 18px;}
        .newton-editor .label {display: block; margin-bottom: 8px;}
        .newton-editor .mathfield {display: block; min-height: 55px; padding: 12px;
          border: 2px solid #64748b; border-radius: 6px; font-size: 24px;
          background: white; color: #111827; overflow-x: auto;}
        .newton-editor button {padding: 8px 12px; margin: 6px 5px 6px 0;
          border: 1px solid #64748b; border-radius: 6px; cursor: pointer;
          background: #f1f5f9; color: #111827;}
        .newton-editor button:focus-visible, .newton-editor input:focus-visible
          {outline: 3px solid #2563eb; outline-offset: 2px;}
        .newton-editor .analyze {background: #1d4ed8; color: white; font-weight: bold;}
        .newton-editor .plain-label {display: block; font-size: 14px;}
        .newton-editor .plain {display: block; width: 100%; box-sizing: border-box;
          padding: 9px; border: 1px solid #64748b; border-radius: 5px; margin-top: 4px;}
        .newton-editor .status {font-size: 14px; min-height: 20px;}
        """,
        js=EDITOR_JS,
        isolate_styles=False,
    )


class Cbrt(sp.Function):
    """Real cube root, including negative arguments."""
    nargs = 1

    @classmethod
    def eval(cls, argument):
        if argument.is_number and argument.is_real is True:
            return sp.real_root(argument, 3)

    def _eval_is_real(self):
        if self.args[0].is_real is True:
            return True

    def _eval_power(self, exponent):
        if exponent.is_Integer and exponent % 3 == 0:
            return self.args[0] ** (exponent / 3)

    def _latex(self, printer):
        return r"\sqrt[3]{%s}" % printer._print(self.args[0])

    def fdiff(self, argindex=1):
        if argindex != 1:
            raise sp.core.function.ArgumentIndexError(self, argindex)
        return 1 / (3 * Cbrt(self.args[0])**2)


def build_expression(tree, depth=0):
    """Build a trusted expression and preserve original domain constraints."""
    if depth > 25 or not isinstance(tree, list) or not tree:
        raise ValueError("Please use a shorter, supported expression.")
    op, *items = tree
    if op == "num":
        if len(items) != 1 or not isinstance(items[0], str) or not re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)", items[0]):
            raise ValueError("Invalid number.")
        if len(items[0]) > 20:
            raise ValueError("Use numbers with at most 20 digits.")
        return sp.Rational(items[0]), []
    constants = {"x": x, "e": sp.E, "pi": sp.pi}
    if op in constants and not items:
        return constants[op], []
    arities = {**dict.fromkeys(["add", "sub", "mul", "div", "pow"], 2),
               **dict.fromkeys(["neg", "sqrt", "cbrt", "sin", "cos", "tan", "ln", "log", "exp", "abs"], 1)}
    if op not in arities or len(items) != arities[op]:
        raise ValueError("Use x and the listed functions.")
    children = [build_expression(t, depth + 1) for t in items]
    a = [item[0] for item in children]
    guards = [guard for item in children for guard in item[1]]
    if op == "add": expr = sp.Add(*a, evaluate=False)
    elif op == "sub": expr = sp.Add(a[0], sp.Mul(-1, a[1], evaluate=False), evaluate=False)
    elif op == "mul": expr = sp.Mul(*a, evaluate=False)
    elif op == "div":
        expr = sp.Mul(a[0], sp.Pow(a[1], -1, evaluate=False), evaluate=False)
        guards.append((a[1], "nonzero"))
    elif op == "pow":
        exponent = sp.simplify(a[1])
        if exponent.is_number and exponent.is_real is True and abs(exponent) > 100:
            raise ValueError("Use exponents between -100 and 100.")
        if isinstance(exponent, sp.Rational) and exponent.q == 3:
            expr = sp.Pow(Cbrt(a[0]), exponent.p, evaluate=False)
            if exponent < 0: guards.append((a[0], "nonzero"))
        else:
            expr = sp.Pow(a[0], exponent, evaluate=False)
            if exponent.is_Integer:
                if exponent < 0: guards.append((a[0], "nonzero"))
            elif exponent.is_number and exponent.is_positive is True:
                guards.append((a[0], "nonnegative"))
            else:
                guards.append((a[0], "positive"))
    elif op == "neg": expr = sp.Mul(-1, a[0], evaluate=False)
    elif op == "sqrt":
        expr = sp.Pow(a[0], sp.Rational(1, 2), evaluate=False)
        guards.append((a[0], "nonnegative"))
    elif op == "cbrt": expr = Cbrt(a[0])
    elif op in ("ln", "log"):
        expr = sp.log(a[0], evaluate=False)
        if op == "log": expr = sp.Mul(expr, 1/sp.log(10), evaluate=False)
        guards.append((a[0], "positive"))
    else:
        fn = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "abs": sp.Abs}[op]
        expr = fn(a[0], evaluate=False)
        if op == "tan": guards.append((sp.cos(a[0]), "nonzero"))
    return expr, guards


def make_numeric(expr, guards):
    modules = [{"Cbrt": np.cbrt}, "numpy"]
    function = sp.lambdify(x, expr, modules=modules)
    checks = [(sp.lambdify(x, g, modules=modules), kind) for g, kind in guards]

    def evaluate(points):
        xx = np.asarray(points, dtype=float)
        with np.errstate(all="ignore"):
            values = np.broadcast_to(np.asarray(function(xx), dtype=complex), xx.shape)
            valid = np.isfinite(values.real) & (values.imag == 0)
            for check, kind in checks:
                raw = np.broadcast_to(np.asarray(check(xx), dtype=complex), xx.shape)
                valid = valid & np.isfinite(raw.real) & (raw.imag == 0)
                if kind == "nonzero": valid = valid & (raw.real != 0)
                elif kind == "positive": valid = valid & (raw.real > 0)
                else: valid = valid & (raw.real >= 0)
            result = np.where(valid, values.real, np.nan)
        return float(result) if xx.ndim == 0 else result

    return evaluate


def compile_problem(tree_json):
    expr, guards = build_expression(json.loads(tree_json))
    derivative = sp.simplify(sp.diff(expr, x))
    function = make_numeric(expr, guards)
    raw_derivative = make_numeric(derivative, guards)
    # sign(0)=0 in a symbolic formula does not establish a derivative at a corner.
    corner_arguments = {item.args[0] for item in expr.atoms(sp.Abs)}
    corner_arguments.update(item.args[0] for item in derivative.atoms(sp.sign, sp.Abs))
    corner_functions = [make_numeric(argument, guards) for argument in corner_arguments]

    def checked_derivative(point):
        value = raw_derivative(point)
        if not math.isfinite(value):
            return math.nan
        if any(safe_evaluate(g, point) == 0 for g in corner_functions):
            try:
                centre = sp.Rational(str(float(point)))
                rewritten = expr.replace(
                    lambda e: e.func == Cbrt,
                    lambda e: sp.sign(e.args[0])*sp.Abs(e.args[0])**sp.Rational(1, 3),
                )
                quotient = (rewritten - rewritten.subs(x, centre))/(x-centre)
                left = sp.limit(quotient, x, centre, dir="-")
                right = sp.limit(quotient, x, centre, dir="+")
                if (left.is_real is True and left.is_finite is True
                        and right.is_real is True and right.is_finite is True
                        and sp.simplify(left-right) == 0):
                    return float(left)
                return math.nan
            except (NotImplementedError, ValueError, TypeError):
                return math.nan
        return value

    return expr, derivative, guards, function, checked_derivative


def safe_evaluate(function, value):
    try:
        with np.errstate(all="ignore"):
            result = complex(function(value))
        if result.imag != 0 or not math.isfinite(result.real):
            return math.nan
        return float(result.real)
    except (ArithmeticError, ValueError, TypeError, NameError):
        return math.nan


def newton_iterations(function, derivative, x0, tol, max_iter, magnitude_limit=1e12):
    """Perform at most max_iter updates, and evaluate the final iterate."""
    if not math.isfinite(x0) or not math.isfinite(tol) or tol <= 0:
        raise ValueError("Use a finite initial guess and a positive finite tolerance.")
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError("The maximum number of iterations must be a positive integer.")
    if not math.isfinite(magnitude_limit) or magnitude_limit <= 0 or abs(x0) > magnitude_limit:
        raise ValueError("The magnitude limit must be positive and include the initial guess.")
    rows, seen = [], {}
    xn = float(x0)
    previous_residual, growing_streak, longest_growth = None, 0, 0

    def add(n, fn, fpn, next_x, status, detail=""):
        rows.append({"n": n, "x_n": xn, "f(x_n)": fn, "f′(x_n)": fpn,
                     "x_(n+1)": next_x, "status": status, "detail": detail})

    for n in range(max_iter + 1):
        fn = safe_evaluate(function, xn)
        if not math.isfinite(fn):
            add(n, fn, math.nan, math.nan, "non_finite_function")
            break
        # Check f first: x=0 is a root of cbrt(x), despite its infinite slope.
        if abs(fn) <= tol:
            add(n, fn, safe_evaluate(derivative, xn), math.nan, "converged")
            break
        if xn in seen:
            period = n - seen[xn]
            add(n, fn, safe_evaluate(derivative, xn), math.nan, "cycle",
                f"The floating-point iteration repeats with period {period}.")
            break
        seen[xn] = n
        if n == max_iter:
            add(n, fn, safe_evaluate(derivative, xn), math.nan, "max_iter")
            break
        fpn = safe_evaluate(derivative, xn)
        if not math.isfinite(fpn):
            add(n, fn, fpn, math.nan, "non_finite_derivative")
            break
        if fpn == 0:
            add(n, fn, fpn, math.nan, "derivative_zero")
            break
        residual = abs(fn)
        if previous_residual is not None and residual > previous_residual * 1.05:
            growing_streak += 1
            longest_growth = max(longest_growth, growing_streak)
        else:
            growing_streak = 0
        previous_residual = residual
        try:
            next_x = xn - fn/fpn
        except ArithmeticError:
            next_x = math.nan
        if not math.isfinite(next_x):
            add(n, fn, fpn, next_x, "non_finite_step")
            break
        if abs(next_x) > magnitude_limit:
            add(n, fn, fpn, next_x, "range_limit")
            break
        if next_x == xn:
            add(n, fn, fpn, next_x, "stagnation")
            break
        add(n, fn, fpn, next_x, "iter")
        xn = float(next_x)
    return rows, longest_growth


STATUS_MESSAGES = {
    "converged": "The residual tolerance was met: |f(x_n)| ≤ tolerance.",
    "max_iter": "The maximum number of updates was reached before the residual tolerance was met.",
    "derivative_zero": "The derivative formula is zero here, so the Newton quotient cannot be computed.",
    "non_finite_function": "The current function value is outside the real domain or is not finite.",
    "non_finite_derivative": "A finite derivative value is unavailable at this iterate.",
    "non_finite_step": "The proposed Newton step is not finite.",
    "range_limit": "The proposed next iterate exceeds the selected magnitude limit. It was not evaluated.",
    "cycle": "The floating-point iteration has entered a repeating cycle without meeting the tolerance.",
    "stagnation": "The update no longer changes x in floating-point arithmetic, but the residual remains too large.",
}


def latex_number(value, digits=10):
    return sp.latex(sp.Float(float(value), digits)) if math.isfinite(value) else r"\text{undefined}"


def make_plot(function, guards, row, half_window, tangent_half_window, include_next):
    xn, fn, fpn, next_x = row["x_n"], row["f(x_n)"], row["f′(x_n)"], row["x_(n+1)"]
    if include_next and math.isfinite(next_x):
        half_window = max(half_window, abs(next_x-xn)*1.1)
    lo, hi = xn-half_window, xn+half_window
    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        raise ValueError("This window is too small or too large for the current floating-point iterate.")
    cuts = {lo, hi}
    if lo < xn < hi: cuts.add(xn)
    for expression, _ in guards:
        try:
            zeros = sp.solveset(expression, x, domain=sp.Interval(lo, hi))
            if isinstance(zeros, sp.FiniteSet): cuts.update(float(p) for p in zeros)
        except (NotImplementedError, ValueError, TypeError):
            pass
    fig = go.Figure()
    shown = False
    cuts = sorted(cuts)
    for start, end in zip(cuts, cuts[1:]):
        xx = np.linspace(start, end, max(30, int(800*(end-start)/(hi-lo))))[1:-1]
        try:
            yy = np.asarray(function(xx), dtype=float)
            yy = np.broadcast_to(yy, xx.shape)
        except (ArithmeticError, ValueError, TypeError, NameError):
            yy = np.asarray([safe_evaluate(function, p) for p in xx])
        # Keep NaNs in the trace so missing values never become a connecting line.
        if np.any(np.isfinite(yy)):
            fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="f(x)", legendgroup="f",
                                    showlegend=not shown, line=dict(color="#2563eb", width=3), connectgaps=False))
            shown = True
    if math.isfinite(fn):
        fig.add_trace(go.Scatter(x=[xn], y=[fn], mode="markers", name="(x_n, f(x_n))",
                                marker=dict(color="#111827", size=10)))
        if math.isfinite(fpn):
            start, end = max(lo, xn-tangent_half_window), min(hi, xn+tangent_half_window)
            if math.isfinite(next_x) and lo <= next_x <= hi:
                start, end = min(start, next_x), max(end, next_x)
            xx = np.linspace(start, end, 100)
            with np.errstate(all="ignore"):
                yy = fn + fpn*(xx-xn)
            yy = np.where(np.isfinite(yy), yy, np.nan)
            fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="Newton tangent line",
                                    line=dict(color="#dc2626", width=2, dash="dash"), connectgaps=False))
        fig.add_shape(type="line", x0=xn, x1=xn, y0=0, y1=fn,
                      line=dict(color="#64748b", dash="dot"))
    if math.isfinite(next_x):
        fig.add_trace(go.Scatter(x=[next_x], y=[0], mode="markers", name="Proposed x_(n+1)",
                                marker=dict(color="#16a34a", size=11, symbol="diamond")))
    fig.add_hline(y=0, line_color="#64748b", line_width=1)
    fig.update_layout(xaxis=dict(title="x", range=[lo, hi]), yaxis_title="f(x)", height=480,
                      margin=dict(l=20, r=20, t=20, b=20), title=f"Newton iteration n = {row['n']}")
    return fig, lo, hi


def run():
    st.header("Newton’s Method Explorer")
    st.write("Use tangent lines to seek a solution of f(x) = 0, and explore how the starting value changes the outcome.")
    st.latex(r"x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}")
    if not hasattr(st.components, "v2"):
        st.error("This editor requires Streamlit 1.51 or newer. Update Streamlit in requirements.txt.")
        return
    st.caption("Type / for a fraction and ^ for a power. Use arrow keys to leave an exponent or denominator. Select Analyze after editing.")
    with st.expander("Examples and input help"):
        st.code("x^3+x-3\nx^2-2\nx^3-2x+2\n\\sqrt[3]{x}\nx^(1/3)\nx**(1/3)\n\\ln(x)", language="latex")
        st.write("Cube roots are real for negative inputs. Both x^(1/3) and x**(1/3) are accepted in the LaTeX input box. Powers with an exact reduced exponent p/3 use the real cube-root convention. Other noninteger powers require a nonnegative base (a positive base when needed).")
        st.write("Supports x, arithmetic, square and cube roots, sin, cos, tan, exp, ln, log, absolute value, e, and π. Trig uses radians; ln is natural log and log is base 10. Enter only f(x), without '= 0'.")
        st.write("Try x^2-2 from x₀=1 for convergence; x^3-2x+2 from x₀=0 for a cycle; and the cube root of x from a nonzero starting value to explore growing iterates.")
    result = get_editor()(data={"labels": ["Function f(x)"], "initial": ["x^3+x-3"]},
                          key="newton_equation", on_value_change=lambda: None)
    if result.value is None:
        st.info("Select Analyze above to begin.")
        return
    try:
        payload = result.value
        if len(json.dumps(payload)) > 16000 or len(payload["asts"]) != 1:
            raise ValueError("Please use a shorter expression.")
        token = json.dumps(payload["asts"][0])
        previous = st.session_state.get("newton_compiled")
        if previous is None or previous[0] != token:
            previous = (token, compile_problem(token))
            st.session_state["newton_compiled"] = previous
        expr, derivative, guards, f_num, fp_num = previous[1]
    except Exception as exc:
        st.error(f"Please check your expression: {exc}")
        return
    st.latex("f(x)=" + sp.latex(expr))
    st.latex("f'(x)=" + sp.latex(derivative))
    st.caption("The derivative formula is used where the real derivative exists. Newton’s method requires a usable nonzero derivative unless the root tolerance is already met.")
    c1, c2, c3 = st.columns(3)
    with c1: x0 = st.number_input("Initial guess x₀", value=1.0, step=0.1, key="newton_x0")
    with c2: tol = st.number_input("Residual tolerance |f(x_n)|", min_value=1e-14, max_value=1.0, value=1e-6, format="%.1e", key="newton_tol")
    with c3: max_iter = st.slider("Maximum Newton updates", 1, 100, 15, key="newton_max_iter")
    with st.expander("Graph and numerical settings"):
        c1, c2 = st.columns(2)
        with c1:
            half_window = st.number_input("Plot half-window", min_value=0.000001, value=1.25, format="%.6f", key="newton_window")
            tangent_window = st.number_input("Tangent half-window", min_value=0.000001, value=0.75, format="%.6f", key="newton_tangent")
        with c2:
            magnitude_limit = st.number_input("Maximum allowed |x_n|", min_value=1.0, value=1e12, format="%.1e", key="newton_limit")
            include_next = st.checkbox("Expand the view to include the proposed next point", value=False, key="newton_include_next")
        st.caption("The magnitude limit is a stopping guard; reaching it is not a proof of mathematical divergence.")
    try:
        rows, growth = newton_iterations(f_num, fp_num, x0, tol, int(max_iter), magnitude_limit)
    except ValueError as exc:
        st.error(str(exc))
        return
    last = rows[-1]
    status = last["status"]
    if status == "converged": st.success(STATUS_MESSAGES[status])
    else: st.warning(STATUS_MESSAGES[status])
    if last["detail"]: st.info(last["detail"])
    if growth >= 3 and status != "converged":
        st.info("The residual grew by more than 5% on several consecutive evaluated steps. This suggests possible divergence but is not a proof; the run stops only for the stated reason.")
    st.subheader("Iteration table")
    st.dataframe([{k: v for k, v in row.items() if k != "detail"} for row in rows],
                 use_container_width=True, hide_index=True,
                 column_config={name: st.column_config.NumberColumn(name, format="%.10g")
                                for name in ["x_n", "f(x_n)", "f′(x_n)", "x_(n+1)"]})
    st.caption("Each row shows an evaluated iterate. The final row includes the point reached after the last allowed update. Blank next-point entries mean no further step was proposed.")
    st.subheader("Inspect a Newton step")
    signature = (token, x0, tol, max_iter, magnitude_limit)
    if st.session_state.get("newton_run_signature") != signature:
        st.session_state["newton_run_signature"] = signature
        st.session_state["newton_step"] = 0
    if len(rows) == 1:
        index = 0
        st.caption("Only the initial point was evaluated.")
    else:
        index = st.slider("Iteration n", 0, len(rows)-1, key="newton_step")
    row = rows[index]
    if math.isfinite(row["f(x_n)"]) and math.isfinite(row["f′(x_n)"]) and row["f′(x_n)"] != 0 and math.isfinite(row["x_(n+1)"]):
        st.latex(r"x_{n+1}=" + latex_number(row["x_n"]) + r"-\frac{" + latex_number(row["f(x_n)"]) + "}{" + latex_number(row["f′(x_n)"]) + "}"
                 + r"\approx " + latex_number(row["x_(n+1)"]))
    else:
        st.write(STATUS_MESSAGES.get(row["status"], "No finite Newton update is available for this row."))
    try:
        fig, lo, hi = make_plot(f_num, guards, row, half_window, tangent_window, include_next)
        st.plotly_chart(fig, use_container_width=True)
        next_x = row["x_(n+1)"]
        if math.isfinite(next_x) and not lo <= next_x <= hi:
            st.caption("The proposed next point is outside this window. Enable the expanded view in the graph settings to include it.")
        st.caption("The green diamond is the proposed tangent-line intercept. Curves are numerical samples; known domain breaks are separated.")
    except (ArithmeticError, ValueError, TypeError) as exc:
        st.info(f"This step could not be plotted: {exc}")
    st.subheader("Result")
    st.write("Approximate solution satisfying the residual tolerance:" if status == "converged" else "Last evaluated iterate (convergence was not established):")
    st.latex("x_{" + str(last["n"]) + r"}\approx " + latex_number(last["x_n"], 12))
    if math.isfinite(last["f(x_n)"]):
        st.latex(r"|f(x_n)|\approx " + latex_number(abs(last["f(x_n)"]), 6))
    else:
        st.write("The function value at this iterate is undefined or not finite.")
    st.caption("A small residual is not a guaranteed bound on the distance to a root. Different starting values can lead to different outcomes.")
    st.subheader("Reflection")
    st.text_area("How did the initial guess affect the outcome? What did the tangent lines and residuals tell you?", key="newton_reflection")
    st.caption("Your reflection stays in this session; it is not submitted to your teacher.")


if __name__ == "__main__":
    st.set_page_config(page_title="Newton’s Method Explorer", layout="wide")
    run()
