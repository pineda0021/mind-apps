import streamlit as st
import numpy as np
import plotly.graph_objects as go
from math import comb, factorial

# ---------- Utilities (no scipy needed) ----------

def binom_pmf(n, p, k):
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def binom_cdf(n, p, k):
    k = int(k)
    return sum(binom_pmf(n, p, i) for i in range(0, k + 1))

def pois_pmf(lam, k):
    # e^{-λ} λ^k / k!
    return np.exp(-lam) * (lam ** k) / factorial(k)

def pois_cdf(lam, k):
    k = int(k)
    return sum(pois_pmf(lam, i) for i in range(0, k + 1))

def parse_array(text):
    return np.array([float(t.strip()) for t in text.split(",") if t.strip() != ""], dtype=float)

# ---------- Streamlit App ----------

def run():
    st.header("🧮 Discrete Distributions Tool")

    tabs = st.tabs(["General Discrete (X & P(X))", "Binomial", "Poisson"])

    # -------- General Discrete --------
    with tabs[0]:
        st.subheader("General Discrete Distribution")
        c1, c2 = st.columns(2)
        with c1:
            xs = st.text_input("Values X (comma-separated)", "0,1,2,3")
        with c2:
            ps = st.text_input("Probabilities P(X) (comma-separated)", "0.1,0.3,0.4,0.2")

        try:
            X = parse_array(xs)
            P = parse_array(ps)
            if len(X) != len(P):
                st.error("X and P(X) must have the same length.")
            elif not np.isclose(P.sum(), 1.0):
                st.error(f"Probabilities must sum to 1. Current sum = {P.sum():.6f}")
            else:
                # Summary
                EX = float(np.sum(X * P))
                VarX = float(np.sum(((X - EX) ** 2) * P))
                CDF = np.cumsum(P)

                st.write(f"**E[X] = {EX:.6f}**,  **Var[X] = {VarX:.6f}**,  **SD = {np.sqrt(VarX):.6f}**")

                # Table
                st.write("**Table**")
                table = []
                for xi, pi, ci in zip(X, P, CDF):
                    table.append({"X": float(xi), "P(X)": float(pi), "Cumulative P(X≤x)": float(ci)})
                st.dataframe(table, use_container_width=True)

                # Plot PMF
                fig = go.Figure(data=[go.Bar(x=X, y=P)])
                fig.update_layout(title="PMF: P(X=x)", xaxis_title="x", yaxis_title="P(X=x)",
                                  template="simple_white")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Input error: {e}")

    # -------- Binomial --------
    with tabs[1]:
        st.subheader("Binomial Distribution (n, p)")
        c1, c2, c3 = st.columns(3)
        with c1:
            n = st.number_input("n (trials)", min_value=1, value=10, step=1)
        with c2:
            p = st.slider("p (success probability)", 0.0, 1.0, 0.5)
        with c3:
            kq = st.number_input("k (query value)", min_value=0, value=3, step=1)

        n = int(n)
        x = np.arange(0, n + 1)
        pmf = np.array([binom_pmf(n, p, int(k)) for k in x])
        cdf = np.cumsum(pmf)

        st.write(f"P(X = {int(kq)}) = {pmf[int(kq)]:.6f}")
        st.write(f"P(X ≤ {int(kq)}) = {cdf[int(kq)]:.6f}")
        geq = 1 - (cdf[int(kq) - 1] if kq > 0 else 0.0)
        st.write(f"P(X ≥ {int(kq)}) = {geq:.6f}")

        with st.expander("Between/More/Fewer"):
            a = st.number_input("a (lower, inclusive)", min_value=0, value=2, step=1)
            b = st.number_input("b (upper, inclusive)", min_value=0, value=6, step=1)
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            between = cdf[b] - (cdf[a - 1] if a > 0 else 0.0)
            st.write(f"P({a} ≤ X ≤ {b}) = {between:.6f}")

            x_more = st.number_input("More than x (strict)", min_value=0, value=5, step=1, key="bin_more")
            st.write(f"P(X > {int(x_more)}) = {1 - cdf[int(x_more)]:.6f}")

            x_less = st.number_input("Fewer than x (strict)", min_value=0, value=5, step=1, key="bin_less")
            st.write(f"P(X < {int(x_less)}) = {(cdf[int(x_less) - 1] if x_less > 0 else 0.0):.6f}")

        # Plot
        figb = go.Figure(data=[go.Bar(x=x, y=pmf)])
        figb.update_layout(title=f"Binomial PMF (n={n}, p={p})", xaxis_title="x", yaxis_title="P(X=x)",
                           template="simple_white")
        st.plotly_chart(figb, use_container_width=True)

    # -------- Poisson --------
    with tabs[2]:
        st.subheader("Poisson Distribution (λ)")
        c1, c2, c3 = st.columns(3)
        with c1:
            lam = st.number_input("λ (rate > 0)", min_value=0.0001, value=3.0, step=0.5, format="%.4f")
        with c2:
            kq = st.number_input("k (query value)", min_value=0, value=2, step=1, key="pois_k")
        with c3:
            kmax = st.number_input("Max k to display", min_value=5, value=20, step=1)

        lam = float(lam)
        x = np.arange(0, int(kmax) + 1)
        pmf = np.array([pois_pmf(lam, int(k)) for k in x])
        cdf = np.cumsum(pmf)

        st.write(f"P(X = {int(kq)}) = {pmf[int(kq)]:.6f}")
        st.write(f"P(X ≤ {int(kq)}) = {cdf[int(kq)]:.6f}")
        geq = 1 - (cdf[int(kq) - 1] if kq > 0 else 0.0)
        st.write(f"P(X ≥ {int(kq)}) = {geq:.6f}")

        with st.expander("Between/More/Fewer"):
            a = st.number_input("k1 (lower, inclusive)", min_value=0, value=1, step=1, key="pois_a")
            b = st.number_input("k2 (upper, inclusive)", min_value=0, value=5, step=1, key="pois_b")
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            between = (cdf[b] - (cdf[a - 1] if a > 0 else 0.0))
            st.write(f"P({a} ≤ X ≤ {b}) = {between:.6f}")

            more = st.number_input("More than k (strict)", min_value=0, value=6, step=1, key="pois_more")
            st.write(f"P(X > {int(more)}) = {1 - cdf[int(more)]:.6f}")

            less = st.number_input("Fewer than k (strict)", min_value=0, value=2, step=1, key="pois_less")
            st.write(f"P(X < {int(less)}) = {(cdf[int(less) - 1] if less > 0 else 0.0):.6f}")

        figp = go.Figure(data=[go.Bar(x=x, y=pmf)])
        figp.update_layout(title=f"Poisson PMF (λ={lam:g})", xaxis_title="k", yaxis_title="P(X=k)",
                           template="simple_white")
        st.plotly_chart(figp, use_container_width=True)
