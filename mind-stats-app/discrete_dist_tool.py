import streamlit as st
import numpy as np
import plotly.graph_objects as go
from math import comb, factorial

# =========================
# Helpers (no SciPy needed)
# =========================

def parse_array(text):
    return np.array([float(t.strip()) for t in text.split(",") if t.strip() != ""], dtype=float)

# ---- Binomial ----
def binom_pmf(n, p, k):
    if k < 0 or k > n:
        return 0.0
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def binom_cdf(n, p, k):
    k = int(k)
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(binom_pmf(n, p, i) for i in range(0, k + 1))

# ---- Poisson ----
def pois_pmf(lam, k):
    if k < 0:
        return 0.0
    return np.exp(-lam) * (lam ** k) / factorial(int(k))

def pois_cdf(lam, k):
    k = int(k)
    if k < 0:
        return 0.0
    return sum(pois_pmf(lam, i) for i in range(0, k + 1))

# ---- Hypergeometric ----
# Parameters: population size N, number of successes in population K, draws n
# Support for X (# of successes drawn): max(0, n-(N-K)) ... min(n, K)
def hypergeom_pmf(N, K, n, k):
    k = int(k)
    if any(v < 0 for v in [N, K, n, k]): 
        return 0.0
    if K > N or n > N:
        return 0.0
    lo = max(0, n - (N - K))
    hi = min(n, K)
    if k < lo or k > hi:
        return 0.0
    denom = comb(N, n)
    if denom == 0:
        return 0.0
    return comb(K, k) * comb(N - K, n - k) / denom

def hypergeom_support(N, K, n):
    lo = max(0, n - (N - K))
    hi = min(n, K)
    return np.arange(lo, hi + 1, dtype=int)

def hypergeom_cdf(N, K, n, k):
    support = hypergeom_support(N, K, n)
    k = int(k)
    valid = support[support <= k]
    if valid.size == 0:
        return 0.0
    return float(sum(hypergeom_pmf(N, K, n, ki) for ki in valid))

# ---- Negative Binomial (failures before r-th success) ----
# Parameters: r (positive integer), p in (0,1)
# Support: k = 0,1,2,...  (k = number of failures before the r-th success)
def negbinom_pmf(r, p, k):
    k = int(k)
    if r <= 0 or k < 0 or not (0.0 <= p <= 1.0):
        return 0.0
    # C(k + r - 1, k) * (1-p)^k * p^r
    return comb(k + r - 1, k) * ((1 - p) ** k) * (p ** r)

def negbinom_cdf(r, p, k):
    k = int(k)
    if k < 0:
        return 0.0
    # Sum up to k
    return float(sum(negbinom_pmf(r, p, i) for i in range(0, k + 1)))


# =========================
# Streamlit App
# =========================

def run():
    st.header("🧮 Discrete Distributions Tool")

    tabs = st.tabs([
        "General Discrete (X & P(X))",
        "Binomial",
        "Poisson",
        "Hypergeometric",
        "Negative Binomial"
    ])

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

        with st.expander("Between / More / Fewer"):
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

        with st.expander("Between / More / Fewer"):
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

    # -------- Hypergeometric --------
    with tabs[3]:
        st.subheader("Hypergeometric Distribution (N, K, n)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            N = st.number_input("N (population size)", min_value=1, value=50, step=1)
        with c2:
            K = st.number_input("K (# successes in population)", min_value=0, value=10, step=1)
        with c3:
            n = st.number_input("n (draws, without replacement)", min_value=1, value=12, step=1)
        with c4:
            kq = st.number_input("k (query successes in sample)", min_value=0, value=3, step=1)

        N, K, n = int(N), int(K), int(n)
        if K > N:
            st.error("K cannot exceed N.")
        elif n > N:
            st.error("n cannot exceed N.")
        else:
            support = hypergeom_support(N, K, n)
            pmf_vals = np.array([hypergeom_pmf(N, K, n, k) for k in support])
            cdf_vals = np.cumsum(pmf_vals)

            # Adjust kq to valid support for display
            kq_int = int(kq)
            if kq_int < support[0]:
                kq_int = support[0]
            if kq_int > support[-1]:
                kq_int = support[-1]

            st.write(f"P(X = {int(kq)}) = {hypergeom_pmf(N, K, n, int(kq)):.6f}")
            # For ≤ and ≥, clip to support safely
            cdf_at_k = float(sum(hypergeom_pmf(N, K, n, k) for k in support if k <= int(kq)))
            st.write(f"P(X ≤ {int(kq)}) = {cdf_at_k:.6f}")
            cdf_before = float(sum(hypergeom_pmf(N, K, n, k) for k in support if k < int(kq)))
            st.write(f"P(X ≥ {int(kq)}) = {1 - cdf_before:.6f}")

            with st.expander("Between / More / Fewer"):
                a = st.number_input("a (lower, inclusive)", min_value=int(support[0]), value=int(support[0]), step=1, key="hg_a")
                b = st.number_input("b (upper, inclusive)", min_value=int(support[0]), value=int(support[-1]), step=1, key="hg_b")
                a, b = int(a), int(b)
                if a > b:
                    a, b = b, a
                between = float(sum(hypergeom_pmf(N, K, n, k) for k in support if a <= k <= b))
                st.write(f"P({a} ≤ X ≤ {b}) = {between:.6f}")

                more = st.number_input("More than x (strict)", min_value=int(support[0]), value=min(int(support[-1]), int(support[0])+1), step=1, key="hg_more")
                fewer = st.number_input("Fewer than x (strict)", min_value=int(support[0]), value=int(support[0]), step=1, key="hg_less")
                p_more = float(sum(hypergeom_pmf(N, K, n, k) for k in support if k > int(more)))
                p_less = float(sum(hypergeom_pmf(N, K, n, k) for k in support if k < int(fewer)))
                st.write(f"P(X > {int(more)}) = {p_more:.6f}")
                st.write(f"P(X < {int(fewer)}) = {p_less:.6f}")

            figh = go.Figure(data=[go.Bar(x=support, y=pmf_vals)])
            figh.update_layout(
                title=f"Hypergeometric PMF (N={N}, K={K}, n={n})",
                xaxis_title="k (successes in sample)",
                yaxis_title="P(X=k)",
                template="simple_white",
                xaxis=dict(dtick=1)
            )
            st.plotly_chart(figh, use_container_width=True)

    # -------- Negative Binomial --------
    with tabs[4]:
        st.subheader("Negative Binomial (failures before r-th success)")
        c1, c2, c3 = st.columns(3)
        with c1:
            r = st.number_input("r (number of successes)", min_value=1, value=3, step=1)
        with c2:
            p = st.slider("p (success probability)", 0.0, 1.0, 0.5, key="nb_p")
        with c3:
            kq = st.number_input("k (failures before r-th success)", min_value=0, value=4, step=1)

        r = int(r)
        # Display up to kmax for chart
        kmax = st.number_input("Max k to display", min_value=max(10, int(kq)+1), value=max(15, int(kq)+5), step=1, key="nb_kmax")
        k_vals = np.arange(0, int(kmax) + 1, dtype=int)
        pmf = np.array([negbinom_pmf(r, p, int(k)) for k in k_vals])
        cdf = np.cumsum(pmf)

        st.write(f"P(X = {int(kq)}) = {negbinom_pmf(r, p, int(kq)):.6f}")
        st.write(f"P(X ≤ {int(kq)}) = {cdf[int(kq)]:.6f}")
        geq = 1 - (cdf[int(kq) - 1] if kq > 0 else 0.0)
        st.write(f"P(X ≥ {int(kq)}) = {geq:.6f}")

        with st.expander("Between / More / Fewer"):
            a = st.number_input("a (lower, inclusive)", min_value=0, value=2, step=1, key="nb_a")
            b = st.number_input("b (upper, inclusive)", min_value=0, value=8, step=1, key="nb_b")
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            # ensure arrays big enough
            need = max(b, int(kmax))
            if need > int(kmax):
                k_vals2 = np.arange(0, need + 1, dtype=int)
                pmf2 = np.array([negbinom_pmf(r, p, int(k)) for k in k_vals2])
                cdf2 = np.cumsum(pmf2)
            else:
                k_vals2, pmf2, cdf2 = k_vals, pmf, cdf

            between = cdf2[b] - (cdf2[a - 1] if a > 0 else 0.0)
            st.write(f"P({a} ≤ X ≤ {b}) = {between:.6f}")

            more = st.number_input("More than x (strict)", min_value=0, value=6, step=1, key="nb_more")
            less = st.number_input("Fewer than x (strict)", min_value=0, value=3, step=1, key="nb_less")
            # extend if needed
            need_more = int(more)
            need_less = int(less)
            need_max = max(need_more, need_less, int(kmax))
            if need_max > int(kmax):
                k_vals3 = np.arange(0, need_max + 1, dtype=int)
                pmf3 = np.array([negbinom_pmf(r, p, int(k)) for k in k_vals3])
                cdf3 = np.cumsum(pmf3)
            else:
                cdf3 = cdf
            p_more = float(1 - cdf3[int(more)])
            p_less = float(cdf3[int(less) - 1] if less > 0 else 0.0)
            st.write(f"P(X > {int(more)}) = {p_more:.6f}")
            st.write(f"P(X < {int(less)}) = {p_less:.6f}")

        fignb = go.Figure(data=[go.Bar(x=k_vals, y=pmf)])
        fignb.update_layout(
            title=f"Negative Binomial PMF (r={r}, p={p})",
            xaxis_title="k (failures before r-th success)",
            yaxis_title="P(X=k)",
            template="simple_white",
            xaxis=dict(dtick=1)
        )
        st.plotly_chart(fignb, use_container_width=True)
