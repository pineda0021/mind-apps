import streamlit as st
import numpy as np
import plotly.graph_objects as go
from math import comb, factorial

def npr(n, r):
    if r < 0 or r > n:
        return 0
    return factorial(n) // factorial(n - r)

def parse_prob(text: str) -> float:
    text = text.strip()
    if text.endswith("%"):
        val = float(text[:-1]) / 100.0
    else:
        val = float(text)
    if not (0.0 <= val <= 1.0):
        raise ValueError("Probability must be in [0, 1] (or 0%–100%).")
    return val

def run():
    st.header("🎲 Probability Tool (Advanced)")

    st.subheader("1) Combinations & Permutations")
    c1, c2 = st.columns(2)
    with c1:
        n = st.number_input("n (total items)", min_value=0, value=5, step=1)
    with c2:
        r = st.number_input("r (selected items)", min_value=0, value=2, step=1)
    if r > n:
        st.error("r cannot be greater than n.")
    else:
        st.write(f"**nCr = C(n, r):** {comb(int(n), int(r))}")
        st.write(f"**nPr (permutations):** {npr(int(n), int(r))}")

    st.markdown("---")

    st.subheader("2) Basic, Conditional, and Bayes’ Probabilities")
    colA, colB = st.columns(2)
    with colA:
        pA_in = st.text_input("P(A)", "0.5")
        pB_in = st.text_input("P(B)", "0.4")
    with colB:
        independent = st.checkbox("Assume A and B are independent", value=True)

    try:
        pA = parse_prob(pA_in)
        pB = parse_prob(pB_in)
        st.write(f"P(A) = {pA:.4f},  P(B) = {pB:.4f}")
        if independent:
            pAandB = pA * pB
            st.info(f"P(A ∩ B) = P(A)·P(B) = {pAandB:.4f}")
            st.info(f"P(A ∪ B) = P(A) + P(B) − P(A)·P(B) = {pA + pB - pAandB:.4f}")
        else:
            pAandB_in = st.text_input("P(A ∩ B) (if not independent)", "0.2")
            pAandB = parse_prob(pAandB_in)
            st.info(f"P(A ∩ B) = {pAandB:.4f}")
            st.info(f"P(A ∪ B) = {pA + pB - pAandB:.4f}")

        if pB == 0:
            st.warning("P(B) = 0 ⇒ P(A|B) undefined.")
        else:
            st.success(f"P(A|B) = P(A ∩ B) / P(B) = {pAandB/pB:.4f}")

        st.write(f"P(¬A) = {1 - pA:.4f},  P(¬B) = {1 - pB:.4f}")

        st.markdown("**Bayes’ Theorem**")
        pA_bayes = st.slider("Prior P(A)", 0.0, 1.0, 0.3)
        pBgA = st.slider("Likelihood P(B|A)", 0.0, 1.0, 0.8)
        pBgNotA = st.slider("Likelihood P(B|¬A)", 0.0, 1.0, 0.1)
        pNotA_bayes = 1 - pA_bayes
        pB_total = pBgA * pA_bayes + pBgNotA * pNotA_bayes
        if pB_total == 0:
            st.warning("P(B) = 0 with current inputs.")
        else:
            st.success(f"P(A|B) = [P(B|A)P(A)] / P(B) = {(pBgA * pA_bayes) / pB_total:.4f}")

    except Exception as e:
        st.error(f"Probability input error: {e}")

    st.markdown("---")

    st.subheader("3) Discrete Random Variable  —  E[X] and Var[X]")
    v = st.text_input("Values X (comma-separated)", "0,1,2,3")
    p = st.text_input("Probabilities P(X) (comma-separated)", "0.1,0.3,0.4,0.2")
    try:
        X = np.array([float(x.strip()) for x in v.split(",") if x.strip() != ""])
        PX = np.array([float(y.strip()) for y in p.split(",") if y.strip() != ""])
        if len(X) != len(PX):
            st.error("X and P(X) must have same length.")
        elif not np.isclose(PX.sum(), 1.0):
            st.error(f"Probabilities must sum to 1. Current sum = {PX.sum():.6f}")
        else:
            EX = float(np.sum(X * PX))
            VarX = float(np.sum(((X - EX) ** 2) * PX))
            st.success(f"E[X] = {EX:.4f},  Var[X] = {VarX:.4f},  SD = {np.sqrt(VarX):.4f}")
            fig = go.Figure(data=[go.Bar(x=X, y=PX)])
            fig.update_layout(title="PMF of X", xaxis_title="x", yaxis_title="P(X=x)", template="simple_white")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Discrete RV input error: {e}")

    st.markdown("---")

    st.subheader("4) Common Distributions")

    # Bernoulli
    with st.expander("Bernoulli(p)"):
        p_bern = st.slider("p (success prob)", 0.0, 1.0, 0.5)
        bern_x = np.array([0, 1])
        bern_p = np.array([1 - p_bern, p_bern])
        figb = go.Figure(data=[go.Bar(x=bern_x, y=bern_p)])
        figb.update_layout(title="Bernoulli PMF", xaxis_title="x", yaxis_title="P(X=x)",
                           template="simple_white", xaxis=dict(dtick=1))
        st.plotly_chart(figb, use_container_width=True)
        st.caption(f"E[X]={p_bern:.4f}, Var[X]={p_bern*(1-p_bern):.4f}")

    # Binomial (no scipy)
    with st.expander("Binomial(n, p)"):
        n_bin = st.number_input("n", min_value=1, value=10, step=1)
        p_bin = st.slider("p", 0.0, 1.0, 0.5)
        k_bin = st.number_input("k (for probability queries)", min_value=0, value=3, step=1)
        x = np.arange(0, int(n_bin) + 1)
        pmf = np.array([comb(int(n_bin), int(k)) * (p_bin ** k) * ((1 - p_bin) ** (int(n_bin) - k)) for k in x], dtype=float)
        cdf = np.cumsum(pmf)
        st.write(f"P(X = {int(k_bin)}) = {pmf[int(k_bin)]:.6f}")
        st.write(f"P(X ≤ {int(k_bin)}) = {cdf[int(k_bin)]:.6f}")
        st.write(f"P(X ≥ {int(k_bin)}) = {1 - (cdf[int(k_bin)-1] if k_bin > 0 else 0):.6f}")
        fig = go.Figure(data=[go.Bar(x=x, y=pmf)])
        fig.update_layout(title="Binomial PMF", xaxis_title="x", yaxis_title="P(X=x)", template="simple_white")
        st.plotly_chart(fig, use_container_width=True)

    # Geometric (support {1,2,...})
    with st.expander("Geometric(p) (first success on trial k)"):
        p_geo = st.slider("p", 0.0, 1.0, 0.5, key="geo_p")
        k_max = st.number_input("Display up to k =", min_value=5, value=15, step=1)
        k_vals = np.arange(1, int(k_max) + 1)
        pmf_g = (1 - p_geo) ** (k_vals - 1) * p_geo
        cdf_g = 1 - (1 - p_geo) ** k_vals
        kq = st.number_input("k (for queries)", min_value=1, value=5, step=1)
        st.write(f"P(X = {int(kq)}) = {((1-p_geo)**(int(kq)-1)*p_geo):.6f}")
        st.write(f"P(X ≤ {int(kq)}) = {1 - (1-p_geo)**int(kq):.6f}")
        st.write(f"P(X ≥ {int(kq)}) = {(1-p_geo)**(int(kq)-1):.6f}")
        figg = go.Figure(data=[go.Bar(x=k_vals, y=pmf_g)])
        figg.update_layout(title="Geometric PMF", xaxis_title="k", yaxis_title="P(X=k)", template="simple_white")
        st.plotly_chart(figg, use_container_width=True)
