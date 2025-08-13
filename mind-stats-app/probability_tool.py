import streamlit as st
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

