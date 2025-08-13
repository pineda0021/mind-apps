import streamlit as st
from math import comb, perm

def run():
    st.header("🎲 Probability Tool")
    st.write("""
        Calculate basic probabilities, combinations (nCr), permutations (nPr),
        and probabilities of compound events (AND, OR, NOT).
    """)

    st.markdown("---")

    # Section 1: Combinations and Permutations Calculator
    st.subheader("1. Combinations and Permutations Calculator")

    n = st.number_input("Enter n (total items)", min_value=0, step=1, format="%d")
    r = st.number_input("Enter r (items selected)", min_value=0, step=1, format="%d")

    if r > n:
        st.error("r cannot be greater than n.")
    else:
        ncr = comb(n, r)
        npr = perm(n, r)
        st.write(f"**Combinations (nCr):** Number of ways to choose {r} items from {n} without order = {ncr}")
        st.write(f"**Permutations (nPr):** Number of ways to arrange {r} items from {n} with order = {npr}")

    st.markdown("---")

    # Section 2: Basic Probability Calculator
    st.subheader("2. Basic Probability Calculator")

    st.write("Enter probability values between 0 and 1 (decimals or percentages).")

    P_A_input = st.text_input("Probability of Event A (P(A))", "0.5")
    P_B_input = st.text_input("Probability of Event B (P(B))", "0.5")

    try:
        P_A = parse_probability(P_A_input)
        P_B = parse_probability(P_B_input)
    except ValueError as e:
        st.error(str(e))
        return

    st.write(f"P(A) = {P_A:.4f}")
    st.write(f"P(B) = {P_B:.4f}")

    # Checkbox if A and B are independent
    independent = st.checkbox("Are events A and B independent?", value=True)

    # Compound probability calculations
    st.subheader("3. Compound Events Probability")

    event = st.selectbox("Select compound event:", [
        "P(A and B)",
        "P(A or B)",
        "P(not A)",
        "P(not B)"
    ])

    if event == "P(A and B)":
        if independent:
            P_and = P_A * P_B
            st.success(f"P(A and B) = P(A) × P(B) = {P_A:.4f} × {P_B:.4f} = {P_and:.4f}")
        else:
            P_and_input = st.text_input("Enter P(A and B) (joint probability)")
            try:
                P_and = parse_probability(P_and_input)
                st.write(f"P(A and B) = {P_and:.4f}")
            except:
                st.warning("Enter a valid probability for P(A and B)")

    elif event == "P(A or B)":
        if independent:
            P_or = P_A + P_B - (P_A * P_B)
            st.success(f"P(A or B) = P(A) + P(B) - P(A)×P(B) = {P_A:.4f} + {P_B:.4f} - {P_A:.4f}×{P_B:.4f} = {P_or:.4f}")
        else:
            P_or_input = st.text_input("Enter P(A or B) (union probability)")
            try:
                P_or = parse_probability(P_or_input)
                st.write(f"P(A or B) = {P_or:.4f}")
            except:
                st.warning("Enter a valid probability for P(A or B)")

    elif event == "P(not A)":
        st.success(f"P(not A) = 1 - P(A) = {1-P_A:.4f}")

    elif event == "P(not B)":
        st.success(f"P(not B) = 1 - P(B) = {1-P_B:.4f}")

def parse_probability(prob_str):
    """
    Parse probability input that can be decimal or percent (e.g. "0.5" or "50%").
    Returns float in [0,1].
    """
    prob_str = prob_str.strip()
    if prob_str.endswith("%"):
        prob_val = float(prob_str[:-1]) / 100
    else:
        prob_val = float(prob_str)

    if prob_val < 0 or prob_val > 1:
        raise ValueError("Probability must be between 0 and 1 (or 0% to 100%).")

    return prob_val

if __name__ == "__main__":
    run()
