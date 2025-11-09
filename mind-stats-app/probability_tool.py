# ==========================================================
# probability_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# Built with the students in MIND
# ==========================================================

import streamlit as st
from math import comb, perm

# ---------- UI Helper ----------
def step_box(text: str):
    st.markdown(
        f"""
        <div style="background-color:#f0f6ff;padding:10px;border-radius:10px;
        border-left:5px solid #007acc;margin-bottom:10px;">
        <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Parser ----------
def parse_probability(prob_str):
    """
    Parse probability input that can be decimal or percent (e.g. "0.5" or "50%").
    Returns float in [0,1] or raises ValueError with a friendly message.
    """
    prob_str = prob_str.strip()
    if prob_str.endswith("%"):
        prob_val = float(prob_str[:-1]) / 100
    else:
        prob_val = float(prob_str)

    if prob_val < 0 or prob_val > 1:
        raise ValueError("Probability must be between 0 and 1 (or 0% to 100%).")
    return prob_val

# ---------- Main ----------
def run():
    st.header("🎲 Probability Tool")
    st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — Built with the students in MIND")

    st.write("""
    Calculate **combinations** (\\(n\\choose r\\)), **permutations** (\\(P(n,r)\\)),
    and probabilities of compound events (**AND, OR, NOT**), with dynamic formulas and step-by-step explanations.
    """)

    st.markdown("---")

    # ==========================================================
    # 1) Combinations and Permutations
    # ==========================================================
    st.subheader("1. Combinations and Permutations Calculator")

    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Enter n (total items)", min_value=0, step=1, format="%d", value=5)
    with col2:
        r = st.number_input("Enter r (items selected)", min_value=0, step=1, format="%d", value=3)

    st.markdown("### 📘 Step-by-Step Solution")
    step_box("**Step 1:** Identify \\(n\\) and \\(r\\).")
    st.latex(fr"n={n},\quad r={r}")

    if r > n:
        st.error("❌ r cannot be greater than n.")
    else:
        step_box("**Step 2:** Apply the formulas.")
        st.latex(r"""
        \binom{n}{r} \;=\; \frac{n!}{r!(n-r)!}
        \qquad\qquad
        P(n,r) \;=\; \frac{n!}{(n-r)!}
        """)

        ncr = comb(n, r)
        npr = perm(n, r)

        step_box("**Step 3:** Compute and interpret.")
        st.markdown(f"**Combinations** \\((nCr)\\): ways to choose without order → **{ncr}**")
        st.markdown(f"**Permutations** \\((nPr)\\): ways to arrange with order → **{npr}**")

    st.markdown("---")

    # ==========================================================
    # 2) Basic Probability Inputs
    # ==========================================================
    st.subheader("2. Basic Probability Inputs")
    st.write("Enter probabilities as decimals or percentages (e.g., `0.4` or `40%`).")

    colA, colB = st.columns(2)
    with colA:
        P_A_input = st.text_input("Probability of Event A,  P(A)", "0.5")
    with colB:
        P_B_input = st.text_input("Probability of Event B,  P(B)", "0.5")

    try:
        P_A = parse_probability(P_A_input)
        P_B = parse_probability(P_B_input)
        st.info(f"P(A) = **{P_A:.4f}**,   P(B) = **{P_B:.4f}**")
    except ValueError as e:
        st.error(str(e))
        return

    independent = st.checkbox("Assume A and B are independent", value=True)

    st.markdown("---")

    # ==========================================================
    # 3) Compound Events
    # ==========================================================
    st.subheader("3. Compound Events Probability")

    event = st.selectbox(
        "Select compound event:",
        ["P(A and B)", "P(A or B)", "P(not A)", "P(not B)"],
        index=0
    )

    st.markdown("### 📘 Step-by-Step Solution")

    # --------------------- P(A and B) ---------------------
    if event == "P(A and B)":
        step_box("**Step 1:** Identify whether A and B are independent.")
        if independent:
            st.latex(r"P(A \cap B) \;=\; P(A)\,P(B)")
            step_box("**Step 2:** Substitute values and compute.")
            P_and = P_A * P_B
            st.latex(fr"P(A \cap B) \;=\; {P_A:.4f}\times{P_B:.4f} \;=\; {P_and:.4f}")
            st.success(f"**Result:**  P(A and B) = {P_and:.4f}")
        else:
            st.latex(r"\text{If not independent, } P(A \cap B) \text{ must be provided.}")
            P_and_input = st.text_input("Enter joint probability P(A ∩ B)", "0.25")
            try:
                P_and = parse_probability(P_and_input)
                step_box("**Step 2:** Use the provided joint probability.")
                st.latex(fr"P(A \cap B) \;=\; {P_and:.4f}")
                st.success(f"**Result:**  P(A and B) = {P_and:.4f}")
            except Exception:
                st.warning("Enter a valid probability for P(A ∩ B) (e.g., 0.25 or 25%).")

    # --------------------- P(A or B) ---------------------
    elif event == "P(A or B)":
        step_box("**Step 1:** Use the addition rule.")
        if independent:
            st.latex(r"""
            \begin{aligned}
            P(A \cup B) &= P(A) + P(B) - P(A \cap B) \\
                        &= P(A) + P(B) - P(A)P(B)
            \end{aligned}
            """)
            step_box("**Step 2:** Substitute values and compute.")
            P_or = P_A + P_B - (P_A * P_B)
            st.latex(fr"P(A \cup B) = {P_A:.4f} + {P_B:.4f} - ({P_A:.4f}\times{P_B:.4f}) = {P_or:.4f}")
            st.success(f"**Result:**  P(A or B) = {P_or:.4f}")
        else:
            st.latex(r"P(A \cup B) \;=\; P(A)+P(B)-P(A \cap B)")
            know_joint = st.radio("How would you like to proceed?", ["I know P(A ∩ B)", "I only know P(A ∪ B)"], horizontal=True)
            if know_joint == "I know P(A ∩ B)":
                P_and_input = st.text_input("Enter joint probability P(A ∩ B)", "0.25", key="or_and")
                try:
                    P_and = parse_probability(P_and_input)
                    step_box("**Step 2:** Substitute values and compute using inclusion-exclusion.")
                    P_or = P_A + P_B - P_and
                    st.latex(fr"P(A \cup B) = {P_A:.4f} + {P_B:.4f} - {P_and:.4f} = {P_or:.4f}")
                    st.success(f"**Result:**  P(A or B) = {P_or:.4f}")
                except Exception:
                    st.warning("Enter a valid probability for P(A ∩ B).")
            else:
                P_or_input = st.text_input("Enter union probability P(A ∪ B)", "0.65", key="or_union")
                try:
                    P_or = parse_probability(P_or_input)
                    step_box("**Step 2:** Use the provided union probability.")
                    st.latex(fr"P(A \cup B) \;=\; {P_or:.4f}")
                    st.success(f"**Result:**  P(A or B) = {P_or:.4f}")
                except Exception:
                    st.warning("Enter a valid probability for P(A ∪ B).")

    # --------------------- P(not A) ---------------------
    elif event == "P(not A)":
        step_box("**Step 1:** Use the complement rule.")
        st.latex(r"P(\text{not }A) \;=\; 1 - P(A)")
        step_box("**Step 2:** Substitute values and compute.")
        P_notA = 1 - P_A
        st.latex(fr"P(\text{{not }}A) \;=\; 1 - {P_A:.4f} \;=\; {P_notA:.4f}")
        st.success(f"**Result:**  P(not A) = {P_notA:.4f}")

    # --------------------- P(not B) ---------------------
    elif event == "P(not B)":
        step_box("**Step 1:** Use the complement rule.")
        st.latex(r"P(\text{not }B) \;=\; 1 - P(B)")
        step_box("**Step 2:** Substitute values and compute.")
        P_notB = 1 - P_B
        st.latex(fr"P(\text{{not }}B) \;=\; 1 - {P_B:.4f} \;=\; {P_notB:.4f}")
        st.success(f"**Result:**  P(not B) = {P_notB:.4f}")

# ---------- Run (for standalone testing) ----------
if __name__ == "__main__":
    run()
