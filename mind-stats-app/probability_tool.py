# ==========================================================
# probability_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# Built with the students in MIND
# ==========================================================

import streamlit as st
from math import comb, perm

# ---------- Helper for step display ----------
def step_box(text):
    st.markdown(
        f"""
        <div style="background-color:#f0f6ff;padding:10px;border-radius:10px;
        border-left:5px solid #007acc;margin-bottom:10px;">
        <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Parse probability (decimal or percent) ----------
def parse_probability(prob_str):
    prob_str = prob_str.strip()
    if prob_str.endswith("%"):
        prob_val = float(prob_str[:-1]) / 100
    else:
        prob_val = float(prob_str)
    if not (0 <= prob_val <= 1):
        raise ValueError("Probability must be between 0 and 1 (or 0% to 100%).")
    return prob_val


# ==========================================================
# MAIN APP
# ==========================================================
def run():
    st.header("🎲 Probability Tool")
    st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — Built with the students in MIND")

    st.write("""
    This tool computes:
    - **Combinations** \\((nCr)\\) and **Permutations** \\((nPr)\\)
    - **Compound event probabilities** (AND, OR, NOT)
    - **Conditional probabilities** and **Bayes’ Theorem**
    with step-by-step explanations and LaTeX formulas.
    """)

    st.markdown("---")

    # ==========================================================
    # 1. COMBINATIONS & PERMUTATIONS
    # ==========================================================
    st.subheader("1️⃣ Combinations and Permutations")

    n = st.number_input("Enter n (total items)", min_value=0, step=1, format="%d", value=5)
    r = st.number_input("Enter r (items selected)", min_value=0, step=1, format="%d", value=3)

    st.markdown("### 📘 Step-by-Step Solution")
    step_box("**Step 1:** Identify n and r.")
    st.latex(fr"n = {n},\quad r = {r}")

    if r > n:
        st.error("❌ r cannot be greater than n.")
    else:
        step_box("**Step 2:** Apply the formulas.")
        st.latex(r"""
        \binom{n}{r} = \frac{n!}{r!(n-r)!}, \qquad
        P(n,r) = \frac{n!}{(n-r)!}
        """)

        ncr, npr = comb(n, r), perm(n, r)
        step_box("**Step 3:** Compute and interpret.")
        st.success(f"Combinations (nCr): **{ncr}** ways (order does not matter).")
        st.success(f"Permutations (nPr): **{npr}** ways (order matters).")

    st.markdown("---")

    # ==========================================================
    # 2. BASIC PROBABILITIES
    # ==========================================================
    st.subheader("2️⃣ Basic Probability Inputs")

    st.write("Enter probabilities as decimals or percentages (e.g., `0.4` or `40%`).")

    col1, col2 = st.columns(2)
    with col1:
        P_A_input = st.text_input("Probability of Event A, P(A)", "0.5")
    with col2:
        P_B_input = st.text_input("Probability of Event B, P(B)", "0.5")

    try:
        P_A, P_B = parse_probability(P_A_input), parse_probability(P_B_input)
    except Exception as e:
        st.error(str(e))
        return

    st.info(f"P(A) = {P_A:.4f},  P(B) = {P_B:.4f}")
    independent = st.checkbox("Assume A and B are independent", value=True)

    st.markdown("---")

    # ==========================================================
    # 3. COMPOUND EVENTS
    # ==========================================================
    st.subheader("3️⃣ Compound Event Probabilities")

    event = st.selectbox("Select a compound event:",
        ["P(A and B)", "P(A or B)", "P(not A)", "P(not B)"], index=0)

    st.markdown("### 📘 Step-by-Step Solution")

    # --- P(A and B) ---
    if event == "P(A and B)":
        step_box("**Step 1:** Identify if A and B are independent.")
        if independent:
            st.latex(r"P(A \cap B) = P(A)P(B)")
            P_and = P_A * P_B
            st.latex(fr"P(A \cap B) = {P_A:.4f}\times{P_B:.4f} = {P_and:.4f}")
        else:
            st.latex(r"\text{For dependent events, } P(A \cap B) \text{ must be known.}")
            P_and_input = st.text_input("Enter P(A ∩ B)", "0.25")
            try:
                P_and = parse_probability(P_and_input)
            except Exception:
                st.error("Invalid probability for P(A ∩ B).")
                return
            st.latex(fr"P(A \cap B) = {P_and:.4f}")
        st.success(f"**Result:** P(A and B) = {P_and:.4f}")

    # --- P(A or B) ---
    elif event == "P(A or B)":
        step_box("**Step 1:** Apply the addition rule.")
        if independent:
            st.latex(r"P(A \cup B) = P(A)+P(B)-P(A)P(B)")
            P_or = P_A + P_B - (P_A * P_B)
        else:
            P_and_input = st.text_input("Enter P(A ∩ B) for dependent events", "0.25")
            try:
                P_and = parse_probability(P_and_input)
            except Exception:
                st.error("Invalid P(A ∩ B).")
                return
            st.latex(r"P(A \cup B) = P(A)+P(B)-P(A \cap B)")
            P_or = P_A + P_B - P_and
        st.latex(fr"P(A \cup B) = {P_or:.4f}")
        st.success(f"**Result:** P(A or B) = {P_or:.4f}")

    # --- P(not A) ---
    elif event == "P(not A)":
        step_box("**Step 1:** Apply complement rule.")
        st.latex(r"P(\text{not }A) = 1 - P(A)")
        P_notA = 1 - P_A
        st.latex(fr"P(\text{{not }}A) = 1 - {P_A:.4f} = {P_notA:.4f}")
        st.success(f"**Result:** P(not A) = {P_notA:.4f}")

    # --- P(not B) ---
    elif event == "P(not B)":
        step_box("**Step 1:** Apply complement rule.")
        st.latex(r"P(\text{not }B) = 1 - P(B)")
        P_notB = 1 - P_B
        st.latex(fr"P(\text{{not }}B) = 1 - {P_B:.4f} = {P_notB:.4f}")
        st.success(f"**Result:** P(not B) = {P_notB:.4f}")

    st.markdown("---")

    # ==========================================================
    # 4. CONDITIONAL PROBABILITY & BAYES’ THEOREM
    # ==========================================================
    st.subheader("4️⃣ Conditional Probability & Bayes’ Theorem")

    mode = st.selectbox(
        "Choose formula type:",
        ["P(A|B)", "P(B|A)", "Bayes' Theorem"],
        index=0
    )

    st.markdown("### 📘 Step-by-Step Solution")

    # --- Conditional P(A|B) ---
    if mode == "P(A|B)":
        step_box("**Step 1:** Recall the definition.")
        st.latex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}")
        P_and_input = st.text_input("Enter joint probability P(A ∩ B)", "0.25", key="ab")
        try:
            P_and = parse_probability(P_and_input)
            P_given_B = P_and / P_B
            st.latex(fr"P(A|B) = \frac{{{P_and:.4f}}}{{{P_B:.4f}}} = {P_given_B:.4f}")
            st.success(f"**Result:** P(A|B) = {P_given_B:.4f}")
        except Exception:
            st.warning("Invalid inputs. Check P(A ∩ B) and P(B).")

    # --- Conditional P(B|A) ---
    elif mode == "P(B|A)":
        step_box("**Step 1:** Recall the definition.")
        st.latex(r"P(B|A) = \frac{P(A \cap B)}{P(A)}")
        P_and_input = st.text_input("Enter joint probability P(A ∩ B)", "0.25", key="ba")
        try:
            P_and = parse_probability(P_and_input)
            P_given_A = P_and / P_A
            st.latex(fr"P(B|A) = \frac{{{P_and:.4f}}}{{{P_A:.4f}}} = {P_given_A:.4f}")
            st.success(f"**Result:** P(B|A) = {P_given_A:.4f}")
        except Exception:
            st.warning("Invalid inputs. Check P(A ∩ B) and P(A).")

    # --- Bayes' Theorem ---
    elif mode == "Bayes' Theorem":
        step_box("**Step 1:** Recall the theorem.")
        st.latex(r"P(A|B) = \frac{P(B|A)\,P(A)}{P(B)}")
        P_B_given_A_input = st.text_input("Enter P(B|A)", "0.7")
        try:
            P_B_given_A = parse_probability(P_B_given_A_input)
            P_given_B = (P_B_given_A * P_A) / P_B
            st.latex(fr"P(A|B) = \frac{{{P_B_given_A:.4f}\times{P_A:.4f}}}{{{P_B:.4f}}} = {P_given_B:.4f}")
            st.success(f"**Result:** P(A|B) = {P_given_B:.4f}")
        except Exception:
            st.warning("Invalid input for P(B|A), P(A), or P(B).")

# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":
    run()

