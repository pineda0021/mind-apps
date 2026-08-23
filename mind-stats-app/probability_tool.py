# ==========================================================
# probability_tool.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
from math import comb, perm


# ==========================================================
# Helper for step display
# ==========================================================
def step_box(text):
    st.info(text)


# ==========================================================
# Parse probability (decimal or percent)
# ==========================================================
def parse_probability(prob_str):
    prob_str = prob_str.strip()

    if prob_str.endswith("%"):
        prob_val = float(prob_str[:-1]) / 100
    else:
        prob_val = float(prob_str)

    if not (0 <= prob_val <= 1):
        raise ValueError(
            "Probability must be between 0 and 1 (or 0% to 100%)."
        )

    return prob_val


# ==========================================================
# MAIN APP
# ==========================================================
def run():

    st.header("🎲 Probability Tool")

    st.markdown(
        """
        This tool computes:

        - **Combinations** (nCr) and **Permutations** (nPr)
        - **Compound event probabilities** (AND, OR, NOT)
        - **Conditional probabilities**
        - **Bayes’ Theorem**

        with step-by-step explanations and LaTeX formulas.
        """
    )

    st.markdown("---")


    # ==========================================================
    # 1. COMBINATIONS & PERMUTATIONS
    # ==========================================================
    st.subheader("1️⃣ Combinations and Permutations")

    n = st.number_input(
        "Enter n (total items)",
        min_value=0,
        step=1,
        format="%d",
        value=5
    )

    r = st.number_input(
        "Enter r (items selected)",
        min_value=0,
        step=1,
        format="%d",
        value=3
    )


    # ==========================================================
    # CALCULATE BUTTON
    # ==========================================================
    if st.button(
        "👨‍💻 Calculate",
        key="calculate_combinations"
    ):
        st.session_state["show_combinations"] = True


    if st.session_state.get("show_combinations", False):

        st.markdown("#### 📘 Step-by-Step Solution")

        step_box("Step 1: Identify n and r.")

        st.latex(
            fr"n = {n},\quad r = {r}"
        )


        if r > n:

            st.error(
                "❌ r cannot be greater than n."
            )

        else:

            step_box(
                "Step 2: Apply the formulas."
            )

            st.latex(
                r"""
                \binom{n}{r}
                =
                \frac{n!}{r!(n-r)!},
                \qquad
                P(n,r)
                =
                \frac{n!}{(n-r)!}
                """
            )


            ncr = comb(n, r)
            npr = perm(n, r)


            step_box(
                "Step 3: Compute and interpret."
            )


            st.success(
                f"Combinations (nCr): {ncr} ways "
                "(order does not matter)."
            )


            st.success(
                f"Permutations (nPr): {npr} ways "
                "(order matters)."
            )


    st.markdown("---")


    # ==========================================================
    # 2. BASIC PROBABILITIES
    # ==========================================================
    st.subheader("2️⃣ Basic Probability Inputs")

    st.markdown(
        """
        Enter probabilities as decimals or percentages  
        (e.g., **0.4** or **40%**).
        """
    )


    col1, col2 = st.columns(2)


    with col1:

        P_A_input = st.text_input(
            "Probability of Event A, P(A)",
            "0.5"
        )


    with col2:

        P_B_input = st.text_input(
            "Probability of Event B, P(B)",
            "0.5"
        )


    try:

        P_A = parse_probability(
            P_A_input
        )

        P_B = parse_probability(
            P_B_input
        )


    except Exception as e:

        st.error(
            str(e)
        )

        return


    independent = st.checkbox(
        "Assume A and B are independent",
        value=True
    )


    st.markdown("---")


    # ==========================================================
    # 3. COMPOUND EVENTS
    # ==========================================================
    st.subheader(
        "3️⃣ Compound Event Probabilities"
    )


    event = st.selectbox(

        "Select a compound event:",

        [
            "P(A and B)",
            "P(A or B)",
            "P(not A)",
            "P(not B)"
        ]

    )


    # ==========================================================
    # Additional input if needed
    # ==========================================================

    P_and_input = None


    if (
        not independent
        and event in [
            "P(A and B)",
            "P(A or B)"
        ]
    ):

        P_and_input = st.text_input(
            "Enter P(A ∩ B)",
            "0.25",
            key="compound_intersection"
        )


    # ==========================================================
    # CALCULATE BUTTON
    # ==========================================================

    if st.button(
        "👨‍💻 Calculate",
        key="calculate_compound"
    ):

        st.session_state[
            "show_compound"
        ] = True


    if st.session_state.get(
        "show_compound",
        False
    ):

        st.markdown(
            "#### 📘 Step-by-Step Solution"
        )


        st.info(
            f"P(A) = {P_A:.4f}, "
            f"P(B) = {P_B:.4f}"
        )


        # ------------------------------------------------------
        # P(A and B)
        # ------------------------------------------------------

        if event == "P(A and B)":

            step_box(
                "Step 1: Identify whether "
                "A and B are independent."
            )


            if independent:

                st.latex(
                    r"P(A \cap B)=P(A)P(B)"
                )


                P_and = P_A * P_B


                st.latex(

                    fr"""
                    P(A \cap B)
                    =
                    {P_A:.4f}
                    \times
                    {P_B:.4f}
                    =
                    {P_and:.4f}
                    """

                )


            else:

                try:

                    P_and = parse_probability(
                        P_and_input
                    )


                except Exception:

                    st.error(
                        "Invalid probability "
                        "for P(A ∩ B)."
                    )

                    return


                st.latex(

                    fr"""
                    P(A \cap B)
                    =
                    {P_and:.4f}
                    """

                )


            st.success(

                f"Result: P(A and B) = "
                f"{P_and:.4f}"

            )


        # ------------------------------------------------------
        # P(A or B)
        # ------------------------------------------------------

        elif event == "P(A or B)":

            step_box(
                "Step 1: Apply the addition rule."
            )


            if independent:

                st.latex(

                    r"""
                    P(A \cup B)
                    =
                    P(A)+P(B)-P(A)P(B)
                    """

                )


                P_or = (
                    P_A
                    + P_B
                    - (P_A * P_B)
                )


                st.latex(

                    fr"""
                    P(A \cup B)
                    =
                    {P_A:.4f}
                    +
                    {P_B:.4f}
                    -
                    ({P_A:.4f})
                    ({P_B:.4f})
                    =
                    {P_or:.4f}
                    """

                )


            else:

                try:

                    P_and = parse_probability(
                        P_and_input
                    )


                except Exception:

                    st.error(
                        "Invalid P(A ∩ B)."
                    )

                    return


                st.latex(

                    r"""
                    P(A \cup B)
                    =
                    P(A)+P(B)-P(A \cap B)
                    """

                )


                P_or = (
                    P_A
                    + P_B
                    - P_and
                )


                st.latex(

                    fr"""
                    P(A \cup B)
                    =
                    {P_A:.4f}
                    +
                    {P_B:.4f}
                    -
                    {P_and:.4f}
                    =
                    {P_or:.4f}
                    """

                )


            st.success(

                f"Result: P(A or B) = "
                f"{P_or:.4f}"

            )


        # ------------------------------------------------------
        # P(not A)
        # ------------------------------------------------------

        elif event == "P(not A)":

            step_box(
                "Step 1: Apply complement rule."
            )


            st.latex(

                r"""
                P(\text{not }A)
                =
                1-P(A)
                """

            )


            P_notA = 1 - P_A


            st.latex(

                fr"""
                P(\text{{not }}A)
                =
                1-{P_A:.4f}
                =
                {P_notA:.4f}
                """

            )


            st.success(

                f"Result: P(not A) = "
                f"{P_notA:.4f}"

            )


        # ------------------------------------------------------
        # P(not B)
        # ------------------------------------------------------

        elif event == "P(not B)":

            step_box(
                "Step 1: Apply complement rule."
            )


            st.latex(

                r"""
                P(\text{not }B)
                =
                1-P(B)
                """

            )


            P_notB = 1 - P_B


            st.latex(

                fr"""
                P(\text{{not }}B)
                =
                1-{P_B:.4f}
                =
                {P_notB:.4f}
                """

            )


            st.success(

                f"Result: P(not B) = "
                f"{P_notB:.4f}"

            )


    st.markdown("---")


    # ==========================================================
    # 4. CONDITIONAL PROBABILITY & BAYES’ THEOREM
    # ==========================================================

    st.subheader(
        "4️⃣ Conditional Probability & Bayes’ Theorem"
    )


    mode = st.selectbox(

        "Choose formula type:",

        [
            "P(A|B)",
            "P(B|A)",
            "Bayes' Theorem"
        ]

    )


    # ==========================================================
    # Inputs
    # ==========================================================

    if mode == "P(A|B)":

        conditional_input = st.text_input(

            "Enter P(A ∩ B)",

            "0.25",

            key="ab"

        )


    elif mode == "P(B|A)":

        conditional_input = st.text_input(

            "Enter P(A ∩ B)",

            "0.25",

            key="ba"

        )


    else:

        conditional_input = st.text_input(

            "Enter P(B|A)",

            "0.7",

            key="bayes_input"

        )


    # ==========================================================
    # CALCULATE BUTTON
    # ==========================================================

    if st.button(

        "👨‍💻 Calculate",

        key="calculate_conditional"

    ):

        st.session_state[
            "show_conditional"
        ] = True


    if st.session_state.get(
        "show_conditional",
        False
    ):

        st.markdown(
            "#### 📘 Step-by-Step Solution"
        )


        # ------------------------------------------------------
        # Conditional P(A|B)
        # ------------------------------------------------------

        if mode == "P(A|B)":

            step_box(
                "Step 1: Recall the definition."
            )


            st.latex(

                r"""
                P(A|B)
                =
                \frac{P(A \cap B)}{P(B)}
                """

            )


            try:

                P_and = parse_probability(
                    conditional_input
                )


                if P_B == 0:

                    st.error(
                        "P(B) cannot be 0 "
                        "when calculating P(A|B)."
                    )

                    return


                P_given_B = (
                    P_and / P_B
                )


                st.latex(

                    fr"""
                    P(A|B)
                    =
                    \frac{{{P_and:.4f}}}
                    {{{P_B:.4f}}}
                    =
                    {P_given_B:.4f}
                    """

                )


                st.success(

                    f"Result: P(A|B) = "
                    f"{P_given_B:.4f}"

                )


            except Exception:

                st.warning(

                    "Invalid inputs. "
                    "Check P(A ∩ B) and P(B)."

                )


        # ------------------------------------------------------
        # Conditional P(B|A)
        # ------------------------------------------------------

        elif mode == "P(B|A)":

            step_box(
                "Step 1: Recall the definition."
            )


            st.latex(

                r"""
                P(B|A)
                =
                \frac{P(A \cap B)}{P(A)}
                """

            )


            try:

                P_and = parse_probability(
                    conditional_input
                )


                if P_A == 0:

                    st.error(
                        "P(A) cannot be 0 "
                        "when calculating P(B|A)."
                    )

                    return


                P_given_A = (
                    P_and / P_A
                )


                st.latex(

                    fr"""
                    P(B|A)
                    =
                    \frac{{{P_and:.4f}}}
                    {{{P_A:.4f}}}
                    =
                    {P_given_A:.4f}
                    """

                )


                st.success(

                    f"Result: P(B|A) = "
                    f"{P_given_A:.4f}"

                )


            except Exception:

                st.warning(

                    "Invalid inputs. "
                    "Check P(A ∩ B) and P(A)."

                )


        # ------------------------------------------------------
        # Bayes' Theorem
        # ------------------------------------------------------

        elif mode == "Bayes' Theorem":

            step_box(
                "Step 1: Recall the theorem."
            )


            st.latex(

                r"""
                P(A|B)
                =
                \frac{P(B|A)P(A)}{P(B)}
                """

            )


            try:

                P_B_given_A = parse_probability(
                    conditional_input
                )


                if P_B == 0:

                    st.error(
                        "P(B) cannot be 0 "
                        "when using Bayes’ Theorem."
                    )

                    return


                P_given_B = (
                    P_B_given_A
                    * P_A
                ) / P_B


                st.latex(

                    fr"""
                    P(A|B)
                    =
                    \frac{{
                        {P_B_given_A:.4f}
                        \times
                        {P_A:.4f}
                    }}{
                        {P_B:.4f}
                    }
                    =
                    {P_given_B:.4f}
                    """

                )


                st.success(

                    f"Result: P(A|B) = "
                    f"{P_given_B:.4f}"

                )


            except Exception:

                st.warning(
                    "Invalid inputs for "
                    "Bayes’ Theorem."
                )


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    run()
