# ==========================================================
# Uniform Distribution (Text-based explanation + interpretation tips)
# ==========================================================
def uniform_distribution(decimal):
    st.markdown("### 🎲 **Uniform Distribution**")
    st.latex(r"f(x) = \frac{1}{b - a}, \quad a \le x \le b")

    a = st.number_input("Lower bound (a):", value=0.0, key="ua_main")
    b = st.number_input("Upper bound (b):", value=10.0, key="ub_main")
    if b <= a:
        st.error("⚠️ Upper bound (b) must be greater than lower bound (a).")
        return

    pdf = 1 / (b - a)
    st.write(f"**Constant PDF:** f(x) = {round(pdf, decimal)} for {a} ≤ x ≤ {b}")

    calc_type = st.selectbox("Choose a calculation:", [
        "P(X < x) = P(X ≤ x)",
        "P(X = x)",
        "P(X > x) = P(X ≥ x)",
        "P(a < X < b)",
        "Inverse: Find x for given probability"
    ], key="uniform_calc_type")

    x = np.linspace(a - (b - a)*0.2, b + (b - a)*0.2, 500)
    y = np.where((x >= a) & (x <= b), pdf, 0)

    # --- Case 1: P(X < x) = P(X ≤ x)
    if calc_type == "P(X < x) = P(X ≤ x)":
        x_val = st.number_input("Enter x value:", value=(a + b) / 2, key="ux_less")
        if x_val <= a:
            prob = 0.0
        elif x_val >= b:
            prob = 1.0
        else:
            prob = (x_val - a) / (b - a)

        st.markdown(f"""
        **🧮 Step-by-step:**
        1. P(X ≤ x) = (x − a) / (b − a)  
        2. P(X ≤ {x_val}) = ({x_val} − {a}) / ({b} − {a}) = {round(prob, decimal)}  
        3. **Final Answer:** P(X ≤ {x_val}) = {round(prob, decimal)}  
        """)
        st.info("📘 Interpretation Tip: This represents the proportion of outcomes where X is less than or equal to the chosen value.")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x >= a) & (x <= x_val), color="skyblue", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # --- Case 2: P(X = x)
    elif calc_type == "P(X = x)":
        x_val = st.number_input("Enter x value:", value=(a + b) / 2, key="ux_equal")
        st.markdown(f"""
        **🧮 Step-by-step:**
        1. In a continuous distribution, P(X = x) = 0  
        2. **Final Answer:** P(X = {x_val}) = 0  
        """)
        st.info("📘 Interpretation Tip: In continuous distributions, exact values have zero probability, but intervals have measurable probabilities.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color="blue")
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # --- Case 3: P(X > x) = P(X ≥ x)
    elif calc_type == "P(X > x) = P(X ≥ x)":
        x_val = st.number_input("Enter x value:", value=(a + b) / 2, key="ux_greater")
        if x_val <= a:
            prob = 1.0
        elif x_val >= b:
            prob = 0.0
        else:
            prob = (b - x_val) / (b - a)

        st.markdown(f"""
        **🧮 Step-by-step:**
        1. P(X ≥ x) = (b − x) / (b − a)  
        2. P(X ≥ {x_val}) = ({b} − {x_val}) / ({b} − {a}) = {round(prob, decimal)}  
        3. **Final Answer:** P(X ≥ {x_val}) = {round(prob, decimal)}  
        """)
        st.info("📘 Interpretation Tip: This represents the proportion of outcomes where X is greater than or equal to the chosen value.")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x >= x_val) & (x <= b), color="lightgreen", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)

    # --- Case 4: P(a < X < b)
    elif calc_type == "P(a < X < b)":
        low = st.number_input("Lower bound (a):", value=a, key="ua_inner")
        high = st.number_input("Upper bound (b):", value=b, key="ub_inner")
        if high <= low:
            st.error("⚠️ Upper bound must be greater than lower bound.")
            return

        if high < a or low > b:
            prob = 0.0
        else:
            lower = max(low, a)
            upper = min(high, b)
            prob = (upper - lower) / (b - a)

        st.markdown(f"""
        **🧮 Step-by-step:**
        1. P(a < X < b) = (b − a) / (B − A)  
        2. P({low} < X < {high}) = ({high} − {low}) / ({b} − {a}) = {round(prob, decimal)}  
        3. **Final Answer:** P({low} < X < {high}) = {round(prob, decimal)}  
        """)
        st.info("📘 Interpretation Tip: This gives the probability that X lies between two specific values within the uniform range.")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x >= low) & (x <= high), color="orange", alpha=0.6)
        ax.axvline(low, color="red", linestyle="--")
        ax.axvline(high, color="red", linestyle="--")
        st.pyplot(fig)

    # --- Case 5: Inverse: Find x for given probability
    elif calc_type == "Inverse: Find x for given probability":
        p = st.number_input("Enter probability p for P(X ≤ x) = p (0 < p < 1):",
                            min_value=0.0, max_value=1.0, value=0.5, key="u_inverse_p")
        x_val = a + p * (b - a)

        st.markdown(f"""
        **🧮 Step-by-step:**
        1. P(X ≤ x) = (x − a) / (b − a)  
        2. Solve for x → x = a + p(b − a)  
        3. x = {a} + {p}({b} − {a}) = {round(x_val, decimal)}  
        4. **Final Answer:** x = {round(x_val, decimal)} for P(X ≤ x) = {p}  
        """)
        st.info("📘 Interpretation Tip: This finds the cutoff x below which a given proportion (p) of values in the uniform distribution fall.")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color="blue")
        ax.fill_between(x, 0, y, where=(x >= a) & (x <= x_val), color="skyblue", alpha=0.6)
        ax.axvline(x_val, color="red", linestyle="--")
        st.pyplot(fig)
