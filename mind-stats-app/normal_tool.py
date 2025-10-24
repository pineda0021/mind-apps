import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def run():
    st.header("📈 Normal Distribution Calculator")

    st.markdown("""
    Explore the properties of the **Normal Distribution**.  
    You can calculate probabilities, critical values, and visualize the distribution.
    """)

    # --- Dropdown for Calculation Type ---
    calc_type = st.selectbox(
        "Choose a calculation type:",
        [
            "Find probability: P(X < x)",
            "Find probability: P(X > x)",
            "Find probability: P(a < X < b)",
            "Find value (given probability): inverse normal"
        ],
        index=None,
        placeholder="Select a calculation type..."
    )

    if not calc_type:
        st.info("👆 Please select a calculation type to begin.")
        return

    # --- Parameters Input ---
    st.markdown("### 🧮 Enter Distribution Parameters")
    mean = st.number_input("Mean (μ):", value=0.0)
    sd = st.number_input("Standard Deviation (σ):", min_value=0.0001, value=1.0)

    if calc_type in ["Find probability: P(X < x)", "Find probability: P(X > x)"]:
        x = st.number_input("Enter x value:", value=0.0)

        if st.button("👨‍💻 Calculate"):
            z = (x - mean) / sd
            if calc_type == "Find probability: P(X < x)":
                prob = norm.cdf(x, mean, sd)
                st.success(f"P(X < {x}) = {prob:.5f}")
            else:
                prob = 1 - norm.cdf(x, mean, sd)
                st.success(f"P(X > {x}) = {prob:.5f}")

            st.markdown(f"**Z-score:** {z:.5f}")
            plot_normal(mean, sd, x_val=x, calc_type=calc_type)

    elif calc_type == "Find probability: P(a < X < b)":
        a = st.number_input("Lower bound (a):", value=-1.0)
        b = st.number_input("Upper bound (b):", value=1.0)

        if st.button("👨‍💻 Calculate"):
            prob = norm.cdf(b, mean, sd) - norm.cdf(a, mean, sd)
            z_a = (a - mean) / sd
            z_b = (b - mean) / sd
            st.success(f"P({a} < X < {b}) = {prob:.5f}")
            st.markdown(f"**Z-scores:** zₐ = {z_a:.5f}, z_b = {z_b:.5f}")
            plot_normal(mean, sd, a_val=a, b_val=b, calc_type=calc_type)

    elif calc_type == "Find value (given probability): inverse normal":
        tail = st.selectbox("Select tail:", ["Left tail", "Right tail", "Middle area"])
        prob = st.number_input("Enter probability:", min_value=0.0, max_value=1.0, value=0.95)

        if st.button("👨‍💻 Calculate"):
            if tail == "Left tail":
                x_val = norm.ppf(prob, mean, sd)
                st.success(f"x = {x_val:.5f} for P(X < x) = {prob}")
                plot_normal(mean, sd, x_val=x_val, calc_type="Find probability: P(X < x)")
            elif tail == "Right tail":
                x_val = norm.ppf(1 - prob, mean, sd)
                st.success(f"x = {x_val:.5f} for P(X > x) = {prob}")
                plot_normal(mean, sd, x_val=x_val, calc_type="Find probability: P(X > x)")
            else:
                st.info("For middle area, enter central probability (e.g., 0.95)")
                tail_prob = (1 - prob) / 2
                lower = norm.ppf(tail_prob, mean, sd)
                upper = norm.ppf(1 - tail_prob, mean, sd)
                st.success(f"{prob*100:.1f}% of data lies between {lower:.5f} and {upper:.5f}")
                plot_normal(mean, sd, a_val=lower, b_val=upper, calc_type="Find probability: P(a < X < b)")


# --- Helper Function for Plot ---
def plot_normal(mean, sd, a_val=None, b_val=None, x_val=None, calc_type=None):
    x = np.linspace(mean - 4*sd, mean + 4*sd, 400)
    y = norm.pdf(x, mean, sd)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, 'b', lw=2)
    ax.set_title(f"Normal Distribution (μ={mean}, σ={sd})")
    ax.set_xlabel("X")
    ax.set_ylabel("Density")

    # Highlight area
    if calc_type == "Find probability: P(X < x)" and x_val is not None:
        ax.fill_between(x, y, 0, where=(x <= x_val), color='skyblue', alpha=0.6)
        ax.axvline(x_val, color='red', linestyle='--')
    elif calc_type == "Find probability: P(X > x)" and x_val is not None:
        ax.fill_between(x, y, 0, where=(x >= x_val), color='lightgreen', alpha=0.6)
        ax.axvline(x_val, color='red', linestyle='--')
    elif calc_type == "Find probability: P(a < X < b)" and a_val is not None and b_val is not None:
        ax.fill_between(x, y, 0, where=(x >= a_val) & (x <= b_val), color='orange', alpha=0.6)
        ax.axvline(a_val, color='red', linestyle='--')
        ax.axvline(b_val, color='red', linestyle='--')

    st.pyplot(fig)


if __name__ == "__main__":
    run()
