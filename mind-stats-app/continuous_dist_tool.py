import streamlit as st
import sample_proportion_tool.py  # this is your combined normal & sampling module

def run():
    st.header("📈 Continuous Probability Distributions")

    categories = [
        "Uniform Distribution",
        "Normal Distribution and Sampling"
    ]

    # --- Main dropdown replaces sidebar ---
    choice = st.selectbox(
        "Choose a distribution:",
        categories,
        index=None,
        placeholder="Select a distribution to begin..."
    )

    if not choice:
        st.info("👆 Please select a distribution to start exploring.")
        return

    # --- Route to the correct module ---
    if choice == "Uniform Distribution":
        uniform_tool.run()
    elif choice == "Normal Distribution and Sampling":
        sample_proportion_tool.run()

# ---------- Run App ----------
if __name__ == "__main__":
    run()
