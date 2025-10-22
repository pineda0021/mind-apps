import streamlit as st
import uniform_tool
import normal_tool
import inverse_normal_tool
import sample_mean_tool
import sample_proportion_tool

def run():
    st.header("📈 Continuous Probability Distributions")

    categories = [
        "Uniform Distribution",
        "Normal Distribution",
        "Inverse Normal Distribution",
        "Distribution of the Sample Mean",
        "Distribution of the Sample Proportion"
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
    elif choice == "Normal Distribution":
        normal_tool.run()
    elif choice == "Inverse Normal Distribution":
        inverse_normal_tool.run()
    elif choice == "Distribution of the Sample Mean":
        sample_mean_tool.run()
    elif choice == "Distribution of the Sample Proportion":
        sample_proportion_tool.run()

# ---------- Run App ----------
if __name__ == "__main__":
    run()
