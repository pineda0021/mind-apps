import streamlit as st
import uniform_tool
import normal_tool
import inverse_normal_tool
import sample_mean_tool
import sample_proportion_tool

def run():
    st.sidebar.title("Choose a Continuous Distribution")
    choice = st.sidebar.radio(
        "",
        [
            "Uniform Distribution",
            "Normal Distribution",
            "Inverse Normal Distribution",
            "Distribution of the Sample Mean",
            "Distribution of the Sample Proportion"
        ]
    )

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
