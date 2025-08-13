import streamlit as st
import uniform_tool
import normal_tool
import inverse_normal_tool

def run():
    st.header("📈 Continuous Distributions")

    choice = st.radio("Choose Distribution:", [
        "Uniform Distribution",
        "Normal Distribution",
        "Inverse Normal Distribution"
    ])

    if choice == "Uniform Distribution":
        uniform_tool.run()
    elif choice == "Normal Distribution":
        normal_tool.run()
    elif choice == "Inverse Normal Distribution":
        inverse_normal_tool.run()
