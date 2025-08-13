from tools import uniform_tool
from tools import normal_tool
from tools import inverse_normal_tool
import streamlit as st

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
