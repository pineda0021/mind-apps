import streamlit as st
import descriptive_tool
import discrete_dist_tool
import binomial_tool
import poisson_tool
import probability_tool  
import uniform_tool
import normal_tool
import inverse_normal_tool

st.set_page_config(page_title="MIND: Statistics Visualizer", layout="wide")
st.title("🧠 MIND: Statistics Visualizer Suite")

st.sidebar.header("📚 Select a Concept")
tool = st.sidebar.radio("Choose a tool:", [
    "Descriptive Statistics",
    "Probability",              
    "Discrete Distributions",
])

if tool == "Descriptive Statistics":
    descriptive_tool.run()
elif tool == "Probability":      # <-- route to probability_tool
    probability_tool.run()
elif tool == "Discrete Distributions":
    discrete_dist_tool.run()

st.sidebar.header("📊 Continuous Distributions")
cont_dist = st.sidebar.radio("Choose Distribution:", [
    "Uniform Distribution",
    "Normal Distribution",
    "Inverse Normal Distribution",
])

if cont_dist == "Uniform Distribution":
    uniform_tool.run()
elif cont_dist == "Normal Distribution":
    normal_tool.run()
elif cont_dist == "Inverse Normal Distribution":
    inverse_normal_tool.run()

st.markdown("""
---
📘 Explore statistics with interactive tools built for conceptual clarity, practice, and fun.

👨‍🏫 **About the Creator:** Professor Edward Pineda-Castro  
Department of Mathematics, Los Angeles City College  
📧 Email: pinedaem@lacitycollege.edu | 📞 Tel: (323) 953-4000 ext. 2827  
Founder of **MIND** — *Making Inference Digestible*
""")
