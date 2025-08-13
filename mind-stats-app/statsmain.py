import streamlit as st
import descriptive_tool
import discrete_dist_tool
import binomial_tool
import poisson_tool
import probability_tool  
import continuous_dist_tool
import confidence_intervals_tool
import inferences_one_sample_tool

st.set_page_config(page_title="MIND: Statistics Visualizer", layout="wide")
st.title("🧠 MIND: Statistics Visualizer Suite")

st.sidebar.header("📚 Select a Concept")
tool = st.sidebar.radio("Choose a tool:", [
    "Descriptive Statistics",
    "Probability",              
    "Discrete Distributions",
    "Continuous Distributions",
    "Confidence Intervals",
    "Inferences on One Sample",
])

if tool == "Descriptive Statistics":
    descriptive_tool.run()
elif tool == "Probability":     
    probability_tool.run()
elif tool == "Discrete Distributions":
    discrete_dist_tool.run()
elif tool == "Continuous Distributions":
     continuous_dist_tool.run()
elif tool == "Confidence Intervals":   
     confidence_intervals_tool.run()
elif tool == "Inferences on One Sample":
     inferences_one_sample_tool.run()

st.markdown("""
---
📘 Explore statistics with interactive tools built for conceptual clarity, practice, and fun.

👨‍🏫 **About the Creator:** Professor Edward Pineda-Castro  
Department of Mathematics, Los Angeles City College  
📧 Email: pinedaem@lacitycollege.edu | 📞 Tel: (323) 953-4000 ext. 2827  
Founder of **MIND** — *Making Inference Digestible*
""")
