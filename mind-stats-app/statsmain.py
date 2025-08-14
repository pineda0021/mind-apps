import streamlit as st
import descriptive_tool
import discrete_dist_tool
import binomial_tool
import poisson_tool
import probability_tool  
import continuous_dist_tool
import confidence_intervals_tool
import inferences_one_sample_tool  
import inferences_two_sample_tool  
import chi_square_tests_tool
import anova_tool  

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
    "Inferences on Two Samples",
    "Chi-Square Tests",
    "One-Way ANOVA"  # <- added ANOVA to sidebar
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
    inferences_one_sample_tool.run_hypothesis_tool() 
elif tool == "Inferences on Two Samples":   
    inferences_two_sample_tool.run_two_sample_tool()  
elif tool == "Chi-Square Tests":
    chi_square_tests_tool.run_chi_square_tool()  
elif tool == "One-Way ANOVA":  
    anova_tool.run_anova_tool()  

st.markdown("""
---
📘 Explore statistics with interactive tools built for conceptual clarity, practice, and fun.

👨‍🏫 **About the Creator:** Professor Edward Pineda-Castro  
Department of Mathematics, Los Angeles City College  
📧 Email: pinedaem@lacitycollege.edu | 📞 Tel: (323) 953-4000 ext. 2827  
Founder of **MIND** — *Making Inference Digestible*
""")



