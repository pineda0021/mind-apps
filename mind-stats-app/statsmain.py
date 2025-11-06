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
import regression_analysis_tool  
import ti84  # ✅ TI-84 embedded calculator
import RStudio  # 👈 RStudio (Colab) integration

# ---------- App Configuration ----------
st.set_page_config(page_title="MIND: Statistics Visualizer", layout="wide")
st.title("🧠 MIND: Statistics Visualizer Suite")


# ---------- Sidebar Navigation ----------
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
    "One-Way ANOVA",
    "Simple Regression",
    "Multiple Regression",
    "TI-84 Calculator",   # ✅ Added comma here
    "RStudio"             # 👈 Separate option for RStudio
])


# ---------- Tool Routing ----------
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

elif tool == "Simple Regression":
    regression_analysis_tool.run_simple_regression_tool()

elif tool == "Multiple Regression":
    regression_analysis_tool.run_multiple_regression_tool()

elif tool == "TI-84 Calculator":  
    ti84.run()  # ✅ Launches embedded TI-84 calculator

elif tool == "RStudio":
    RStudio.run()  # 👈 Activates your new RStudio module


# ---------- Footer ----------
st.markdown("""
---
📘 Explore statistics with interactive tools built for conceptual clarity, practice, and fun.

👨‍🏫 **About the Creator:** Professor Edward Pineda-Castro  
Department of Mathematics, Los Angeles City College  
📧 Email: pinedaem@lacitycollege.edu | 📞 Tel: (323) 953-4000 ext. 2827  
Founder of **MIND** — *Making Inference Digestible*
""")
