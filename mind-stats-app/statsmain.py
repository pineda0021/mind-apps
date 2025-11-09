# ====================================================
# 🧠 MIND: Statistics Visualizer Suite
# Professor Edward Pineda-Castro, Los Angeles City College
# ====================================================

import streamlit as st

# ---------- Import All Modules ----------
import descriptive_tool
import probability_tool
import discrete_dist_tool
import continuous_dist_tool
import binomial_tool
import poisson_tool
import confidence_intervals_tool
import inferences_one_sample_tool
import inferences_two_sample_tool
import chi_square_tests_tool
import anova_tool
import regression_analysis_tool
import multiple_regression_advanced_tool  # ✅ Advanced version
import ti84                # ✅ TI-84 embedded calculator
import RStudio              # ✅ RStudio module
import Python               # ✅ New Python module

# ---------- App Configuration ----------
st.set_page_config(page_title="MIND: Statistics Visualizer", layout="wide")
st.title("🧠 MIND: Statistics Visualizer Suite")

# ---------- Sidebar Navigation ----------
st.sidebar.header("📚 Select a Concept")
tool = st.sidebar.radio(
    "Choose a tool:",
    [
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
        "Multiple Regression (Advanced)",  # ✅ Only advanced version
        "TI-84 Calculator",
        "RStudio",
        "Python"  # 👈 Added Python launcher
    ]
)

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

elif tool == "Multiple Regression (Advanced)":
    multiple_regression_advanced_tool.run()

elif tool == "TI-84 Calculator":
    ti84.run()

elif tool == "RStudio":
    RStudio.run()

elif tool == "Python":
    Python.run()

# ---------- Footer ----------
st.markdown("""
---
📘 **Explore statistics interactively** — built for conceptual clarity, practice, and fun.

👨‍🏫 **About the Creator:**  
Professor Edward Pineda-Castro  
Department of Mathematics, Los Angeles City College  

📧 [pinedaem@lacitycollege.edu](mailto:pinedaem@lacitycollege.edu)  
📞 (323) 953-4000 ext. 2827  

*Founder of* **MIND** — *Making Inference Digestible*
""")
