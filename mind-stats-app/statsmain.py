import streamlit as st

# Import tool modules (make sure these files exist)
import descriptive_tool
# Future modules (placeholders)
# import probability_tool
# import discrete_dist_tool
# import continuous_dist_tool
# import ci_tool
# import ht_one_sample_tool
# import ht_two_samples_tool
# import chi_square_tool
# import anova_tool
# import regression_tool

# App header
st.set_page_config(page_title="MIND: Statistics Visualizer", layout="wide")
st.title("🧠 MIND: Statistics Visualizer Suite")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

# Sidebar navigation
st.sidebar.header("📚 Select a Concept")
tool = st.sidebar.radio("Choose a tool:", [
    "Descriptive Statistics",
    # "Probability",
    # "Discrete Distributions",
    # "Continuous Distributions",
    # "Confidence Interval",
    # "Hypothesis Testing (One Sample)",
    # "Hypothesis Testing (Two Samples)",
    # "Chi-Square Test",
    # "ANOVA",
    # "Regression Analysis"
])

# Route to tools
if tool == "Descriptive Statistics":
    descriptive_tool.run()
# elif tool == "Probability":
#     probability_tool.run()
# elif tool == "Discrete Distributions":
#     discrete_dist_tool.run()
# elif tool == "Continuous Distributions":
#     continuous_dist_tool.run()
# elif tool == "Confidence Interval":
#     ci_tool.run()
# elif tool == "Hypothesis Testing (One Sample)":
#     ht_one_sample_tool.run()
# elif tool == "Hypothesis Testing (Two Samples)":
#     ht_two_samples_tool.run()
# elif tool == "Chi-Square Test":
#     chi_square_tool.run()
# elif tool == "ANOVA":
#     anova_tool.run()
# elif tool == "Regression Analysis":
#     regression_tool.run()

# Footer
st.markdown("""
---
📘 Explore statistics with interactive tools built for conceptual clarity, practice, and fun.

👨‍🏫 **About the Creator:** Professor Edward Pineda-Castro  
Department of Mathematics, Los Angeles City College  
📧 Email: pinedaem@lacitycollege.edu | 📞 Tel: (323) 953-4000 ext. 2827  
Founder of **MIND** — *Making Inference Digestible*
""")
