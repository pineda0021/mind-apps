import streamlit as st

# App header configuration
st.set_page_config(page_title="MIND: Statistics Visualizer", layout="wide")

st.title("🧠 MIND: Statistics Visualizer Suite")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

# Try importing tools, but handle if they are missing
try:
    import descriptive_tool
except ImportError:
    descriptive_tool = None

try:
    import probability_tool
except ImportError:
    probability_tool = None

try:
    import discrete_dist_tool
except ImportError:
    discrete_dist_tool = None

# Sidebar navigation
st.sidebar.header("📚 Select a Concept")
tool = st.sidebar.radio("Choose a tool:", [
    "Descriptive Statistics",
    "Probability",
    "Discrete Distributions",
    # "Continuous Distributions",
    # "Confidence Interval",
    # "Hypothesis Testing (One Sample)",
    # "Hypothesis Testing (Two Samples)",
    # "Chi-Square Test",
    # "ANOVA",
    # "Regression Analysis"
])

# Tool routing with safe execution
if tool == "Descriptive Statistics":
    if descriptive_tool:
        descriptive_tool.run()
    else:
        st.error("❌ Descriptive Statistics module is missing.")

elif tool == "Probability":
    if probability_tool:
        probability_tool.run()
    else:
        st.error("❌ Probability module is missing.")

elif tool == "Discrete Distributions":
    if discrete_dist_tool:
        discrete_dist_tool.run()
    else:
        st.error("❌ Discrete Distributions module is missing.")

# Footer
st.markdown("""
---
📘 Explore statistics with interactive tools built for conceptual clarity, practice, and fun.

👨‍🏫 **About the Creator:** Professor Edward Pineda-Castro  
Department of Mathematics, Los Angeles City College  
📧 Email: pinedaem@lacitycollege.edu | 📞 Tel: (323) 953-4000 ext. 2827  
Founder of **MIND** — *Making Inference Digestible*
""")
