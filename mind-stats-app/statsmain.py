# 🔹 Always import Streamlit first
import streamlit as st

# 🔹 Try importing available tools, skip if missing
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

# Future tools (commented out until implemented)
# import continuous_dist_tool
# import ci_tool
# import ht_one_sample_tool
# import ht_two_samples_tool
# import chi_square_tool
# import anova_tool
# import regression_tool

# 🔹 Page setup
st.set_page_config(page_title="MIND: Statistics Visualizer", layout="wide")

# 🔹 App title & description
st.title("🧠 MIND: Statistics Visualizer Suite")
st.caption(
    "Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND."
)

# 🔹 Sidebar navigation
st.sidebar.header("📚 Select a Concept")
tool = st.sidebar.radio(
    "Choose a tool:",
    [
        "Descriptive Statistics",
        "Probability",
        "Discrete Distributions",
        # "Continuous Distributions",
        # "Confidence Interval",
        # "Hypothesis Testing (One Sample)",
        # "Hypothesis Testing (Two Samples)",
        # "Chi-Square Test",
        # "ANOVA",
        # "Regression Analysis",
    ],
)

# 🔹 Route to selected tool
if tool == "Descriptive Statistics":
    if descriptive_tool:
        descriptive_tool.run()
    else:
        st.error("❌ Descriptive Statistics module not found.")

elif tool == "Probability":
    if probability_tool:
        probability_tool.run()
    else:
        st.error("❌ Probability module not found.")

elif tool == "Discrete Distributions":
    if discrete_dist_tool:
        discrete_dist_tool.run()
    else:
        st.error("❌ Discrete Distributions module not found.")

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

# 🔹 Footer
st.markdown(
    """
---
📘 Explore statistics with interactive tools built for conceptual clarity, practice, and fun.

👨‍🏫 **About the Creator:** Professor Edward Pineda-Castro  
Department of Mathematics, Los Angeles City College  
📧 Email: pinedaem@lacitycollege.edu | 📞 Tel: (323) 953-4000 ext. 2827  
Founder of **MIND** — *Making Inference Digestible*
"""
)
