import streamlit as st

import descriptive_tool
import probability_tool
import discrete_dist_tool
# Other modules commented out for now, or have placeholders

st.set_page_config(page_title="MIND: Statistics Visualizer", layout="wide")
st.title("🧠 MIND: Statistics Visualizer Suite")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

tool = st.sidebar.radio("Choose a tool:", [
    "Descriptive Statistics",
    "Probability",
    "Discrete Distributions",
    # "Continuous Distributions",
])

if tool == "Descriptive Statistics":
    descriptive_tool.run()
elif tool == "Probability":
    probability_tool.run()
elif tool == "Discrete Distributions":
    discrete_dist_tool.run()
# elif tool == "Continuous Distributions":
#     continuous_dist_tool.run()

st.markdown("""
---
📘 Explore statistics with interactive tools built for conceptual clarity, practice, and fun.

👨‍🏫 **About the Creator:** Professor Edward Pineda-Castro  
Department of Mathematics, Los Angeles City College  
📧 Email: pinedaem@lacitycollege.edu | 📞 Tel: (323) 953-4000 ext. 2827  
Founder of **MIND** — *Making Inference Digestible*
""")


