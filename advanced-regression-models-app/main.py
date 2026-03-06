import streamlit as st
import glm_tool
import boxcox_tool
import gamma_glm_tool

st.set_page_config(layout="wide")

st.sidebar.title("Select a Concept")

tool = st.sidebar.radio(
    "Choose a tool:",
    [
        "Gaussian Linear Model (OLS)",
        "Box-Cox Transformation",
        "Gamma GLM"
    ]
)

st.title(tool)

if tool == "Gaussian Linear Model (OLS)":
    glm_tool.run()

elif tool == "Box-Cox Transformation":
    boxcox_tool.run()

elif tool == "Gamma GLM":
    gamma_glm_tool.run()
    
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
