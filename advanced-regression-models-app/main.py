import streamlit as st
import glm_tool
import boxcox_tool
import gamma_glm_tool
import logistic_regression_tool
import probit_regression_tool
import cloglog_regression_tool
import cumulative_logit_tool
import cumulative_probit_tool
import cumulative_cloglog_tool
import generalized_logit_tool

st.set_page_config(layout="wide")

st.sidebar.title("Select a Concept")

tool = st.sidebar.radio(
    "Choose a tool:",
    [
        "General Linear Regression Model",
        "Box-Cox Transformation",
        "Gamma GLM",
        "Binary Logistic Regression Model",
        "Probit Regression Model",
        "Complementary Log-Log Model",
        "Cumulative Logit Model",
        "Cumulative Probit Model",
        "Cumulative Complementary Log-Log Model",
        "Generalized Logit Model for Nominal Response"
    ]
)

st.title(tool)

if tool == "General Linear Regression Model":
    glm_tool.run()

elif tool == "Box-Cox Transformation":
    boxcox_tool.run()

elif tool == "Gamma GLM":
    gamma_glm_tool.run()

elif tool == "Binary Logistic Regression Model":
    logistic_regression_tool.run()

elif tool == "Probit Regression Model":
    probit_regression_tool.run()

elif tool == "Complementary Log-Log Model":
    cloglog_regression_tool.run()

elif tool == "Cumulative Logit Model":
    cumulative_logit_tool.run()

elif tool == "Cumulative Probit Model":
    cumulative_probit_tool.run()

elif tool == "Cumulative Complementary Log-Log Model":
    cumulative_cloglog_tool.run()

elif tool == "Generalized Logit Model for Nominal Response":
    generalized_logit_tool.run()


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
