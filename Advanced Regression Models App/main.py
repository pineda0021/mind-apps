import glm_tool
import boxcox_tool
import gamma_glm_tool

tool = st.sidebar.radio(
    "Choose a tool:",
    [
        "Descriptive Statistics",
        ...
        "Multiple Regression (Advanced)",
        "Gaussian Linear Model (OLS)",
        "Box-Cox Transformation",
        "Gamma GLM"
    ]
)

elif tool == "Gaussian Linear Model (OLS)":
    glm_tool.run()

elif tool == "Box-Cox Transformation":
    boxcox_tool.run()

elif tool == "Gamma GLM":
    gamma_glm_tool.run()
