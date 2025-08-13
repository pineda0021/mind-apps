import streamlit as st
import importlib

st.set_page_config(page_title="MIND: Statistics Visualizer", layout="wide")
st.title("🧠 MIND: Statistics Visualizer Suite")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

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

def load_and_run(tool_module_name):
    try:
        module = importlib.import_module(tool_module_name)
        if hasattr(module, "run"):
            module.run()
        else:
            st.error(f"Module '{tool_module_name}' does not have a run() function.")
    except ModuleNotFoundError:
        st.error(f"Module '{tool_module_name}' not found. Please ensure the file exists.")
    except Exception as e:
        st.error(f"Error running '{tool_module_name}': {e}")

if tool == "Descriptive Statistics":
    load_and_run("descriptive_tool")
elif tool == "Probability":
    load_and_run("probability_tool")
elif tool == "Discrete Distributions":
    load_and_run("discrete_dist_tool")
# elif tool == "Continuous Distributions":
#     load_and_run("continuous_dist_tool")
# elif tool == "Confidence Interval":
#     load_and_run("ci_tool")
# elif tool == "Hypothesis Testing (One Sample)":
#     load_and_run("ht_one_sample_tool")
# elif tool == "Hypothesis Testing (Two Samples)":
#     load_and_run("ht_two_samples_tool")
# elif tool == "Chi-Square Test":
#     load_and_run("chi_square_tool")
# elif tool == "ANOVA":
#     load_and_run("anova_tool")
# elif tool == "Regression Analysis":
#     load_and_run("regression_tool")

st.markdown("""
---
📘 Explore statistics with interactive tools built for conceptual clarity, practice, and fun.

👨‍🏫 **About the Creator:** Professor Edward Pineda-Castro  
Department of Mathematics, Los Angeles City College  
📧 Email: pinedaem@lacitycollege.edu | 📞 Tel: (323) 953-4000 ext. 2827  
Founder of **MIND** — *Making Inference Digestible*
""")
