import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title="MIND: Statistics Visualizer", layout="wide")

st.title("🧠 MIND: Statistics Visualizer Suite")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

# Safe imports — do not crash if missing
def safe_import(name):
    try:
        return __import__(name)
    except ImportError:
        return None

descriptive_tool = safe_import("descriptive_tool")
probability_tool = safe_import("probability_tool")
discrete_dist_tool = safe_import("discrete_dist_tool")

# Sidebar
st.sidebar.header("📚 Select a Concept")
tool = st.sidebar.radio("Choose a tool:", [
    "Descriptive Statistics",
    "Probability",
    "Discrete Distributions"
])

# Route
if tool == "Descriptive Statistics":
    if descriptive_tool and hasattr(descriptive_tool, "run"):
        descriptive_tool.run()
    else:
        st.warning("Module not available yet.")

elif tool == "Probability":
    if probability_tool and hasattr(probability_tool, "run"):
        probability_tool.run()
    else:
        st.warning("Module not available yet.")

elif tool == "Discrete Distributions":
    if discrete_dist_tool and hasattr(discrete_dist_tool, "run"):
        discrete_dist_tool.run()
    else:
        st.warning("Module not available yet.")

# Footer
st.markdown("""
---
📘 Explore statistics with interactive tools built for conceptual clarity, practice, and fun.

👨‍🏫 **About the Creator:** Professor Edward Pineda-Castro  
Department of Mathematics, Los Angeles City College  
📧 Email: pinedaem@lacitycollege.edu | 📞 Tel: (323) 953-4000 ext. 2827  
Founder of **MIND** — *Making Inference Digestible*
""")
