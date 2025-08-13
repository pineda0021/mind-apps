import streamlit as st

st.set_page_config(page_title="MIND: Statistics Visualizer", layout="wide")
st.title("🧠 MIND: Statistics Visualizer Suite")

st.sidebar.header("📚 Select a Concept")
tool = st.sidebar.radio("Choose a tool:", [
    "Descriptive Statistics",
    "Probability",
    "Discrete Distributions",
])

if tool == "Descriptive Statistics":
    try:
        import descriptive_tool
        descriptive_tool.run()
    except Exception as e:
        st.error(f"Error loading descriptive_tool: {e}")
elif tool == "Probability":
    try:
        import probability_tool
        probability_tool.run()
    except Exception as e:
        st.error(f"Error loading probability_tool: {e}")
elif tool == "Discrete Distributions":
    try:
        import discrete_dist_tool
        discrete_dist_tool.run()
    except Exception as e:
        st.error(f"Error loading discrete_dist_tool: {e}")

st.markdown("""
---
📘 Explore statistics with interactive tools built for conceptual clarity, practice, and fun.

👨‍🏫 **About the Creator:** Professor Edward Pineda-Castro  
Department of Mathematics, Los Angeles City College  
📧 Email: pinedaem@lacitycollege.edu | 📞 Tel: (323) 953-4000 ext. 2827  
Founder of **MIND** — *Making Inference Digestible*
""")

📘 Explore statistics with interactive tools built for conceptual clarity, practice, and fun.

👨‍🏫 **About the Creator:** Professor Edward Pineda-Castro  
Department of Mathematics, Los Angeles City College  
📧 Email: pinedaem@lacitycollege.edu | 📞 Tel: (323) 953-4000 ext. 2827  
Founder of **MIND** — *Making Inference Digestible*
""")
