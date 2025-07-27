# main.py
import streamlit as st

st.set_page_config("🧠 MIND: Calculus Suite", layout="wide")
st.title("🧠 MIND: Calculus Visualizer Suite")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College")

st.markdown("""
Welcome to the **MIND** Calculus Visualizer Suite! Use the dropdown below to explore interactive tools for:
- Limits and Discontinuities
- Derivatives
- Riemann Sums
- Antiderivatives
- Solids of Revolution
""")

option = st.selectbox("Select a tool:", [
    "Limits Visualizer",
    "Derivative Visualizer",
    "Riemann Sum Explorer",
    "Antiderivative Visualizer",
    "Solid of Revolution Tool"
])

if option == "Limits Visualizer":
    import limits_tool
elif option == "Derivative Visualizer":
    import derivative_tool
elif option == "Riemann Sum Explorer":
    import riemann_tool
elif option == "Antiderivative Visualizer":
    import antiderivative_tool
elif option == "Solid of Revolution Tool":
    import solid_volume_tool
