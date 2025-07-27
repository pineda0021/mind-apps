# main.py
import streamlit as st
import os

st.set_page_config("MIND: Calculus Visualizer Suite", layout="wide")
st.title("🧠 MIND: Calculus Visualizer Suite")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College")

st.markdown("""
Welcome! Use the selector below to explore interactive calculus tools:
- Visualize **limits**, **derivatives**, **Riemann sums**, **antiderivatives**, and **solids of revolution**
- Step-by-step symbolic solutions and graphs
- Includes embedded student challenges!
""")

# Navigation
option = st.selectbox("📚 Choose a module:", [
    "Limits", "Derivative", "Riemann Sum", "Antiderivative", "Solid of Revolution"
])

if option == "Limits":
    exec(open("limits_tool.py").read())
elif option == "Derivative":
    exec(open("derivative_tool.py").read())
elif option == "Riemann Sum":
    exec(open("riemann_tool.py").read())
elif option == "Antiderivative":
    exec(open("antiderivative_tool.py").read())
elif option == "Solid of Revolution":
    exec(open("solid_volume_tool.py").read())


