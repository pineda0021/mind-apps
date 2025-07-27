import streamlit as st

# Import all tool modules
import limits_tool
import derivative_tool
import riemann_tool
import antiderivative_tool
import solid_volume_tool

# App title and branding
st.set_page_config(page_title="MIND: Calculus Visualizer Suite", layout="wide")
st.title("🧠 MIND: Calculus Visualizer Suite")
st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND.")

# Sidebar navigation
st.sidebar.header("📚 Select a Concept")
tool = st.sidebar.radio("Choose a tool:", [
    "Limits Visualizer",
    "Derivative Visualizer",
    "Riemann Sum Explorer",
    "Antiderivative Visualizer",
    "Solid of Revolution Tool"
])

# Route to selected module
if tool == "Limits Visualizer":
    limits_tool.run()
elif tool == "Derivative Visualizer":
    derivative_tool.run()
elif tool == "Riemann Sum Explorer":
    riemann_tool.run()
elif tool == "Antiderivative Visualizer":
    antiderivative_tool.run()
elif tool == "Solid of Revolution Tool":
    solid_volume_tool.run()

# Footer
st.markdown("""
---
📘 Explore calculus with interactive tools built for conceptual clarity, practice, and fun.

ℹ️ **Note:** Each visualizer includes explanations and symbolic math support using SymPy.
""")
