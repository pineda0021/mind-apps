import streamlit as st

def run():
    st.header("🧮 TI-84 Plus CE Online Calculator")

    st.markdown("""
    Use the embedded **TI-84 graphing calculator** below to perform quick calculations, graph functions,
    or verify results from your statistical analyses.
    """)

    st.components.v1.iframe(
        "https://ti84hub.com/",
        height=700,
        width=900,
        scrolling=True
    )

    st.markdown("""
    ---
    💡 **Tip:** You can press the blue `ON` button on the calculator below to start it.  
    Use your keyboard for number entry and arrow keys for navigation.
    """)
