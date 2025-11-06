# ======================================================
# RStudio.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ======================================================

import streamlit as st

def run():
    st.title("📘 RStudio & Google Colab Integration")

    st.write("""
    Welcome to the **R Notebook Companion** —  
    this tool connects you directly to Google Colab to run **R code interactively** in the cloud.  
    Use it to perform simulations, statistical analysis, or data visualization in R,  
    complementing your work in the **MIND: Statistics Visualizer**.
    """)

    # --- Display side-by-side launch buttons
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧮 TI-84 Calculator")
        st.markdown(
            """
            <a href="https://www.ti84calcwiz.com" target="_blank">
                <button style="background-color:#4CAF50;color:white;
                padding:10px 20px;border:none;border-radius:10px;
                cursor:pointer;font-size:16px;">
                🔢 Open TI-84 Emulator
                </button>
            </a>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.subheader("📊 R Colab Notebook")
        st.markdown(
            """
            <a href="https://colab.research.google.com/drive/1hhooUeCY8h4zWiUVpcrd2nuU8GBZpQPH?usp=sharing"
               target="_blank">
                <button style="background-color:#0072B2;color:white;
                padding:10px 20px;border:none;border-radius:10px;
                cursor:pointer;font-size:16px;">
                🚀 Launch R Notebook
                </button>
            </a>
            """,
            unsafe_allow_html=True
        )

    # --- Optional embedded Colab preview
    st.markdown("---")
    st.subheader("📄 Preview (Read-Only)")
    st.markdown("""
    You can scroll and preview the notebook below.  
    To execute the R code, click **Open in Colab** in the frame header.
    """)

    st.markdown(
        """
        <iframe src="https://colab.research.google.com/drive/1hhooUeCY8h4zWiUVpcrd2nuU8GBZpQPH?usp=sharing"
                width="100%" height="600" style="border:none;" allowfullscreen>
        </iframe>
        """,
        unsafe_allow_html=True
    )

    # --- Footer
    st.markdown("---")
    st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND 🎓")

