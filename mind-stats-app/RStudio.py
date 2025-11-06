# ======================================================
# RStudio.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ======================================================

import streamlit as st

def run():
    st.title("📘 RStudio: Run R in Google Colab")

    st.write("""
    This page connects you directly to **Google Colab** where you can run **R code interactively**.  
    It serves as a cloud-based version of RStudio, allowing you to execute scripts, visualize data,  
    and explore statistical analysis in real time without local installation.
    """)

    # --- URLs ---
    colab_url = "https://colab.research.google.com/drive/1hhooUeCY8h4zWiUVpcrd2nuU8GBZpQPH?usp=sharing"
    nbviewer_url = "https://nbviewer.org/github/pineda0021/mind-apps/blob/9d010d7c832f6c55d037898a3872683dd7b06f43/RStudio_Notebook.ipynb.ipynb"

    # --- Launch Button ---
    st.markdown(
        f"""
        <div style="text-align:center; margin-top:20px; margin-bottom:30px;">
            <a href="{colab_url}" target="_blank">
                <button style="
                    background-color:#0072B2;
                    color:white;
                    padding:12px 28px;
                    border:none;
                    border-radius:10px;
                    cursor:pointer;
                    font-size:18px;">
                    🚀 Open R Notebook in Google Colab
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Read-Only Preview ---
    st.markdown("---")
    st.subheader("📄 Preview (Read-Only)")
    st.write("""
    You can scroll through the notebook below.  
    To **execute R code**, click the “Open in Colab” button above.
    """)

    st.markdown(
        f"""
        <iframe src="{nbviewer_url}" width="100%" height="600" style="border:none;">
        </iframe>
        """,
        unsafe_allow_html=True
    )

    # --- Footer ---
    st.markdown("---")
    st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND 🎓")
