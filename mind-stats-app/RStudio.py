# ======================================================
# RStudio.py
# Created by Professor Edward Pineda-Castro, Los Angeles City College
# Part of the MIND: Statistics Visualizer Suite
# ======================================================

import streamlit as st

def run():
    st.title("📘 RStudio: Run R in Google Colab")

    st.write("""
    This page connects you directly to **Google Colab**, where you can run **R code interactively**.  
    It serves as a cloud-based version of RStudio — allowing you to execute scripts, visualize data,  
    and perform statistical analyses in real time without installing any software locally.
    """)

    # --- Colab Notebook Link ---
    colab_url = "https://colab.research.google.com/drive/1hhooUeCY8h4zWiUVpcrd2nuU8GBZpQPH?usp=sharing"

    # --- Google Colab Logo (official SVG) ---
    colab_logo = "https://colab.research.google.com/img/colab_favicon_256px.png"

    # --- Centered Button with Logo ---
    st.markdown(
        f"""
        <div style="text-align:center; margin-top:40px;">
            <a href="{colab_url}" target="_blank" style="text-decoration:none;">
                <button style="
                    background-color:#3B6EA5;
                    color:white;
                    padding:16px 34px;
                    border:none;
                    border-radius:16px;
                    font-size:20px;
                    font-weight:600;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    gap:12px;
                    cursor:pointer;">
                    <img src="{colab_logo}" alt="Colab" width="28" height="28" style="vertical-align:middle; border-radius:5px;">
                    Open R Notebook in Google Colab
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
