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

    # --- Colab Link ---
    colab_url = "https://colab.research.google.com/drive/1hhooUeCY8h4zWiUVpcrd2nuU8GBZpQPH?usp=sharing"

    # --- Launch Button ---
    st.markdown(
        f"""
        <div style="text-align:center; margin-top:40px; margin-bottom:40px;">
            <a href="{colab_url}" target="_blank">
                <button style="
                    background-color:#0072B2;
                    color:white;
                    padding:14px 34px;
                    border:none;
                    border-radius:12px;
                    cursor:pointer;
                    font-size:20px;">
                    🚀 Open R Notebook in Google Colab
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Helpful Note ---
    st.info("""
    💡 **Tip:** Once the notebook opens in Colab, choose **Runtime → Change runtime type → R**  
    to enable the R environment before running the code.
    """)

    # --- Footer ---
    st.markdown("---")
    st.caption("Created by Professor Edward Pineda-Castro, Los Angeles City College — built with the students in MIND 🎓")
