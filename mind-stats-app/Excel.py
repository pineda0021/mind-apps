import streamlit as st
import streamlit.components.v1 as components


def run():
    st.header("📊 Excel / Google Sheets")

    st.markdown("""
    Use the spreadsheet below to explore data,
    perform calculations, and practice statistical analysis.
    """)

    sheet_url = (
        "https://docs.google.com/spreadsheets/d/"
        "1YkDzA_ED9vT5L2EL16QOFWBdPY3z7j33k3fn7MBUYP8"
        "/edit?usp=sharing"
    )

    components.iframe(
        src=sheet_url,
        height=800,
        scrolling=True
    )

    st.markdown(
        f"[Open the spreadsheet in a new tab]({sheet_url})"
    )

    st.info(
        "Editing requires permission to edit the Google Sheet. "
        "If the spreadsheet does not load, open it in a new tab."
    )
