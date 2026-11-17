import streamlit as st
import streamlit.components.v1 as components


# ---------- Google Sheets Module ----------
def run():
    st.header("📊 Excel / Google Sheets")

    st.markdown("""
    Use the spreadsheet below to explore data, perform calculations,
    and practice statistical analysis.
    """)

    sheet_url = (
        "https://docs.google.com/spreadsheets/d/"
        "1HJFJHeY4pinkHCuJFn-rv-q-YCzVm9bxI8APlJG0Tuw"
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
        "You must have editing permission to change the spreadsheet. "
        "If it does not load here, use the link above."
    )import streamlit as st
import streamlit.components.v1 as components


# ---------- Google Sheets Module ----------
def run():
    st.header("📊 Excel / Google Sheets")

    st.markdown("""
    Use the spreadsheet below to explore data, perform calculations,
    and practice statistical analysis.
    """)

    sheet_url = (
        "https://docs.google.com/spreadsheets/d/"
        "1HJFJHeY4pinkHCuJFn-rv-q-YCzVm9bxI8APlJG0Tuw"
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
        "You must have editing permission to change the spreadsheet. "
        "If it does not load here, use the link above."
    )
