import streamlit as st
import streamlit.components.v1 as components


def run():
    st.header("📊 Excel / Google Sheets")

    st.markdown("""
    Use the spreadsheet below to explore data and perform calculations.
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
        "Editing requires permission to edit the Google Sheet. "
        "If the embedded sheet does not load, open it in a new tab."
    )


if __name__ == "__main__":
    run()
