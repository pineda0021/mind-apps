import streamlit as st
import pandas as pd
from lambda_transform import lambda_transform


def run():

    st.header("Box–Cox Transformation Tool")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload a CSV file.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Unable to read CSV file.")
        return

    if df.empty:
        st.warning("Uploaded CSV is empty.")
        return

    st.subheader("Data Preview")
    st.dataframe(df.head())

    response = st.selectbox("Select Response Variable", df.columns)

    df_model, transformed, chosen_lambda, transformed_response = \
        lambda_transform(df, response)

    if transformed:
        st.success(f"λ selected: {chosen_lambda}")
        st.write(f"Transformed variable created: {transformed_response}")
    else:
        st.warning("Box–Cox not applied (check positivity and variation).")
