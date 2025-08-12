import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import io

def run():
    st.header("📊 Descriptive Statistics Tool")
    st.write("Analyze qualitative, discrete, or continuous data with frequency tables and charts.")

    # Sidebar menu
    choice = st.sidebar.radio(
        "Select Data Type:",
        ["Qualitative", "Quantitative (Discrete)", "Quantitative (Continuous)"]
    )

    # --- Upload Feature ---
    st.markdown("### 📤 Upload Data File (CSV or Excel)")
    uploaded_file = st.file_uploader("Upload your dataset:", type=["csv", "xlsx"])
    raw_data = ""

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_uploaded = pd.read_csv(uploaded_file)
            else:
                df_uploaded = pd.read_excel(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.dataframe(df_uploaded)

            # Use first column for analysis
            raw_data = df_uploaded.iloc[:, 0].dropna().astype(str).tolist()
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        raw_data = st.text_area("Or enter comma-separated values:", "")

    if raw_data:
        # Convert string -> list if necessary
        if isinstance(raw_data, str):
            data = [val.strip() for val in raw_data.split(',') if val.strip() != ""]
        else:
            data = raw_data  # from uploaded file

        df = None

        if choice == "Qualitative":
            df = display_frequency_table(data)
            st.subheader("Frequency and Relative Frequency Table")
            st.dataframe(df)
            plot_qualitative(df)

        elif choice == "Quantitative (Discrete)":
            try:
                numeric_data = list(map(int, data))
                df = display_frequency_table(numeric_data)
                st.subheader("Frequency and Relative Frequency Table")
                st.dataframe(df)
                plot_histograms(numeric_data, discrete=True)
            except ValueError:
                st.error("Please enter valid integers for discrete quantitative data.")

        elif choice == "Quantitative (Continuous)":
            try:
                numeric_data = list(map(float, data))
                bins = st.slider("Select number of bins", 2, 20, 5)
                df, bin_edges = group_continuous_data(numeric_data, bins)
                st.subheader("Grouped Frequency Table (Continuous Data)")
                st.dataframe(df)
                plot_histograms(numeric_data, discrete=False, bins=bin_edges)
            except ValueError:
                st.error("Please enter valid numeric values for continuous data.")

        # Export Options
        if df is not None:
            st.markdown("### 📥 Download Results")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download as CSV",
                data=csv,
                file_name="frequency_table.csv",
                mime="text/csv"
            )

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            st.download_button(
                label="⬇️ Download as Excel",
                data=excel_buffer,
                file_name="frequency_table.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# ---------- Helper Functions ----------

def display_frequency_table(data):
    freq = dict(Counter(data))
    total = sum(freq.values())
    rel_freq = {k: round(v / total, 4) for k, v in freq.items()}
    df = pd.DataFrame({
        'Category': list(freq.keys()),
        'Frequency': list(freq.values()),
        'Relative Frequency': list(rel_freq.values())
    })
    return df.sort_values(by="Category").reset_index(drop=True)


def plot_qualitative(df):
    labels = df['Category'].astype(str)
    freq = df['Frequency']
    rel_freq = df['Relative Frequency']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, freq, color='skyblue')
    axes[0].set_title('Frequency Bar Chart')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Frequency')

    axes[1].pie(rel_freq, labels=labels, autopct='%.2f%%', startangle=90)
    axes[1].set_title('Pie Chart (Relative Frequency)')

    plt.tight_layout()
    st.pyplot(fig)


def plot_histograms(data, discrete=True, bins=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if discrete:
        # Frequency Histogram
        axes[0].hist(data, bins=range(min(data), max(data) + 2), edgecolor='black', color='lightblue')
        axes[0].set_title('Frequency Histogram')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')

        # Relative Frequency Histogram
        counts, bin_edges = np.histogram(data, bins=range(min(data), max(data) + 2))
        rel_freq = counts / counts.sum()
        axes[1].bar(bin_edges[:-1], rel_freq, width=0.8, align='center', color='salmon', edgecolor='black')
        axes[1].set_title('Relative Frequency Histogram')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Relative Frequency')

    else:
        # Continuous Data Histograms
        axes[0].hist(data, bins=bins, edgecolor='black', color='lightgreen')
        axes[0].set_title('Frequency Histogram')
        axes[0].set_xlabel('Class Intervals')
        axes[0].set_ylabel('Frequency')

        counts, _ = np.histogram(data, bins=bins)
        rel_freq = counts / counts.sum()
        midpoints = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
        axes[1].bar(midpoints, rel_freq, width=(bins[1]-bins[0])*0.9, color='orange', edgecolor='black')
        axes[1].set_title('Relative Frequency Histogram')
        axes[1].set_xlabel('Class Intervals')
        axes[1].set_ylabel('Relative Frequency')

    plt.tight_layout()
    st.pyplot(fig)


def group_continuous_data(data, bins=5):
    counts, bin_edges = np.histogram(data, bins=bins)
    rel_freq = np.round(counts / counts.sum(), 4)
    categories = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]

    df = pd.DataFrame({
        'Class Interval': categories,
        'Frequency': counts,
        'Relative Frequency': rel_freq
    })
    return df, bin_edges
