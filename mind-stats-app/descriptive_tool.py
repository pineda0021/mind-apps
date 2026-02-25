# ==========================================================
# Descriptive Statistics Tool
# Plotly Version (Original Structure Preserved)
# Created by Professor Edward Pineda-Castro
# MIND: Statistics Visualizer Suite
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import re
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# Helper Functions
# ==========================================================

def get_summary_stats(data, decimals=2):
    minimum = round(np.min(data), decimals)
    q1 = round(np.percentile(data, 25), decimals)
    median = round(np.median(data), decimals)
    q3 = round(np.percentile(data, 75), decimals)
    maximum = round(np.max(data), decimals)
    iqr = round(q3 - q1, decimals)

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = sorted([round(x, decimals) for x in data if x < lower_bound or x > upper_bound])

    mean = round(np.mean(data), decimals)
    mode_result = stats.mode(data, keepdims=True)
    mode = ", ".join(map(str, np.round(mode_result.mode, decimals)))

    range_val = round(maximum - minimum, decimals)
    pop_var = round(np.var(data, ddof=0), decimals)
    pop_std = round(np.std(data, ddof=0), decimals)
    samp_var = round(np.var(data, ddof=1), decimals)
    samp_std = round(np.std(data, ddof=1), decimals)

    return {
        "Minimum": minimum, "Q1": q1, "Median": median, "Q3": q3,
        "Maximum": maximum, "IQR": iqr,
        "Lower Bound": round(lower_bound, decimals),
        "Upper Bound": round(upper_bound, decimals),
        "Outliers": outliers if outliers else "None",
        "Mean": mean, "Mode": mode, "Range": range_val,
        "σ² (Population Variance)": pop_var, "σ (Population Std Dev)": pop_std,
        "s² (Sample Variance)": samp_var, "s (Sample Std Dev)": samp_std
    }

def parse_intervals(interval_text):
    intervals = []
    pattern = r"\[(\s*\d+\.?\d*\s*),(\s*\d+\.?\d*\s*)\]"
    matches = re.findall(pattern, interval_text)
    for match in matches:
        try:
            low = float(match[0])
            high = float(match[1])
            if low < high:
                intervals.append((low, high))
        except:
            continue
    return intervals

def compute_frequency_table(data, intervals, manual_freq=None):
    if manual_freq:
        freq = manual_freq
    else:
        freq = [np.sum((data >= low) & (data <= high)) for low, high in intervals]
    total = np.sum(freq)
    rel_freq = [round(f / total, 4) if total > 0 else 0 for f in freq]
    cum_freq = np.cumsum(freq)
    df = pd.DataFrame({
        "Class Interval": [f"[{int(low)},{int(high)}]" for low, high in intervals],
        "Frequency": freq,
        "Relative Freq": rel_freq,
        "Cumulative Freq": cum_freq
    })
    return df, total

# ==========================================================
# Summary Statistics & Boxplots (RESTORED)
# ==========================================================

def run_summary(df_uploaded=None):
    st.subheader("📊 Summary Statistics & Boxplots")

    mode = st.radio("Mode:", ["Single Dataset", "Multiple Datasets"], horizontal=True)

    # ================= SINGLE DATASET =================

    if mode == "Single Dataset":
        input_mode = st.radio("Input:", ["Upload File", "Manual Entry"], horizontal=True)

        if input_mode == "Upload File":
            if df_uploaded is None:
                st.warning("Upload file first.")
                return
            numeric_cols = df_uploaded.select_dtypes(include=[np.number]).columns
            col = st.selectbox("Column:", numeric_cols)
            data = df_uploaded[col].dropna().astype(float).values
        else:
            raw = st.text_area("Numbers:", "56, 57, 54, 61, 63, 58, 59, 62")
            try:
                data = np.array([float(x.strip()) for x in raw.split(",") if x.strip()])
            except:
                st.error("Invalid input.")
                return

        summary = pd.DataFrame(get_summary_stats(data).items(),
                               columns=["Statistic", "Value"])
        st.dataframe(summary, use_container_width=True)

        fig = px.box(x=data, orientation="h")
        fig.update_layout(
            title="📦 Boxplot",
            xaxis_title="Values"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ================= MULTIPLE DATASETS =================

    else:
        input_mode = st.radio("Input:", ["Upload File", "Manual Entry"], horizontal=True)
        data_dict = {}

        if input_mode == "Upload File":
            if df_uploaded is None:
                st.warning("Upload first.")
                return

            numeric_cols = df_uploaded.select_dtypes(include=[np.number]).columns
            selected = st.multiselect(
                "Select numeric columns:",
                numeric_cols,
                default=list(numeric_cols)[:2]
            )

            for col in selected:
                data_dict[col] = df_uploaded[col].dropna().astype(float).values

        else:
            raw = st.text_area(
                "Enter datasets separated by semicolons:",
                "56,57,54; 49,51,55; 65,64,68"
            )
            blocks = [b.strip() for b in raw.split(";") if b.strip()]

            for i, block in enumerate(blocks, 1):
                try:
                    data_dict[f"Dataset {i}"] = np.array(
                        [float(x.strip()) for x in block.split(",") if x.strip()]
                    )
                except:
                    st.error(f"Invalid input in Dataset {i}")
                    return

        combined = pd.DataFrame()
        for name, d in data_dict.items():
            stats_dict = get_summary_stats(d)
            df_stats = pd.DataFrame(stats_dict, index=[name])
            combined = pd.concat([combined, df_stats])

        st.dataframe(combined, use_container_width=True)

        fig = go.Figure()

        for name, d in data_dict.items():
            fig.add_trace(go.Box(
                x=d,
                name=name,
                orientation="h"
            ))

        fig.update_layout(
            title="📦 Boxplots (Multiple)",
            xaxis_title="Values"
        )

        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# MAIN APP
# ==========================================================

def run():
    st.header("🧠 Descriptive Statistics Tool")

    categories = [
        "Qualitative (Categorical)",
        "Quantitative (Discrete or Continuous)",
        "Summary Statistics & Boxplot"
    ]

    choice = st.selectbox("Choose a category:", categories, index=None)

    if not choice:
        st.info("👆 Please select a category to begin.")

    uploaded_file = st.file_uploader("📂 Upload CSV or Excel (optional):",
                                     type=["csv", "xlsx"])
    df_uploaded = None

    if uploaded_file:
        try:
            df_uploaded = (
                pd.read_csv(uploaded_file)
                if uploaded_file.name.endswith(".csv")
                else pd.read_excel(uploaded_file)
            )
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return

    if choice == "Summary Statistics & Boxplot":
        run_summary(df_uploaded)

if __name__ == "__main__":
    run()
