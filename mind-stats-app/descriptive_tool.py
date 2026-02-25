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
# Quantitative Analyzer
# ==========================================================

def run_quantitative(df_uploaded=None):
    st.subheader("📊 Quantitative Data Analyzer")

    q_type = st.radio("Select Data Type:", ["Discrete", "Continuous"], horizontal=True)
    input_mode = st.radio("Data Input Mode:", ["Upload File", "Manual Entry"], horizontal=True)

    data = None

    if input_mode == "Upload File":
        if df_uploaded is not None:
            numeric_cols = df_uploaded.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns found.")
                return
            col = st.selectbox("Select numeric column:", numeric_cols)
            data = df_uploaded[col].dropna().astype(float).values
            st.success(f"Loaded {len(data)} observations.")
        else:
            st.warning("Upload a dataset first.")
            return
    else:
        example = (
            "3, 5, 4, 4, 3, 2, 5, 5, 3, 4" if q_type == "Discrete"
            else "728,730,726,698,721,722,700,720,729,678"
        )
        raw_data = st.text_area("Enter comma-separated numeric values:", example)
        try:
            data = np.array([float(x.strip()) for x in raw_data.split(",") if x.strip()])
            st.success(f"Loaded {len(data)} observations.")
        except:
            st.error("Invalid numeric input.")
            return

    # ---------------- DISCRETE ----------------

    if q_type == "Discrete":
        counts = pd.Series(data).value_counts().sort_index()
        freq_df = pd.DataFrame({
            "Value": counts.index,
            "Frequency": counts.values,
            "Relative Frequency": np.round(counts.values / len(data), 4)
        })
        freq_df["Cumulative Frequency"] = freq_df["Frequency"].cumsum()

        st.markdown("### Frequency Table")
        st.dataframe(freq_df, use_container_width=True)

        fig = px.histogram(
            x=data,
            nbins=len(np.unique(data))
        )
        fig.update_layout(
            title="📊 Discrete Histogram (No Gaps)",
            xaxis_title="Values",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    # ---------------- CONTINUOUS ----------------

    min_val, max_val = np.min(data), np.max(data)
    n = len(data)
    k = int(np.ceil(1 + 3.322 * np.log10(n)))
    class_width = np.ceil((max_val - min_val) / k)

    auto_intervals = []
    start = np.floor(min_val)
    for i in range(k):
        low = start + i * class_width
        high = low + class_width - 1
        auto_intervals.append((low, high))

    default_intervals_text = ", ".join(f"[{int(l)},{int(h)}]" for l, h in auto_intervals)

    st.info(f"Generated {k} class intervals (width ≈ {int(class_width)}).")

    edit = st.checkbox("Edit class intervals?", value=False)
    intervals = parse_intervals(
        st.text_area("Class Intervals:", default_intervals_text)
    ) if edit else auto_intervals

    df_freq, total = compute_frequency_table(data, intervals)

    st.markdown("### Frequency Table")
    st.dataframe(df_freq, use_container_width=True)

    plot_option = st.radio(
        "Choose visualization:",
        ["Histogram", "Histogram + Ogive", "Boxplot"],
        horizontal=True
    )

    bins = [low for low, _ in intervals] + [intervals[-1][1]]

    if plot_option == "Histogram":
        fig = px.histogram(x=data, nbins=len(intervals))
        fig.update_layout(
            title="📊 Continuous Histogram (No Gaps)",
            xaxis_title="Class Intervals",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif plot_option == "Histogram + Ogive":
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, nbinsx=len(intervals), name="Frequency"))

        upper_bounds = [high for _, high in intervals]
        cum_freq = df_freq["Cumulative Freq"].values

        fig.add_trace(go.Scatter(
            x=upper_bounds,
            y=cum_freq,
            mode="lines+markers",
            name="Cumulative Frequency",
            yaxis="y2"
        ))

        fig.update_layout(
            title="📊 Histogram + Ogive",
            xaxis_title="Class Intervals",
            yaxis_title="Frequency",
            yaxis2=dict(
                overlaying="y",
                side="right",
                title="Cumulative Frequency"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        fig = px.box(x=data, orientation="h")
        fig.update_layout(
            title="📦 Boxplot",
            xaxis_title="Values"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# QUALITATIVE ANALYZER
# ==========================================================

def run_qualitative(df_uploaded=None):
    st.subheader("🎨 Qualitative (Categorical) Analyzer")

    input_mode = st.radio("Input Mode:", ["Upload File", "Manual Entry"], horizontal=True)

    if input_mode == "Upload File":
        if df_uploaded is None:
            st.warning("Upload file first.")
            return
        text_cols = df_uploaded.select_dtypes(include="object").columns
        if not len(text_cols):
            st.error("No categorical columns found.")
            return
        col = st.selectbox("Select column:", text_cols)
        data = df_uploaded[col].dropna().astype(str).values
    else:
        raw_data = st.text_area("Categories:", "Red, Blue, Red, Green, Yellow")
        data = [x.strip() for x in raw_data.split(",") if x.strip()]

    counts = pd.Series(data).value_counts()

    freq_df = pd.DataFrame({
        "Category": counts.index,
        "Frequency": counts.values,
        "Relative Freq": np.round(counts.values / len(data), 4)
    })

    st.dataframe(freq_df, use_container_width=True)

    chart_type = st.radio("Choose chart:", ["Bar Chart", "Pie Chart"], horizontal=True)

    if chart_type == "Bar Chart":
        fig = px.bar(x=counts.index, y=counts.values)
        fig.update_layout(
            title="🎨 Bar Chart",
            xaxis_title="Category",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.pie(names=counts.index, values=counts.values)
        fig.update_layout(title="🥧 Pie Chart")
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

    uploaded_file = st.file_uploader("📂 Upload CSV or Excel (optional):", type=["csv", "xlsx"])
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

    if choice == "Qualitative (Categorical)":
        run_qualitative(df_uploaded)
    elif choice == "Quantitative (Discrete or Continuous)":
        run_quantitative(df_uploaded)

if __name__ == "__main__":
    run()
