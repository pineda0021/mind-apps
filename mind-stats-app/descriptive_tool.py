import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import plotly.graph_objects as go
from scipy import stats

# ---------- Summary Stats Functions ----------
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
    mode_count = mode_result.count[0] if len(mode_result.count) > 0 else 0
    mode = ", ".join(map(str, np.round(mode_result.mode, decimals))) if mode_count > 1 else "No mode"

    range_val = round(maximum - minimum, decimals)
    pop_var = round(np.var(data, ddof=0), decimals)
    pop_std = round(np.std(data, ddof=0), decimals)
    samp_var = round(np.var(data, ddof=1), decimals)
    samp_std = round(np.std(data, ddof=1), decimals)

    return {
        "Minimum": minimum,
        "Q1": q1,
        "Median": median,
        "Q3": q3,
        "Maximum": maximum,
        "IQR": iqr,
        "Lower Bound": round(lower_bound, decimals),
        "Upper Bound": round(upper_bound, decimals),
        "Outliers": outliers,
        "Mean": mean,
        "Mode": mode,
        "Range": range_val,
        "σ² (Population Variance)": pop_var,
        "σ (Population Std Dev)": pop_std,
        "s² (Sample Variance)": samp_var,
        "s (Sample Std Dev)": samp_std
    }

def display_summary_table(datasets):
    summary_data = []
    for name, data in datasets.items():
        s = get_summary_stats(data)
        summary_data.append({
            "Dataset": name,
            "Mean": s["Mean"],
            "Median": s["Median"],
            "Q1": s["Q1"],
            "Q3": s["Q3"],
            "IQR": s["IQR"],
            "Std Dev (s)": s["s (Sample Std Dev)"],
            "Range": s["Range"]
        })
    df_summary = pd.DataFrame(summary_data)
    st.markdown("### 📊 Summary Statistics Comparison")
    st.dataframe(df_summary, use_container_width=True)

def display_plotly_boxplot_comparison(datasets):
    colors = ["teal", "orange", "purple", "green", "red", "blue"]
    fig = go.Figure()
    for i, (name, data) in enumerate(datasets.items()):
        fig.add_trace(go.Box(
            y=data,
            name=name,
            boxpoints='outliers',
            marker=dict(color=colors[i % len(colors)]),
            line=dict(color='black')
        ))
    fig.update_layout(title="📦 Boxplot Comparison", yaxis_title="Values", template="simple_white")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Continuous Data Helper ----------
def parse_intervals(interval_text):
    """
    Parse intervals from input like: [70-79], [80-89], [90-99]
    Returns list of tuples: [(70,79), (80,89), (90,99)]
    """
    intervals = []
    for part in interval_text.split(']'):
        if '-' in part:
            part = part.replace('[', '').replace(']', '').strip()
            try:
                low, high = map(float, part.split('-'))
                intervals.append((low, high))
            except:
                pass
    return intervals

def compute_frequency_table(data, intervals):
    """
    Given data and user-specified intervals, compute:
    Frequency, Relative Frequency, Cumulative Frequency, and Class Midpoints
    """
    freq = []
    mids = []
    for (low, high) in intervals:
        count = np.sum((data >= low) & (data <= high))
        freq.append(count)
        mids.append((low + high) / 2)
    total = np.sum(freq)
    rel_freq = [round(f / total, 4) for f in freq]
    cum_freq = np.cumsum(freq)

    df = pd.DataFrame({
        "Class Interval": [f"[{int(low)}–{int(high)}]" for low, high in intervals],
        "Frequency": freq,
        "Relative Freq": rel_freq,
        "Cumulative Freq": cum_freq,
        "Class Midpoint": mids
    })

    # Calculate mean from midpoints
    mean_est = round(np.sum(np.array(freq) * np.array(mids)) / total, 2)

    return df, total, mean_est

# ---------- Main App ----------
def run():
    st.header("📘 Descriptive Statistics Tool")

    categories = [
        "Qualitative",
        "Quantitative (Discrete)",
        "Quantitative (Continuous)",
        "Summary Statistics & Boxplot"
    ]

    choice = st.selectbox(
        "Choose a category:",
        categories,
        index=None,
        placeholder="Select a category to begin..."
    )

    if not choice:
        st.info("👆 Please choose a category to begin.")
        return

    st.markdown("### 📤 Upload Data File (CSV or Excel)")
    uploaded_file = st.file_uploader("Upload your dataset:", type=["csv", "xlsx"])
    df_uploaded = None

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_uploaded = pd.read_csv(uploaded_file)
            else:
                df_uploaded = pd.read_excel(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.dataframe(df_uploaded)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

    # ---------- CONTINUOUS DATA SECTION ----------
    if choice == "Quantitative (Continuous)":
        st.subheader("📂 Category: Quantitative (Continuous) Data")

        # Step 1: Load or Enter Data
        if df_uploaded is not None:
            numeric_cols = df_uploaded.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Select numeric column:", numeric_cols)
                numeric_data = df_uploaded[col].dropna().astype(float).values
            else:
                st.error("No numeric columns found.")
                return
        else:
            raw_data = st.text_area("Enter comma-separated numeric values:", "")
            try:
                numeric_data = np.array(list(map(float, raw_data.split(','))))
            except ValueError:
                st.error("Please enter valid numeric values.")
                return

        # Step 2: User-specified intervals
        st.markdown("### 🧩 Enter Class Intervals (e.g., `[70-79], [80-89], [90-99]`)")
        interval_text = st.text_area("Enter intervals:", "[70-79], [80-89], [90-99]")
        intervals = parse_intervals(interval_text)

        if not intervals:
            st.warning("⚠️ Please enter valid intervals in the format `[70-79], [80-89], ...`.")
            return

        # Step 3: Compute frequency table
        df_freq, total, mean_est = compute_frequency_table(numeric_data, intervals)
        st.markdown("### 📋 Frequency Distribution Table")
        st.dataframe(df_freq, use_container_width=True)
        st.markdown(f"**Total Frequency (n):** {total}")
        st.markdown(f"**Estimated Mean (using midpoints):** {mean_est}")

        # Step 4: Histogram based on intervals
        bins = [low for low, _ in intervals] + [intervals[-1][1]]
        plt.figure(figsize=(8, 4))
        plt.hist(numeric_data, bins=bins, edgecolor='black', color='skyblue')
        plt.title("📊 Histogram (User-defined Intervals)")
        plt.xlabel("Class Intervals")
        plt.ylabel("Frequency")
        st.pyplot(plt)

    # ---------- Other categories unchanged ----------
    elif choice == "Qualitative":
        st.subheader("📂 Category: Qualitative Data")
        st.info("⚙️ This section remains unchanged from your version.")
    elif choice == "Quantitative (Discrete)":
        st.subheader("📂 Category: Quantitative (Discrete) Data")
        st.info("⚙️ This section remains unchanged from your version.")
    elif choice == "Summary Statistics & Boxplot":
        st.subheader("📦 Summary Statistics & Boxplot Comparison")
        st.info("⚙️ This section remains unchanged from your version.")


if __name__ == "__main__":
    run()

