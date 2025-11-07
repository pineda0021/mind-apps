import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import re

# ==========================================================
# Helper Functions
# ==========================================================

def get_summary_stats(data, decimals=2):
    """Compute summary statistics for a numeric dataset."""
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
        "Minimum": minimum,
        "Q1": q1,
        "Median": median,
        "Q3": q3,
        "Maximum": maximum,
        "IQR": iqr,
        "Lower Bound": round(lower_bound, decimals),
        "Upper Bound": round(upper_bound, decimals),
        "Outliers": outliers if outliers else "None",
        "Mean": mean,
        "Mode": mode,
        "Range": range_val,
        "σ² (Population Variance)": pop_var,
        "σ (Population Std Dev)": pop_std,
        "s² (Sample Variance)": samp_var,
        "s (Sample Std Dev)": samp_std
    }

def parse_intervals(interval_text):
    """Parse intervals like [670,679], [680,689], etc."""
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
    """Compute frequencies or accept manual ones."""
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
# Quantitative Analyzer (Discrete + Continuous)
# ==========================================================

def run_quantitative(df_uploaded=None):
    st.subheader("📊 Quantitative Data Analyzer")

    st.markdown("""
    Analyze **quantitative (numeric)** data — either *discrete* or *continuous*.  
    Generate frequency tables, visualizations, and summary insights.
    """)

    q_type = st.radio("Select Data Type:", ["Discrete", "Continuous"], horizontal=True)
    input_mode = st.radio("Data Input Mode:", ["Upload File", "Manual Entry"], horizontal=True)

    data = None

    # ---------- Data Input ----------
    if input_mode == "Upload File":
        if df_uploaded is not None:
            numeric_cols = df_uploaded.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns found.")
                return
            col = st.selectbox("Select numeric column:", numeric_cols)
            data = df_uploaded[col].dropna().astype(float).values
            st.success(f"✅ Loaded {len(data)} observations from '{col}'")
        else:
            st.warning("Please upload a dataset first.")
            return
    else:
        example = "3, 5, 4, 4, 3, 2, 5, 5, 3, 4" if q_type == "Discrete" else \
                  "728,730,726,698,721,722,700,720,729,678,722,716,702"
        raw_data = st.text_area("Enter comma-separated numeric values:", example)
        try:
            data = np.array([float(x.strip()) for x in raw_data.split(",") if x.strip() != ""])
            st.success(f"✅ Loaded {len(data)} observations.")
        except ValueError:
            st.error("❌ Invalid numeric input.")
            return

    # ---------- DISCRETE ----------
    if q_type == "Discrete":
        counts = pd.Series(data).value_counts().sort_index()
        freq_df = pd.DataFrame({
        "Value": counts.index,
        "Frequency": counts.values,
        "Relative Frequency": np.round(counts.values / len(data), 4)
    })
    freq_df["Cumulative Frequency"] = freq_df["Frequency"].cumsum()

    st.markdown("### 📋 Frequency Distribution Table")
    st.dataframe(freq_df, use_container_width=True)
    st.markdown(f"**Total Frequency (n):** {len(data)}")

    # ✅ FIXED: Histogram with no gaps for discrete numeric data
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=np.arange(min(data) - 0.5, max(data) + 1.5, 1),
            color="skyblue", edgecolor="black")
    ax.set_title("📊 Discrete Data Histogram (No Gaps)")
    ax.set_xlabel("Data Values")
    ax.set_ylabel("Frequency")
    ax.set_xticks(sorted(freq_df["Value"]))
    st.pyplot(fig)

    # ---------- CONTINUOUS ----------
    else:
        min_val = np.min(data)
        max_val = np.max(data)
        n = len(data)
        k = int(np.ceil(1 + 3.322 * np.log10(n)))  # Sturges' Rule
        class_width = np.ceil((max_val - min_val) / k)

        auto_intervals = []
        start = np.floor(min_val)
        for i in range(k):
            low = start + i * class_width
            high = low + class_width - 1
            auto_intervals.append((low, high))

        default_intervals_text = ", ".join([f"[{int(l)},{int(h)}]" for l, h in auto_intervals])
        st.info(f"✅ Generated {k} class intervals (width ≈ {int(class_width)}). You may modify them below.")

        edit_toggle = st.checkbox("✏️ Edit class intervals manually?", value=False)
        if edit_toggle:
            interval_text = st.text_area("Enter custom class intervals (optional):", default_intervals_text)
            intervals = parse_intervals(interval_text)
            if not intervals:
                st.warning("⚠️ Invalid format. Use e.g., [670,679], [680,689], etc.")
                return
        else:
            intervals = auto_intervals

        df_freq, total = compute_frequency_table(data, intervals)
        st.markdown("### 📋 Frequency Distribution Table")
        st.dataframe(df_freq, use_container_width=True)
        st.markdown(f"**Total Frequency (n):** {total}")

        plot_option = st.radio("Choose visualization:",
                               ["Histogram (No Gaps)", "Histogram + Ogive", "Boxplot"],
                               horizontal=True)

        bins = [low for low, _ in intervals] + [intervals[-1][1]]
        fig, ax = plt.subplots(figsize=(8, 4))

        # ✅ Histogram for continuous data — no gaps
        if plot_option in ["Histogram (No Gaps)", "Histogram + Ogive"]:
            ax.hist(data, bins=bins, edgecolor="black", color="skyblue", linewidth=1)
            ax.set_title("📊 Continuous Data Histogram (No Gaps)")
            ax.set_xlabel("Class Intervals")
            ax.set_ylabel("Frequency")
            ax.set_xticks(bins)
            if plot_option == "Histogram + Ogive":
                cum_freq = df_freq["Cumulative Freq"].values
                upper_bounds = [high for _, high in intervals]
                ax2 = ax.twinx()
                ax2.plot(upper_bounds, cum_freq, marker="o", color="darkred", linewidth=2)
                ax2.set_ylabel("Cumulative Frequency", color="darkred")

        elif plot_option == "Boxplot":
            ax.boxplot(data, vert=False, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', color='black'),
                       medianprops=dict(color='red'))
            ax.set_title("📦 Horizontal Boxplot")
            ax.set_xlabel("Values")
            ax.set_yticks([])

        st.pyplot(fig)

# ==========================================================
# Qualitative Analyzer
# ==========================================================

def run_qualitative(df_uploaded=None):
    st.subheader("🎨 Qualitative (Categorical) Data Analyzer")

    st.markdown("""
    Analyze **categorical data** by generating a frequency table  
    and visualizing it with a bar chart or pie chart.
    """)

    input_mode = st.radio("Data Input Mode:", ["Upload File", "Manual Entry"], horizontal=True)
    data = None

    if input_mode == "Upload File":
        if df_uploaded is not None:
            text_cols = df_uploaded.select_dtypes(include=["object"]).columns.tolist()
            if not text_cols:
                st.error("No categorical columns found.")
                return
            col = st.selectbox("Select categorical column:", text_cols)
            data = df_uploaded[col].dropna().astype(str).values
        else:
            st.warning("Please upload a dataset first.")
            return
    else:
        raw_data = st.text_area("Enter comma-separated categories:",
                                "Red, Blue, Red, Green, Blue, Blue, Red, Green, Yellow")
        data = [x.strip() for x in raw_data.split(",") if x.strip() != ""]

    counts = pd.Series(data).value_counts()
    freq_df = pd.DataFrame({
        "Category": counts.index,
        "Frequency": counts.values,
        "Relative Frequency": np.round(counts.values / len(data), 4)
    })
    st.markdown("### 📋 Frequency Distribution Table")
    st.dataframe(freq_df, use_container_width=True)
    st.markdown(f"**Total Categories:** {len(counts)}")

    chart_type = st.radio("Choose visualization:", ["Bar Chart", "Pie Chart"], horizontal=True)
    fig, ax = plt.subplots(figsize=(8, 4))

    if chart_type == "Bar Chart":
        ax.bar(freq_df["Category"], freq_df["Frequency"], color="lightcoral", edgecolor="black")
        ax.set_title("🎨 Bar Chart of Categories")
        ax.set_xlabel("Category")
        ax.set_ylabel("Frequency")
    else:
        ax.pie(freq_df["Frequency"], labels=freq_df["Category"], autopct="%1.1f%%", colors=plt.cm.Pastel1.colors)
        ax.set_title("🥧 Pie Chart of Categories")

    st.pyplot(fig)

# ==========================================================
# Summary Statistics & Boxplot
# ==========================================================

def run_summary(df_uploaded=None):
    st.subheader("📊 Summary Statistics & Boxplot")

    input_mode = st.radio("Data Input Mode:", ["Upload File", "Manual Entry"], horizontal=True)

    if input_mode == "Upload File":
        if df_uploaded is not None:
            numeric_cols = df_uploaded.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns found.")
                return
            col = st.selectbox("Select numeric column:", numeric_cols)
            data = df_uploaded[col].dropna().astype(float).values
        else:
            st.warning("Please upload a dataset first.")
            return
    else:
        raw_data = st.text_area("Enter comma-separated numeric values:",
                                "56, 57, 54, 61, 63, 58, 59, 62, 55, 57")
        data = np.array([float(x.strip()) for x in raw_data.split(",") if x.strip() != ""])

    stats_dict = get_summary_stats(data)
    df_summary = pd.DataFrame(stats_dict.items(), columns=["Statistic", "Value"])
    st.dataframe(df_summary, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, vert=False, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='black'),
               medianprops=dict(color='red'))
    ax.set_title("📦 Horizontal Boxplot")
    ax.set_xlabel("Values")
    st.pyplot(fig)

# ==========================================================
# Main App
# ==========================================================

def run():
    st.header("🧠 Descriptive Statistics Tool")

    categories = [
        "Qualitative (Categorical)",
        "Quantitative (Discrete or Continuous)",
        "Summary Statistics & Boxplot"
    ]

    choice = st.selectbox("Choose a category:", categories, index=None, placeholder="Select a category...")

    if not choice:
        st.info("👆 Please select a category to begin.")
        return

    uploaded_file = st.file_uploader("📂 Upload CSV or Excel file (optional):", type=["csv", "xlsx"])
    df_uploaded = None
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_uploaded = pd.read_csv(uploaded_file)
            else:
                df_uploaded = pd.read_excel(uploaded_file)
            st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

    if choice == "Qualitative (Categorical)":
        run_qualitative(df_uploaded)
    elif choice == "Quantitative (Discrete or Continuous)":
        run_quantitative(df_uploaded)
    elif choice == "Summary Statistics & Boxplot":
        run_summary(df_uploaded)

# ==========================================================
# Run App
# ==========================================================
if __name__ == "__main__":
    run()
