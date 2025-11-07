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
# Discrete Data Analyzer
# ==========================================================

def run_discrete(df_uploaded=None):
    st.subheader("🎯 Quantitative (Discrete) Data Analyzer")

    st.markdown("""
    This tool analyzes **discrete (countable) data** by generating a frequency table  
    and visualizing it with a bar chart.
    """)

    input_mode = st.radio("Data Input Mode:", ["Upload File", "Manual Entry"], horizontal=True)

    if input_mode == "Upload File":
        if df_uploaded is not None:
            numeric_cols = df_uploaded.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns found.")
                return
            col = st.selectbox("Select numeric column:", numeric_cols)
            data = df_uploaded[col].dropna().astype(int).values
        else:
            st.warning("Please upload a dataset first.")
            return
    else:
        raw_data = st.text_area("Enter comma-separated integer values:",
                                "3, 5, 4, 4, 3, 2, 5, 5, 3, 4, 2, 5, 1, 4, 3, 3, 2, 5, 4, 3")
        try:
            data = np.array([int(x.strip()) for x in raw_data.split(",") if x.strip() != ""])
        except ValueError:
            st.error("❌ Invalid input. Please enter integers separated by commas.")
            return

    # ---------- Frequency Table ----------
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

    # ---------- Visualization ----------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(freq_df["Value"], freq_df["Frequency"], color="skyblue", edgecolor="black")
    ax.set_title("🎨 Discrete Data Bar Chart")
    ax.set_xlabel("Data Values")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# ==========================================================
# Continuous Data Analyzer
# ==========================================================

def run_continuous(df_uploaded=None, mode="Manual Entry"):
    st.subheader("📂 Quantitative (Continuous) Data Analyzer")

    st.markdown("""
    This tool constructs **class intervals and frequency tables** from continuous data.  
    You can also adjust intervals manually or enter frequencies yourself.
    """)

    # ---------- Data Input ----------
    numeric_data = None

    if mode == "Upload File":
        if df_uploaded is not None:
            numeric_cols = df_uploaded.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Select numeric column:", numeric_cols)
                numeric_data = df_uploaded[col].dropna().astype(float).values
                st.success(f"✅ Loaded {len(numeric_data)} observations from '{col}'")
            else:
                st.error("No numeric columns found.")
                return
        else:
            st.warning("Please upload a dataset first.")
            return

    else:  # Manual Entry
        raw_data = st.text_area(
            "Enter comma-separated numeric values:",
            "728,730,726,698,721,722,700,720,729,678,722,716,702,703,718,703,723,699,703,713,672,711,695,731,726,695,718"
        )
        try:
            numeric_data = np.array([float(x.strip()) for x in raw_data.split(",") if x.strip() != ""])
            st.success(f"✅ Loaded {len(numeric_data)} observations.")
        except ValueError:
            st.error("❌ Invalid numeric input.")
            return

    # ---------- Automatic Class Intervals ----------
    st.markdown("### 🧮 Automatically Generated Class Intervals")

    min_val = np.min(numeric_data)
    max_val = np.max(numeric_data)
    n = len(numeric_data)

    # Use Sturges' Rule
    k = int(np.ceil(1 + 3.322 * np.log10(n)))
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

    # ---------- Frequency Input ----------
    freq_mode = st.radio("Would you like to enter frequencies manually?",
                         ["No (calculate automatically)", "Yes (enter manually)"], horizontal=True)
    manual_freq = None

    if freq_mode == "Yes (enter manually)":
        freq_input = st.text_area("Enter frequencies separated by commas:", "")
        try:
            manual_freq = [int(x.strip()) for x in freq_input.split(",")]
            if len(manual_freq) != len(intervals):
                st.error("⚠️ Number of frequencies must match number of intervals.")
                return
        except ValueError:
            st.error("❌ Invalid frequency input.")
            return

    df_freq, total = compute_frequency_table(numeric_data, intervals, manual_freq)
    st.markdown("### 📋 Frequency Distribution Table")
    st.dataframe(df_freq, use_container_width=True)
    st.markdown(f"**Total Frequency (n):** {total}")

    # ---------- Visualization ----------
    plot_option = st.radio("Choose visualization:",
                           ["Histogram", "Histogram + Cumulative Frequency Polygon (Ogive)", "Boxplot"],
                           horizontal=True)
    bins = [low for low, _ in intervals] + [intervals[-1][1]]
    fig, ax = plt.subplots(figsize=(8, 4))

    if plot_option in ["Histogram", "Histogram + Cumulative Frequency Polygon (Ogive)"]:
        ax.hist(numeric_data, bins=bins, edgecolor="black", color="skyblue")
        ax.set_title("📊 Frequency Histogram")
        ax.set_xlabel("Class Intervals")
        ax.set_ylabel("Frequency")
        ax.set_xticks(bins)
        if plot_option == "Histogram + Cumulative Frequency Polygon (Ogive)":
            cum_freq = df_freq["Cumulative Freq"].values
            upper_bounds = [high for _, high in intervals]
            ax2 = ax.twinx()
            ax2.plot(upper_bounds, cum_freq, marker="o", color="darkred", linewidth=2, label="Cumulative Frequency")
            ax2.set_ylabel("Cumulative Frequency", color="darkred")
            ax2.legend(loc="upper left")

    elif plot_option == "Boxplot":
        ax.boxplot(numeric_data, vert=False, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='black'),
                   medianprops=dict(color='red'))
        ax.set_title("📦 Horizontal Boxplot")
        ax.set_xlabel("Values")
        ax.set_yticks([])

    st.pyplot(fig)

# ==========================================================
# Summary Statistics & Boxplot
# ==========================================================

def run_summary(df_uploaded=None):
    st.subheader("📊 Summary Statistics & Boxplot")

    st.markdown("""
    Compute **key descriptive statistics** and visualize them with a horizontal boxplot.  
    Outliers are determined using the **IQR rule**.
    """)

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
        raw_data = st.text_area(
            "Enter comma-separated numeric values:",
            "56, 57, 54, 61, 63, 58, 59, 62, 55, 57, 61, 60, 59, 58, 64"
        )
        try:
            data = np.array([float(x.strip()) for x in raw_data.split(",") if x.strip() != ""])
        except ValueError:
            st.error("❌ Invalid numeric input.")
            return

    # ---------- Compute Stats ----------
    stats_dict = get_summary_stats(data)
    df_summary = pd.DataFrame(stats_dict.items(), columns=["Statistic", "Value"])
    st.dataframe(df_summary, use_container_width=True)

    # ---------- Boxplot ----------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, vert=False, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='black'),
               medianprops=dict(color='red'))
    ax.set_title("📦 Horizontal Boxplot with Outliers")
    ax.set_xlabel("Values")
    ax.set_yticks([])
    st.pyplot(fig)


# ==========================================================
# Main App
# ==========================================================

def run():
    st.header("📘 Descriptive Statistics Tool")

    categories = [
        "Quantitative (Discrete)",
        "Quantitative (Continuous) — Upload File",
        "Quantitative (Continuous) — Manual Entry",
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

    # ---------- ROUTING ----------
    if choice == "Quantitative (Discrete)":
        run_discrete(df_uploaded)
    elif "Quantitative (Continuous)" in choice:
        mode = "Upload File" if "Upload File" in choice else "Manual Entry"
        run_continuous(df_uploaded, mode)
    elif choice == "Summary Statistics & Boxplot":
        run_summary(df_uploaded)

# ==========================================================
# Run App
# ==========================================================
if __name__ == "__main__":
    run()
