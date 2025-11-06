import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
import plotly.graph_objects as go

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

def display_frequency_table(data):
    freq = dict(Counter(data))
    total = sum(freq.values())
    rel_freq = {k: round(v / total, 4) for k, v in freq.items()}
    df = pd.DataFrame({
        'Category': list(freq.keys()),
        'Frequency': list(freq.values()),
        'Relative Frequency': list(rel_freq.values())
    })
    try:
        df['Category'] = pd.to_numeric(df['Category'])
        df = df.sort_values(by='Category')
    except:
        df = df.sort_values(by='Category')
    return df.reset_index(drop=True)

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

# ==========================================================
# Quantitative (Continuous) Enhanced
# ==========================================================

def parse_intervals(interval_text):
    """Parse intervals entered as [670,679], [680,689], etc."""
    intervals = []
    for part in interval_text.split(']'):
        if ',' in part:
            part = part.replace('[', '').replace(']', '').strip()
            try:
                low, high = map(float, part.split(','))
                if low < high:
                    intervals.append((low, high))
            except:
                pass
    return intervals

def compute_frequency_table(data, intervals, manual_freq=None):
    """Compute or accept manual frequencies."""
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

def run_continuous(df_uploaded=None):
    st.subheader("📂 Quantitative (Continuous) Data Analyzer")

    st.markdown("""
    Enter your continuous data and class intervals below.  
    You can upload a dataset or enter your own values manually.
    """)

    # ---------------- STEP 1: DATA INPUT ----------------
    input_mode = st.radio("Data Input Mode:", ["Upload File", "Manual Entry"], horizontal=True)
    numeric_data = None

    if input_mode == "Upload File":
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
            st.warning("Please upload a dataset using the file uploader above.")
            return
    else:
        raw_data = st.text_area("Enter comma-separated numeric values:", "670,678,690,700,710,720,729,730,735")
        try:
            numeric_data = np.array([float(x.strip()) for x in raw_data.split(",") if x.strip() != ""])
            st.success(f"✅ Loaded {len(numeric_data)} observations.")
        except ValueError:
            st.error("❌ Invalid numeric input. Please enter comma-separated numbers.")
            return

    # ---------------- STEP 2: CLASS INTERVALS ----------------
    st.markdown("### 🧩 Enter Class Intervals")
    st.info("Example: `[670,679], [680,689], [690,699], [700,709], [710,719], [720,729], [730,739]`")

    interval_text = st.text_area(
        "Enter class intervals:",
        "[670,679], [680,689], [690,699], [700,709], [710,719], [720,729], [730,739]"
    )
    intervals = parse_intervals(interval_text)

    if not intervals:
        st.warning("⚠️ Please enter valid intervals like `[670,679], [680,689]`.")
        return

    # ---------------- STEP 3: FREQUENCY OPTIONS ----------------
    freq_mode = st.radio("Would you like to enter frequencies manually?",
                         ["No (calculate automatically)", "Yes (enter manually)"], horizontal=True)

    manual_freq = None
    if freq_mode == "Yes (enter manually)":
        freq_input = st.text_area("Enter frequencies separated by commas (must match number of intervals):", "2,0,4,5,5,9,2")
        try:
            manual_freq = [int(x.strip()) for x in freq_input.split(",")]
            if len(manual_freq) != len(intervals):
                st.error("⚠️ Number of frequencies must match the number of class intervals.")
                return
        except ValueError:
            st.error("❌ Please enter valid integers for frequencies.")
            return

    # ---------------- STEP 4: COMPUTE TABLE ----------------
    df_freq, total = compute_frequency_table(numeric_data, intervals, manual_freq)
    st.markdown("### 📋 Frequency Distribution Table")
    st.dataframe(df_freq, use_container_width=True)
    st.markdown(f"**Total Frequency (n):** {total}")

    # ---------------- STEP 5: VISUALIZATION ----------------
    plot_option = st.radio("Choose visualization:",
                           ["Histogram", "Histogram + Cumulative Frequency Polygon (Ogive)"],
                           horizontal=True)

    bins = [low for low, _ in intervals] + [intervals[-1][1]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(numeric_data, bins=bins, edgecolor="black", color="skyblue")
    ax.set_title("📊 Frequency Histogram (User-defined Intervals)")
    ax.set_xlabel("Class Intervals")
    ax.set_ylabel("Frequency")
    ax.set_xticks(bins)

    # Ogive Option
    if plot_option == "Histogram + Cumulative Frequency Polygon (Ogive)":
        cum_freq = df_freq["Cumulative Freq"].values
        upper_bounds = [high for _, high in intervals]
        ax2 = ax.twinx()
        ax2.plot(upper_bounds, cum_freq, marker="o", color="darkred", linewidth=2, label="Cumulative Frequency")
        ax2.set_ylabel("Cumulative Frequency", color="darkred")
        ax2.tick_params(axis="y", labelcolor="darkred")
        ax2.legend(loc="upper left")

    st.pyplot(fig)

    # ---------------- STEP 6: INTERPRETATION ----------------
    st.markdown("### 🧭 Interpretation")
    st.write(
        f"The dataset contains **{total} observations** across **{len(intervals)} intervals**. "
        "Use this visualization to identify skewness or modality patterns in your distribution."
    )
    st.caption("Tip: Use manual frequency entry for classroom exercises where students construct frequency tables by hand.")


# ==========================================================
# Main App
# ==========================================================

def run():
    st.header("📘 Descriptive Statistics Tool")

    categories = [
        "Qualitative",
        "Quantitative (Discrete)",
        "Quantitative (Continuous)",
        "Summary Statistics & Boxplot"
    ]

    choice = st.selectbox("Choose a category:", categories, index=None, placeholder="Select a category to begin...")

    if not choice:
        st.info("👆 Please choose a category to begin.")
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
            st.dataframe(df_uploaded)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

    if choice == "Qualitative":
        st.subheader("📂 Qualitative Data")
        if df_uploaded is not None:
            col = st.selectbox("Select column for analysis:", df_uploaded.columns)
            data = df_uploaded[col].dropna().astype(str).tolist()
        else:
            raw_data = st.text_area("Enter comma-separated categories:", "")
            data = [val.strip() for val in raw_data.split(',') if val.strip()]
        if data:
            df = display_frequency_table(data)
            st.dataframe(df)
            plot_qualitative(df)

    elif choice == "Quantitative (Discrete)":
        st.subheader("📂 Quantitative (Discrete) Data")
        if df_uploaded is not None:
            numeric_cols = df_uploaded.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Select numeric column:", numeric_cols)
                numeric_data = df_uploaded[col].dropna().astype(int).values
            else:
                st.error("No numeric columns found.")
                return
        else:
            raw_data = st.text_area("Enter comma-separated integers:", "")
            try:
                numeric_data = np.array(list(map(int, raw_data.split(','))))
            except ValueError:
                st.error("Please enter valid integers.")
                return
        plt.hist(numeric_data, bins=range(min(numeric_data), max(numeric_data) + 2), edgecolor='black')
        plt.title("Discrete Data Histogram")
        st.pyplot(plt)

    elif choice == "Quantitative (Continuous)":
        run_continuous(df_uploaded)

    elif choice == "Summary Statistics & Boxplot":
        st.subheader("📦 Summary Statistics & Boxplot Comparison")
        input_mode = st.radio("Input Mode:", ["Manual Entry", "File Upload"])
        datasets = {}
        if input_mode == "Manual Entry":
            num_datasets = st.selectbox("How many datasets?", [1, 2, 3, 4, 5], index=0)
            for i in range(num_datasets):
                raw_data = st.text_area(f"Dataset {i+1} (comma-separated):", key=f"data_{i}")
                if raw_data:
                    try:
                        numeric_data = np.array(list(map(float, raw_data.split(','))))
                        datasets[f"Dataset {i+1}"] = numeric_data
                    except ValueError:
                        st.error(f"Dataset {i+1}: invalid numeric input.")
        else:
            if df_uploaded is not None:
                numeric_cols = df_uploaded.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_cols = st.multiselect("Select columns to compare:", numeric_cols)
                    for col in selected_cols:
                        datasets[col] = df_uploaded[col].dropna().values
                else:
                    st.error("No numeric columns found.")
            else:
                st.warning("Please upload a dataset first.")
        if len(datasets) >= 1:
            if len(datasets) == 1:
                name, data = list(datasets.items())[0]
                st.markdown(f"### 📋 Summary for {name}")
                s = get_summary_stats(data)
                for k, v in s.items():
                    if k != "Outliers":
                        st.write(f"**{k}:** {v}")
                if s["Outliers"]:
                    st.warning(f"Potential outliers: {s['Outliers']}")
                else:
                    st.success("No potential outliers detected.")
                display_plotly_boxplot_comparison(datasets)
            else:
                st.success(f"✅ Comparing {len(datasets)} datasets.")
                display_summary_table(datasets)
                display_plotly_boxplot_comparison(datasets)


# ==========================================================
# Run App
# ==========================================================
if __name__ == "__main__":
    run()
