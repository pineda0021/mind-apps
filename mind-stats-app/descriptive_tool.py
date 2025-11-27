import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import re

# ==========================================================
# UNIVERSAL VISIBILITY STYLE (Google-proof)
# ==========================================================

def apply_universal_style(ax):
    """
    Ensures all plots remain readable in:
    - Google Classroom dark mode
    - iPhone/Android dark mode
    - Gmail dark mode
    - Google Docs dark mode
    - Streamlit light/dark mode
    """
    background = "#2B2B2B"    # Neutral dark gray
    text = "white"            # High contrast white text

    ax.set_facecolor(background)
    ax.tick_params(colors=text)
    ax.xaxis.label.set_color(text)
    ax.yaxis.label.set_color(text)
    ax.title.set_color(text)

    for spine in ax.spines.values():
        spine.set_color(text)

    return background, text


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

    # ----------------- INPUT -----------------
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

    # ====================================================
    # DISCRETE
    # ====================================================

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

        # --------- UNIVERSAL VISIBILITY HISTOGRAM ----------
        fig, ax = plt.subplots(figsize=(8, 4))
        apply_universal_style(ax)

        ax.hist(
            data,
            bins=np.arange(min(data) - 0.5, max(data) + 1.5, 1),
            edgecolor="white",
            color="#6AA5FF",
            linewidth=1.2
        )

        ax.set_title("📊 Discrete Histogram (No Gaps)")
        ax.set_xlabel("Values")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)
        return

    # ====================================================
    # CONTINUOUS
    # ====================================================

    # Auto intervals (Sturges' Rule)
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
    intervals = parse_intervals(st.text_area(
        "Class Intervals:", default_intervals_text
    )) if edit else auto_intervals

    df_freq, total = compute_frequency_table(data, intervals)

    st.markdown("### Frequency Table")
    st.dataframe(df_freq, use_container_width=True)

    # ---------------- PLOTS -----------------
    plot_option = st.radio(
        "Choose visualization:",
        ["Histogram", "Histogram + Ogive", "Boxplot"],
        horizontal=True
    )

    bins = [low for low, _ in intervals] + [intervals[-1][1]]

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    apply_universal_style(ax)

    if plot_option in ["Histogram", "Histogram + Ogive"]:
        ax.hist(
            data,
            bins=bins,
            edgecolor="white",
            color="#6AA5FF",
            linewidth=1.2
        )
        ax.set_title("📊 Continuous Histogram (No Gaps)")
        ax.set_xlabel("Class Intervals")
        ax.set_ylabel("Frequency")

        if plot_option == "Histogram + Ogive":
            cum_freq = df_freq["Cumulative Freq"].values
            upper_bounds = [high for _, high in intervals]
            ax2 = ax.twinx()
            ax2.plot(
                upper_bounds,
                cum_freq,
                marker="o",
                color="#FFDD55",
                linewidth=2.5
            )
            ax2.tick_params(colors="white")
            ax2.yaxis.label.set_color("white")

        st.pyplot(fig)

    # Boxplot
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        apply_universal_style(ax)

        ax.boxplot(
            data,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="#444444", color="white"),
            medianprops=dict(color="#FFDD55", linewidth=2),
            whiskerprops=dict(color="white"),
            capprops=dict(color="white"),
            flierprops=dict(markeredgecolor="white")
        )
        ax.set_title("📦 Boxplot")
        ax.set_xlabel("Values")
        st.pyplot(fig)


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

    # Bar chart
    if chart_type == "Bar Chart":
        fig, ax = plt.subplots(figsize=(8, 4))
        apply_universal_style(ax)

        ax.bar(counts.index, counts.values, color="#6AA5FF", edgecolor="white")
        ax.set_title("🎨 Bar Chart")
        ax.set_xlabel("Category")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)

    # Pie chart
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        apply_universal_style(ax)

        colors = ["#6AA5FF", "#FF6A6A", "#FFD966", "#8AFF8A", "#FFB6FF", "#A0A0FF"]
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=colors,
            textprops={'color': "white"}
        )
        ax.set_title("🥧 Pie Chart")

        st.pyplot(fig)


# ==========================================================
# SUMMARY STATS + BOX PLOTS
# ==========================================================

def run_summary(df_uploaded=None):
    st.subheader("📊 Summary Statistics & Boxplots")

    mode = st.radio("Mode:", ["Single Dataset", "Multiple Datasets"], horizontal=True)

    # ---------------------------- SINGLE DATASET --------------------------
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
            data = np.array([float(x.strip()) for x in raw.split(",") if x.strip()])

        summary = pd.DataFrame(get_summary_stats(data).items(), columns=["Statistic", "Value"])
        st.dataframe(summary, use_container_width=True)

        # Boxplot
        fig, ax = plt.subplots(figsize=(8, 4))
        apply_universal_style(ax)

        ax.boxplot(
            data,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="#444444", color="white"),
            medianprops=dict(color="#FFDD55", linewidth=2)
        )
        ax.set_title("📦 Boxplot")
        ax.set_xlabel("Values")
        st.pyplot(fig)

    # ---------------------------- MULTIPLE DATASETS --------------------------
    else:
        input_mode = st.radio("Input:", ["Upload File", "Manual Entry"], horizontal=True)
        data_dict = {}

        if input_mode == "Upload File":
            if df_uploaded is None:
                st.warning("Upload first.")
                return

            numeric_cols = df_uploaded.select_dtypes(include=[np.number]).columns
            selected = st.multiselect("Select numeric columns:", numeric_cols, default=list(numeric_cols)[:2])

            for col in selected:
                data_dict[col] = df_uploaded[col].dropna().astype(float).values

        else:
            raw = st.text_area(
                "Enter datasets separated by semicolons:",
                "56,57,54; 49,51,55; 65,64,68"
            )
            blocks = [b.strip() for b in raw.split(";") if b.strip()]
            for i, block in enumerate(blocks, 1):
                data_dict[f"Dataset {i}"] = np.array([float(x.strip()) for x in block.split(",") if x.strip()])

        # Summary Table
        combined = pd.DataFrame()
        for name, d in data_dict.items():
            stats_dict = get_summary_stats(d)
            df_stats = pd.DataFrame(stats_dict, index=[name])
            combined = pd.concat([combined, df_stats])

        st.dataframe(combined, use_container_width=True)

        # Combined Boxplot
        fig, ax = plt.subplots(figsize=(8, 4 + 0.3 * len(data_dict)))
        apply_universal_style(ax)

        ax.boxplot(
            [d for d in data_dict.values()],
            labels=data_dict.keys(),
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="#444444", color="white"),
            medianprops=dict(color="#FFDD55", linewidth=2)
        )
        ax.set_title("📦 Boxplots (Multiple)")
        ax.set_xlabel("Values")
        st.pyplot(fig)


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
    elif choice == "Summary Statistics & Boxplot":
        run_summary(df_uploaded)


# ==========================================================
# RUN APP
# ==========================================================
if __name__ == "__main__":
    run()
