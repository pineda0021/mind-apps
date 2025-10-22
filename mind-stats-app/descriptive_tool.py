import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import io
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
    if mode_count <= 1:
        mode = "No mode"
    else:
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
        "Outliers": outliers,
        "Mean": mean,
        "Mode": mode,
        "Range": range_val,
        "σ² (Population Variance)": pop_var,
        "σ (Population Std Dev)": pop_std,
        "s² (Sample Variance)": samp_var,
        "s (Sample Std Dev)": samp_std
    }

def display_summary_streamlit(data):
    stats_summary = get_summary_stats(data)
    st.markdown("### 📊 Five-Number Summary & IQR")
    for key in ["Minimum", "Q1", "Median", "Q3", "Maximum", "IQR"]:
        st.write(f"**{key}:** {stats_summary[key]}")

    st.markdown("### 📈 Descriptive Statistics")
    st.write(f"**Mean:** {stats_summary['Mean']}")
    st.write(f"**Mode:** {stats_summary['Mode']}")
    st.write(f"**Range:** {stats_summary['Range']}")
    st.write(f"**Population Variance (σ²):** {stats_summary['σ² (Population Variance)']}")
    st.write(f"**Population Std Dev (σ):** {stats_summary['σ (Population Std Dev)']}")
    st.write(f"**Sample Variance (s²):** {stats_summary['s² (Sample Variance)']}")
    st.write(f"**Sample Std Dev (s):** {stats_summary['s (Sample Std Dev)']}")

    st.markdown("### 🚨 Outlier Analysis")
    st.write(f"**Lower Bound:** {stats_summary['Lower Bound']}")
    st.write(f"**Upper Bound:** {stats_summary['Upper Bound']}")

    if stats_summary["Outliers"]:
        st.warning(f"Potential outliers: {stats_summary['Outliers']}")
    else:
        st.success("No potential outliers detected.")

def display_plotly_boxplot_streamlit(data):
    stats_summary = get_summary_stats(data)
    fig = go.Figure()

    fig.add_trace(go.Box(
        x=data,
        boxpoints='outliers',
        orientation='h',
        marker=dict(color='red'),
        line=dict(color='black'),
        fillcolor='lightblue',
        name='Boxplot'
    ))

    fig.update_layout(
        title="Interactive Boxplot for the Dataset",
        xaxis_title="Values",
        yaxis=dict(showticklabels=False),
        showlegend=False,
        template="simple_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🚨 Outlier Analysis")
    if stats_summary["Outliers"]:
        st.warning(f"Potential outliers: {stats_summary['Outliers']}")
    else:
        st.success("No potential outliers detected.")

# ---------- Continuous grouping & plotting ----------

def parse_intervals(interval_string):
    intervals = []
    parts = [p.strip() for p in interval_string.split(",") if p.strip()]
    for p in parts:
        if "-" not in p:
            raise ValueError(f"Invalid interval '{p}' (expected left-right with '-')")
        left_s, right_s = p.split("-", 1)
        left = float(left_s.strip())
        right = float(right_s.strip())
        if right < left:
            raise ValueError(f"Interval upper bound < lower bound in '{p}'")
        intervals.append((left, right))
    intervals = sorted(intervals, key=lambda x: x[0])
    return intervals

def group_continuous_data_explicit_counts(data, intervals):
    data_arr = np.array(data, dtype=float)
    n = len(data_arr)
    categories, counts, rel_freqs = [], [], []

    for i, (left, right) in enumerate(intervals):
        adjacent_to_next = False
        if i < len(intervals) - 1:
            next_left = intervals[i + 1][0]
            adjacent_to_next = np.isclose(right, next_left, atol=1e-8)

        if adjacent_to_next:
            mask = (data_arr >= left) & (data_arr < right)
            label = f"{left:.0f} ≤ x < {right:.0f}" if left.is_integer() and right.is_integer() else f"{left:.2f} ≤ x < {right:.2f}"
        else:
            mask = (data_arr >= left) & (data_arr <= right)
            label = f"{left:.0f} ≤ x ≤ {right:.0f}" if left.is_integer() and right.is_integer() else f"{left:.2f} ≤ x ≤ {right:.2f}"

        cnt = int(np.sum(mask))
        categories.append(label)
        counts.append(cnt)
        rel_freqs.append(round(cnt / n, 6) if n > 0 else 0.0)

    df = pd.DataFrame({
        "Class Interval": categories,
        "Frequency": counts,
        "Relative Frequency": rel_freqs
    })
    return df

def plot_histogram_from_intervals(data, intervals):
    data = np.array(data, dtype=float)
    bin_edges = [intervals[0][0]] + [right for _, right in intervals]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(data, bins=bin_edges, edgecolor='black', align='mid')
    axes[0].set_title("Frequency Histogram")
    axes[0].set_xlabel("Class Interval")
    axes[0].set_ylabel("Frequency")

    weights = np.ones_like(data) / len(data)
    axes[1].hist(data, bins=bin_edges, weights=weights, edgecolor='black', align='mid')
    axes[1].set_title("Relative Frequency Histogram")
    axes[1].set_xlabel("Class Interval")
    axes[1].set_ylabel("Relative Frequency")

    plt.tight_layout()
    st.pyplot(fig)

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

def plot_histograms(data, discrete=True, bins=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if discrete:
        axes[0].hist(data, bins=range(min(data), max(data) + 2), edgecolor='black')
        axes[0].set_title('Frequency Histogram')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')

        weights = np.ones_like(data) / len(data)
        axes[1].hist(data, bins=range(min(data), max(data) + 2), weights=weights, edgecolor='black')
        axes[1].set_title('Relative Frequency Histogram')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Relative Frequency')

    else:
        if bins is not None:
            axes[0].hist(data, bins=bins, edgecolor='black')
            axes[0].set_title('Frequency Histogram (visual)')
            axes[0].set_xlabel('Value')
            axes[0].set_ylabel('Frequency')

            weights = np.ones_like(data) / len(data)
            axes[1].hist(data, bins=bins, weights=weights, edgecolor='black')
            axes[1].set_title('Relative Frequency Histogram (visual)')
            axes[1].set_xlabel('Value')
            axes[1].set_ylabel('Relative Frequency')
        else:
            axes[0].text(0.5, 0.5, 'No histogram available', ha='center')
            axes[1].text(0.5, 0.5, 'No histogram available', ha='center')

    plt.tight_layout()
    st.pyplot(fig)

# ---------- Main App ----------

def run():
    st.title("📊 Descriptive Statistics Analyzer")

    # Dropdown with search bar (like Confidence Interval Calculator)
    st.markdown("### 🧭 Choose a Category:")
    choice = st.selectbox(
        "Choose a category:",
        [
            "Qualitative",
            "Quantitative (Discrete)",
            "Quantitative (Continuous)",
            "Summary Statistics & Boxplot"
        ],
        index=None,
        placeholder="Select a category to begin..."
    )

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
            raw_data = df_uploaded.iloc[:, 0].dropna().astype(str).tolist()
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        raw_data = st.text_area("Or enter comma-separated values:", "")

    if not choice:
        st.info("👆 Please choose a category from the dropdown to get started.")
        return

    if raw_data:
        if isinstance(raw_data, str):
            data = [val.strip() for val in raw_data.split(',') if val.strip() != ""]
        else:
            data = raw_data

        df = None

        # ---------- QUALITATIVE ----------
        if choice == "Qualitative":
            st.subheader("📂 Category: Qualitative Data")
            df = display_frequency_table(data)
            st.markdown("### Frequency and Relative Frequency Table")
            st.dataframe(df)
            plot_qualitative(df)

        # ---------- DISCRETE ----------
        elif choice == "Quantitative (Discrete)":
            st.subheader("📂 Category: Quantitative (Discrete) Data")
            try:
                numeric_data = list(map(int, data))
                df = display_frequency_table(numeric_data)
                st.markdown("### Frequency and Relative Frequency Table")
                st.dataframe(df)
                plot_histograms(numeric_data, discrete=True)
            except ValueError:
                st.error("Please enter valid integers for discrete quantitative data.")

        # ---------- CONTINUOUS ----------
        elif choice == "Quantitative (Continuous)":
            st.subheader("📂 Category: Quantitative (Continuous) Data")
            try:
                numeric_data = list(map(float, data))
                st.markdown("### 📌 Select or Enter Class Intervals")
                interval_choice = st.selectbox(
                    "Choose a preset interval grouping or enter custom intervals below:",
                    ["Custom", "0-2,3-5,6-8,9-11,12-14,15-17", "0-5,5-10,10-15"]
                )

                class_interval_input = ""
                if interval_choice == "Custom":
                    class_interval_input = st.text_input("Enter class intervals (e.g. 0-2,3-5,6-8)")
                else:
                    class_interval_input = interval_choice

                if class_interval_input.strip():
                    try:
                        intervals = parse_intervals(class_interval_input)
                    except Exception as e:
                        st.error(f"Error parsing intervals: {e}")
                        return
                else:
                    default_bins = 5
                    bin_edges = np.histogram_bin_edges(numeric_data, bins=default_bins)
                    intervals = [(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges)-1)]

                df = group_continuous_data_explicit_counts(numeric_data, intervals)
                st.markdown("### Grouped Frequency Table (Continuous Data)")
                st.dataframe(df)
                plot_histogram_from_intervals(numeric_data, intervals)

            except ValueError:
                st.error("Please enter valid numeric values for continuous data.")

        # ---------- SUMMARY STATISTICS ----------
        elif choice == "Summary Statistics & Boxplot":
            st.subheader("📂 Category: Summary Statistics & Boxplot")
            try:
                numeric_data = np.array(list(map(float, data)))
                display_summary_streamlit(numeric_data)
                display_plotly_boxplot_streamlit(numeric_data)
            except ValueError:
                st.error("Please enter valid numeric values for summary statistics.")

        # ---------- DOWNLOAD ----------
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

if __name__ == "__main__":
    run()
