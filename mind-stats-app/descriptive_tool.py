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
    mode = list(np.round(mode_result.mode, decimals))
    range_val = round(maximum - minimum, decimals)
    pop_var = round(np.var(data), decimals)
    pop_std = round(np.std(data), decimals)
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
        "Population Variance": pop_var,
        "Population Std Dev": pop_std,
        "Sample Variance": samp_var,
        "Sample Std Dev": samp_std
    }

def display_summary_streamlit(data):
    stats = get_summary_stats(data)
    st.markdown("### 📊 Five-Number Summary & IQR")
    for key in ["Minimum", "Q1", "Median", "Q3", "Maximum", "IQR"]:
        st.write(f"**{key}:** {stats[key]}")

    st.markdown("### 📈 Descriptive Statistics")
    st.write(f"**Mean:** {stats['Mean']}")
    st.write(f"**Mode:** {', '.join(map(str, stats['Mode']))}")
    st.write(f"**Range:** {stats['Range']}")
    st.write(f"**Population Variance:** {stats['Population Variance']}")
    st.write(f"**Population Std Dev:** {stats['Population Std Dev']}")
    st.write(f"**Sample Variance:** {stats['Sample Variance']}")
    st.write(f"**Sample Std Dev:** {stats['Sample Std Dev']}")

    st.markdown("### 🚨 Outlier Analysis")
    st.write(f"**Lower Bound:** {stats['Lower Bound']}")
    st.write(f"**Upper Bound:** {stats['Upper Bound']}")

    if stats["Outliers"]:
        st.warning(f"Potential outliers: {stats['Outliers']}")
    else:
        st.success("No potential outliers detected.")

def display_plotly_boxplot_streamlit(data):
    stats = get_summary_stats(data)
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
    if stats["Outliers"]:
        st.warning(f"Potential outliers: {stats['Outliers']}")
    else:
        st.success("No potential outliers detected.")

# ---------- Main App ----------

def run():
    st.header("📊 Descriptive Statistics Tool")
    st.write("Analyze qualitative, discrete, continuous data or get detailed summary statistics with boxplot.")

    choice = st.sidebar.radio(
        "Select Data Type:",
        [
            "Qualitative",
            "Quantitative (Discrete)",
            "Quantitative (Continuous)",
            "Summary Statistics & Boxplot"
        ]
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

    if raw_data:
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
                st.markdown("Enter class intervals as comma-separated ranges, e.g.: 0-2,3-5,6-8")
                class_interval_input = st.text_input("Class Intervals")

                if class_interval_input.strip():
                    try:
                        intervals = [item.strip() for item in class_interval_input.split(",") if item.strip()]
                        bins = []

                        for interval in intervals:
                            if "-" not in interval:
                                st.error(f"Invalid interval format: '{interval}'. Use format like 0-2.")
                                return
                            left_str, right_str = interval.split("-")
                            left, right = float(left_str), float(right_str)
                            if right < left:
                                st.error(f"Invalid interval: upper bound must be >= lower bound in '{interval}'")
                                return
                            bins.append((left, right))

                        bins = sorted(bins, key=lambda x: x[0])
                        bin_edges = [bins[0][0]]
                        for left, right in bins:
                            bin_edges.append(right)
                    except Exception as e:
                        st.error(f"Error parsing intervals: {e}")
                        return
                else:
                    default_bins = 5
                    bin_edges = np.histogram_bin_edges(numeric_data, bins=default_bins)

                df, bin_edges = group_continuous_data(numeric_data, bin_edges)
                st.subheader("Grouped Frequency Table (Continuous Data)")
                st.dataframe(df)
                plot_histograms(numeric_data, discrete=False, bins=bin_edges)

            except ValueError:
                st.error("Please enter valid numeric values for continuous data.")

        elif choice == "Summary Statistics & Boxplot":
            try:
                numeric_data = np.array(list(map(float, data)))
                display_summary_streamlit(numeric_data)
                display_plotly_boxplot_streamlit(numeric_data)
            except ValueError:
                st.error("Please enter valid numeric values for summary statistics.")

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

# ---------- Other Helper Functions ----------

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
        axes[0].hist(data, bins=range(min(data), max(data) + 2), edgecolor='black', color='lightblue')
        axes[0].set_title('Frequency Histogram')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')

        weights = np.ones_like(data) / len(data)
        axes[1].hist(data, bins=range(min(data), max(data) + 2), weights=weights, edgecolor='black', color='salmon')
        axes[1].set_title('Relative Frequency Histogram')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Relative Frequency')

    else:
        bins_adj = bins_adjusted_for_inclusivity(bins)
        axes[0].hist(data, bins=bins_adj, edgecolor='black', color='lightgreen')
        axes[0].set_title('Frequency Histogram')
        axes[0].set_xlabel('Class Intervals')
        axes[0].set_ylabel('Frequency')

        weights = np.ones_like(data) / len(data)
        axes[1].hist(data, bins=bins_adj, weights=weights, edgecolor='black', color='orange')
        axes[1].set_title('Relative Frequency Histogram')
        axes[1].set_xlabel('Class Intervals')
        axes[1].set_ylabel('Relative Frequency')

    plt.tight_layout()
    st.pyplot(fig)

def bins_adjusted_for_inclusivity(bin_edges):
    # Shift bins slightly to include upper bound in last bin
    eps = 1e-8
    adjusted = np.array(bin_edges, dtype=float)
    adjusted[:-1] = adjusted[:-1] - eps  # subtract tiny amount to make intervals closed on right
    return adjusted

def group_continuous_data(data, bin_edges):
    adjusted_edges = bins_adjusted_for_inclusivity(bin_edges)
    counts, _ = np.histogram(data, bins=adjusted_edges)
    rel_freq = np.round(counts / counts.sum(), 4)
    categories = [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]

    df = pd.DataFrame({
        'Class Interval': categories,
        'Frequency': counts,
        'Relative Frequency': rel_freq
    })
    return df, bin_edges

if __name__ == "__main__":
    run()
