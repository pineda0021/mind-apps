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


def display_plotly_boxplot_comparison(datasets):
    fig = go.Figure()
    for name, data in datasets.items():
        fig.add_trace(go.Box(
            y=data,
            name=name,
            boxpoints='outliers',
            marker=dict(color='teal'),
            line=dict(color='black')
        ))
    fig.update_layout(
        title="📦 Boxplot Comparison",
        yaxis_title="Values",
        template="simple_white"
    )
    st.plotly_chart(fig, use_container_width=True)


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
    raw_data = ""

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
    else:
        df_uploaded = None

    # ---------- SUMMARY ---------- #
    if choice == "Summary Statistics & Boxplot":
        st.subheader("📦 Summary Statistics & Boxplot Comparison")

        mode = st.radio("Select Mode:", ["Single Dataset", "Compare Multiple Datasets"])

        if mode == "Single Dataset":
            raw_data = st.text_area("Enter comma-separated values:", "")
            if raw_data:
                try:
                    numeric_data = np.array(list(map(float, raw_data.split(','))))
                    display_summary_streamlit(numeric_data)
                    display_plotly_boxplot_comparison({"Dataset 1": numeric_data})
                except ValueError:
                    st.error("Please enter valid numeric values.")
        else:
            st.info("💡 Select two or more numeric columns from your uploaded file to compare.")
            if df_uploaded is not None:
                numeric_cols = df_uploaded.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_cols = st.multiselect("Select columns to compare:", numeric_cols)
                    if len(selected_cols) >= 2:
                        datasets = {col: df_uploaded[col].dropna().values for col in selected_cols}
                        display_plotly_boxplot_comparison(datasets)
                    else:
                        st.warning("Please select at least two numeric columns.")
                else:
                    st.error("No numeric columns found in uploaded file.")
            else:
                st.warning("Please upload a dataset with multiple numeric columns to compare.")

    # ---------- QUALITATIVE ---------- #
    elif choice == "Qualitative":
        st.subheader("📂 Category: Qualitative Data")
        raw_data = st.text_area("Enter comma-separated values:", "")
        if raw_data:
            data = [val.strip() for val in raw_data.split(',') if val.strip() != ""]
            df = display_frequency_table(data)
            st.dataframe(df)
            plot_qualitative(df)

    # ---------- DISCRETE ---------- #
    elif choice == "Quantitative (Discrete)":
        st.subheader("📂 Category: Quantitative (Discrete) Data")
        raw_data = st.text_area("Enter comma-separated integers:", "")
        if raw_data:
            try:
                numeric_data = list(map(int, raw_data.split(',')))
                df = display_frequency_table(numeric_data)
                st.dataframe(df)
                st.markdown("### 📊 Frequency Histogram")
                plt.hist(numeric_data, bins=range(min(numeric_data), max(numeric_data)+2), edgecolor='black')
                st.pyplot(plt)
            except ValueError:
                st.error("Please enter valid integers for discrete data.")

    # ---------- CONTINUOUS ---------- #
    elif choice == "Quantitative (Continuous)":
        st.subheader("📂 Category: Quantitative (Continuous) Data")
        raw_data = st.text_area("Enter comma-separated numeric values:", "")
        if raw_data:
            try:
                numeric_data = list(map(float, raw_data.split(',')))
                st.markdown("### 📊 Continuous Data Histogram")
                plt.hist(numeric_data, bins=10, edgecolor='black')
                st.pyplot(plt)
            except ValueError:
                st.error("Please enter valid numeric values.")


if __name__ == "__main__":
    run()
