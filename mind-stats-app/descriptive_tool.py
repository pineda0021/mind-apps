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

def display_summary_table(datasets):
    """Display summary stats for multiple datasets side by side."""
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

    # ---------- QUALITATIVE ---------- #
    if choice == "Qualitative":
        st.subheader("📂 Category: Qualitative Data")

        if df_uploaded is not None:
            col = st.selectbox("Select column for analysis:", df_uploaded.columns)
            data = df_uploaded[col].dropna().astype(str).tolist()
        else:
            raw_data = st.text_area("Enter comma-separated values:", "")
            data = [val.strip() for val in raw_data.split(',') if val.strip() != ""]

        if data:
            df = display_frequency_table(data)
            st.dataframe(df)
            plot_qualitative(df)

    # ---------- DISCRETE ---------- #
    elif choice == "Quantitative (Discrete)":
        st.subheader("📂 Category: Quantitative (Discrete) Data")

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
        st.pyplot(plt)

    # ---------- CONTINUOUS ---------- #
    elif choice == "Quantitative (Continuous)":
        st.subheader("📂 Category: Quantitative (Continuous) Data")

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

        plt.hist(numeric_data, bins=10, edgecolor='black')
        st.pyplot(plt)

    # ---------- SUMMARY ---------- #
    elif choice == "Summary Statistics & Boxplot":
        st.subheader("📦 Summary Statistics & Boxplot Comparison")

        input_mode = st.radio("Choose input mode:", ["Manual Entry", "File Upload"])

        datasets = {}

        # Manual mode (1–5 datasets)
        if input_mode == "Manual Entry":
            num_datasets = st.selectbox(
                "How many datasets do you want to compare?",
                options=[1, 2, 3, 4, 5],
                index=0
            )
            for i in range(num_datasets):
                raw_data = st.text_area(f"Enter values for Dataset {i+1} (comma-separated):", key=f"data_{i}")
                if raw_data:
                    try:
                        numeric_data = np.array(list(map(float, raw_data.split(','))))
                        datasets[f"Dataset {i+1}"] = numeric_data
                    except ValueError:
                        st.error(f"Dataset {i+1}: Please enter valid numeric values.")

        # Upload mode (select multiple numeric columns)
        else:
            if df_uploaded is not None:
                numeric_cols = df_uploaded.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_cols = st.multiselect("Select columns to compare:", numeric_cols)
                    for col in selected_cols:
                        datasets[col] = df_uploaded[col].dropna().values
                else:
                    st.error("No numeric columns found in uploaded file.")
            else:
                st.warning("Please upload a dataset first.")

        # Display results
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


if __name__ == "__main__":
    run()
