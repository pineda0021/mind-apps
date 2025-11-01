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

    # ---------- SUMMARY ---------- #
    if choice == "Summary Statistics & Boxplot":
        st.subheader("📦 Summary Statistics & Boxplot Comparison")

        num_datasets = st.selectbox(
            "How many datasets do you want to compare?",
            options=[1, 2, 3, 4, 5],
            index=0
        )

        datasets = {}
        for i in range(num_datasets):
            raw_data = st.text_area(f"Enter values for Dataset {i+1} (comma-separated):", key=f"data_{i}")
            if raw_data:
                try:
                    numeric_data = np.array(list(map(float, raw_data.split(','))))
                    datasets[f"Dataset {i+1}"] = numeric_data
                except ValueError:
                    st.error(f"Dataset {i+1}: Please enter valid numeric values.")

        if len(datasets) >= 1:
            if len(datasets) == 1:
                name, data = list(datasets.items())[0]
                st.markdown(f"### 📋 Summary for {name}")
                single_summary = get_summary_stats(data)
                for k, v in single_summary.items():
                    if k not in ["Outliers"]:
                        st.write(f"**{k}:** {v}")
                if single_summary["Outliers"]:
                    st.warning(f"Potential outliers: {single_summary['Outliers']}")
                else:
                    st.success("No potential outliers detected.")
                display_plotly_boxplot_comparison(datasets)
            else:
                st.success(f"✅ Comparing {len(datasets)} datasets.")
                display_summary_table(datasets)
                display_plotly_boxplot_comparison(datasets)

    # ---------- QUALITATIVE ---------- #
    elif choice == "Qualitative":
        st.subheader("📂 Category: Qualitative Data")
        raw_data = st.text_area("Enter comma-separated values:", "")
        if raw_data:
            data = [val.strip() for val in raw_data.split(',') if val.strip() != ""]
            freq = dict(Counter(data))
            total = sum(freq.values())
            df = pd.DataFrame({
                "Category": list(freq.keys()),
                "Frequency": list(freq.values()),
                "Relative Frequency": [round(v / total, 4) for v in freq.values()]
            })
            st.dataframe(df)

    # ---------- DISCRETE ---------- #
    elif choice == "Quantitative (Discrete)":
        st.subheader("📂 Category: Quantitative (Discrete) Data")
        raw_data = st.text_area("Enter comma-separated integers:", "")
        if raw_data:
            try:
                numeric_data = list(map(int, raw_data.split(',')))
                plt.hist(numeric_data, bins=range(min(numeric_data), max(numeric_data) + 2), edgecolor='black')
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
                plt.hist(numeric_data, bins=10, edgecolor='black')
                st.pyplot(plt)
            except ValueError:
                st.error("Please enter valid numeric values.")


if __name__ == "__main__":
    run()
