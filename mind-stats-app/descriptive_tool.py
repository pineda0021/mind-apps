import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _parse_numbers(text: str) -> np.ndarray:
    # Accept commas, spaces, and newlines
    raw = text.replace("\n", ",").replace(" ", ",")
    parts = [p for p in raw.split(",") if p.strip() != ""]
    return np.array([float(x) for x in parts], dtype=float)

def _parse_intervals(text: str):
    # "0-2,3-5,6-8" -> [(0,2),(3,5),(6,8)]
    pairs = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        a, b = token.split("-")
        a, b = float(a.strip()), float(b.strip())
        if b < a:
            a, b = b, a
        pairs.append((a, b))
    return pairs

def _five_number_summary(x: np.ndarray):
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return float(np.min(x)), float(q1), float(q2), float(q3), float(np.max(x))

def _iqr_bounds(x: np.ndarray):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr
    return float(lb), float(ub), float(iqr)

def run():
    st.header("📊 Descriptive Statistics")

    with st.expander("📥 Data Input", expanded=True):
        left, right = st.columns([2, 1])
        with left:
            data_input = st.text_area(
                "Enter your dataset (comma, space, or newline separated):",
                "1, 1, 1, 2, 2, 4, 4, 4, 5, 5, 5, 6, 6, 6, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17",
                height=120
            )
        with right:
            st.markdown("**Class Intervals (inclusive on both ends)**")
            intervals_input = st.text_input(
                "Format: start-end,start-end,...",
                "0-2,3-5,6-8,9-11,12-14,15-17"
            )

    # Parse data
    try:
        data = _parse_numbers(data_input)
    except Exception:
        st.error("⚠ Please enter valid numeric data.")
        return

    if data.size == 0:
        st.warning("Please provide at least one value.")
        return

    st.write(f"**Sample size (n):** {len(data)}")

    # Summary statistics
    st.subheader("📈 Summary Statistics")
    mean = float(np.mean(data))
    median = float(np.median(data))
    mode_vals = pd.Series(data).mode().tolist()
    data_min, data_q1, data_q2, data_q3, data_max = _five_number_summary(data)
    samp_var = float(np.var(data, ddof=1)) if len(data) > 1 else 0.0
    samp_std = float(np.sqrt(samp_var))

    stats_df = pd.DataFrame({
        "Statistic": [
            "Mean", "Median", "Mode", "Minimum", "Q1", "Q2 (Median)",
            "Q3", "Maximum", "Range", "Sample Variance", "Sample Std Dev"
        ],
        "Value": [
            mean, median, ", ".join(f"{m:g}" for m in mode_vals),
            data_min, data_q1, data_q2, data_q3, data_max,
            data_max - data_min, samp_var, samp_std
        ]
    })
    st.dataframe(stats_df, use_container_width=True)

    # Boxplot & Outliers
    lb, ub, iqr = _iqr_bounds(data)
    outliers = [float(val) for val in data if val < lb or val > ub]

    st.subheader("🧰 Boxplot & Outlier Detection")
    fig_box, ax_box = plt.subplots(figsize=(6, 1.8))
    ax_box.boxplot(data, vert=False)
    ax_box.set_yticks([])
    ax_box.set_title("Boxplot")
    st.pyplot(fig_box, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Lower Bound", f"{lb:g}")
    c2.metric("Upper Bound", f"{ub:g}")
    c3.metric("IQR", f"{iqr:g}")
    st.write(f"**Potential Outliers:** {outliers if outliers else 'None'}")

    # Frequency table from intervals
    st.subheader("📊 Grouped Frequency Table (Custom Intervals)")
    try:
        intervals = _parse_intervals(intervals_input)
    except Exception:
        st.error("⚠ Invalid interval format. Use e.g. 0-2,3-5,6-8")
        return

    rows = []
    n = len(data)
    for a, b in intervals:
        # inclusive on both ends
        count = int(np.sum((data >= a) & (data <= b)))
        rows.append({
            "Interval": f"{a:g}-{b:g}",
            "Frequency": count,
            "Relative Frequency": round(count / n, 4),
            "Cumulative Frequency": None,  # fill next
            "Cumulative Rel. Freq": None
        })

    if rows:
        # cumulative columns
        freq = np.array([r["Frequency"] for r in rows], dtype=int)
        cumf = np.cumsum(freq)
        cumrf = np.round(cumf / n, 4)
        for i, r in enumerate(rows):
            r["Cumulative Frequency"] = int(cumf[i])
            r["Cumulative Rel. Freq"] = float(cumrf[i])

    freq_df = pd.DataFrame(rows)
    st.dataframe(freq_df, use_container_width=True)

    # Frequency histogram (bars by interval label)
    st.subheader("📉 Histograms")
    left_h, right_h = st.columns(2)

    with left_h:
        fig_f, ax_f = plt.subplots(figsize=(6, 3))
        ax_f.bar(freq_df["Interval"], freq_df["Frequency"])
        ax_f.set_title("Frequency Histogram (Grouped)")
        ax_f.set_xlabel("Interval")
        ax_f.set_ylabel("Frequency")
        plt.setp(ax_f.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig_f, use_container_width=True)

    with right_h:
        fig_rf, ax_rf = plt.subplots(figsize=(6, 3))
        ax_rf.bar(freq_df["Interval"], freq_df["Relative Frequency"])
        ax_rf.set_title("Relative Frequency Histogram (Grouped)")
        ax_rf.set_xlabel("Interval")
        ax_rf.set_ylabel("Relative Frequency")
        plt.setp(ax_rf.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig_rf, use_container_width=True)

    st.caption("Intervals are **inclusive on both ends** (e.g., 0–2 counts values equal to 0, 1, and 2).")
