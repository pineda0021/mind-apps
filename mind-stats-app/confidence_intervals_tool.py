# =====================================================
# 5. Confidence Interval for Mean (With Raw Data)
# =====================================================
elif choice == categories[4]:
    data = load_uploaded_data()
    raw_input = st.text_area("Or enter comma-separated values:")
    confidence_level = st.number_input("Confidence level", value=0.95, format="%.3f")

    # Build data from raw input if no file uploaded
    if data is None and raw_input:
        try:
            data = np.array([float(x.strip()) for x in raw_input.split(",") if x.strip() != ""])
        except:
            st.error("❌ Invalid data input. Please check your entries.")
            data = None

    if st.button("👨‍💻 Calculate"):
        if data is None or len(data) == 0:
            st.warning("⚠️ Please provide data via file upload or manual entry.")
        elif len(data) < 2:
            st.error("❌ Need at least 2 observations to compute a t-interval (sample SD).")
        else:
            n = len(data)
            mean = float(np.mean(data))
            sd = float(np.std(data, ddof=1))
            df = n - 1
            t_crit = float(stats.t.ppf((1 + confidence_level)/2, df=df))
            se = sd / np.sqrt(n)
            moe = t_crit * se
            lower, upper = mean - moe, mean + moe

            st.text(f"""
=====================
Confidence Interval for Mean (With Data)
=====================
Sample size (n) = {n}
Sample mean = {mean:.{decimal}f}
Sample SD (s) = {sd:.{decimal}f}
Degrees of freedom = {df}
Critical Value (t) = {t_crit:.{decimal}f}
Standard Error = {se:.{decimal}f}
{confidence_level*100:.1f}% CI = ({lower:.{decimal}f}, {upper:.{decimal}f})
""")

            # Optional histogram with CI shading
            fig, ax = plt.subplots()
            ax.hist(data, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
            ax.axvline(lower, linestyle="--", label="Lower CI")
            ax.axvline(upper, linestyle="--", label="Upper CI")
            ax.axvspan(lower, upper, alpha=0.2)
            ax.set_title("Histogram with Confidence Interval")
            ax.legend()
            st.pyplot(fig)


