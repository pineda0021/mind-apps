# confidence_intervals_tool.py
import math
import numpy as np
import streamlit as st
from scipy import stats

# ---------- helpers ----------
def parse_number_list(text: str):
    """Parse a comma/space/newline-separated list of numbers into a numpy array."""
    if not text.strip():
        return np.array([])
    # split on commas/newlines/spaces
    parts = [p for chunk in text.replace(",", " ").splitlines() for p in chunk.split(" ")]
    vals = []
    for p in parts:
        p = p.strip()
        if p:
            vals.append(float(p))
    return np.array(vals, dtype=float)

def z_from_conf_level(conf_level):
    """Two-sided Z* from confidence level, e.g., 0.95 -> 1.96"""
    alpha = 1 - conf_level
    return stats.norm.ppf(1 - alpha / 2)

def t_from_conf_level(conf_level, df):
    """Two-sided t* from confidence level and df."""
    alpha = 1 - conf_level
    return stats.t.ppf(1 - alpha / 2, df)

def chi2_bounds_from_conf(conf_level, df):
    """Return (chi2_upper, chi2_lower) criticals for variance CI
       so that lower = (df*s2)/chi2_upper, upper = (df*s2)/chi2_lower."""
    alpha = 1 - conf_level
    chi2_lower = stats.chi2.ppf(alpha / 2, df)     # small quantile
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df) # large quantile
    return chi2_upper, chi2_lower  # note the order for convenient formula

def pretty_interval(low, high, dec):
    return f"[{low:.{dec}f}, {high:.{dec}f}]"

# ---------- calculators ----------
def ci_proportion(conf_level, x, n, decimals):
    if n <= 0 or x < 0 or x > n:
        raise ValueError("n must be > 0 and 0 ≤ x ≤ n.")
    phat = x / n
    z = z_from_conf_level(conf_level)
    se = math.sqrt(phat * (1 - phat) / n)
    moe = z * se
    low = phat - moe
    high = phat + moe
    return {
        "point": phat,
        "se": se,
        "moe": moe,
        "low": low,
        "high": high,
        "interval_str": pretty_interval(low, high, decimals)
    }

def ss_proportion(conf_level, margin, p_star=0.5):
    if not (0 < margin < 1):
        raise ValueError("Margin of error must be in (0, 1).")
    if not (0 <= p_star <= 1):
        raise ValueError("Anticipated proportion must be in [0, 1].")
    z = z_from_conf_level(conf_level)
    n = (z**2) * p_star * (1 - p_star) / (margin**2)
    return math.ceil(n)

def ci_mean_known_sigma(conf_level, xbar, sigma, n, decimals):
    if n <= 0 or sigma <= 0:
        raise ValueError("n must be > 0 and σ must be > 0.")
    z = z_from_conf_level(conf_level)
    se = sigma / math.sqrt(n)
    moe = z * se
    low, high = xbar - moe, xbar + moe
    return {
        "point": xbar,
        "se": se,
        "moe": moe,
        "low": low,
        "high": high,
        "interval_str": pretty_interval(low, high, decimals)
    }

def ci_mean_with_data(conf_level, data, decimals):
    data = np.asarray(data, dtype=float)
    n = data.size
    if n < 2:
        raise ValueError("Need at least 2 data points.")
    xbar = data.mean()
    s = data.std(ddof=1)
    tstar = t_from_conf_level(conf_level, n - 1)
    se = s / math.sqrt(n)
    moe = tstar * se
    low, high = xbar - moe, xbar + moe
    return {
        "n": n, "xbar": xbar, "s": s,
        "se": se, "moe": moe,
        "low": low, "high": high,
        "interval_str": pretty_interval(low, high, decimals)
    }

def ss_mean(conf_level, margin, sigma):
    if margin <= 0 or sigma <= 0:
        raise ValueError("Margin and σ must be > 0.")
    z = z_from_conf_level(conf_level)
    n = (z * sigma / margin)**2
    return math.ceil(n)

def ci_variance_summary(conf_level, n, s2, decimals):
    if n < 2 or s2 <= 0:
        raise ValueError("Need n ≥ 2 and sample variance s² > 0.")
    df = n - 1
    chi2_upper, chi2_lower = chi2_bounds_from_conf(conf_level, df)
    lower_var = (df * s2) / chi2_upper
    upper_var = (df * s2) / chi2_lower
    return {
        "df": df,
        "s2": s2,
        "low_var": lower_var,
        "upper_var": upper_var,
        "interval_str": pretty_interval(lower_var, upper_var, decimals)
    }

def ci_variance_with_data(conf_level, data, decimals):
    data = np.asarray(data, dtype=float)
    n = data.size
    if n < 2:
        raise ValueError("Need at least 2 data points.")
    s2 = data.var(ddof=1)
    out = ci_variance_summary(conf_level, n, s2, decimals)
    out.update({"n": n, "s": math.sqrt(s2)})
    return out

def ci_sd_from_variance_bounds(var_low, var_high, decimals):
    sd_low = math.sqrt(var_low)
    sd_high = math.sqrt(var_high)
    return sd_low, sd_high, pretty_interval(sd_low, sd_high, decimals)

# ---------- UI ----------
def run():
    st.header("📏 Confidence Intervals & Sample Sizes")

    categories = [
        "Confidence Interval for Proportion",
        "Sample Size for Proportion",
        "Confidence Interval for Mean (Known Standard Deviation)",
        "Confidence Interval for Mean (With Data)",
        "Sample Size for Mean",
        "Confidence Interval for Variance (Without Data)",
        "Confidence Interval for Variance (With Data)",
        "Confidence Interval for Standard Deviation (Without Data)",
        "Confidence Interval for Standard Deviation (With Data)",
    ]

    choice = st.sidebar.radio("Choose a category:", categories)
    st.markdown("---")

    # shared inputs
    conf_display = st.select_slider(
        "Confidence level",
        options=[0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.999],
        value=0.95,
        format_func=lambda x: f"{int(100*x)}%"
    )
    decimals = st.number_input("Decimal places", min_value=0, max_value=10, value=4, step=1)

    st.markdown("---")

    try:
        # 1) CI for Proportion
        if choice == categories[0]:
            c1, c2 = st.columns(2)
            with c1:
                x = st.number_input("Number of successes (x)", min_value=0, step=1, value=50)
            with c2:
                n = st.number_input("Sample size (n)", min_value=1, step=1, value=100)

            if st.button("Compute CI"):
                res = ci_proportion(conf_display, x, n, decimals)
                st.success(
                    f"p̂ = {res['point']:.{decimals}f}, SE = {res['se']:.{decimals}f}, "
                    f"MOE = {res['moe']:.{decimals}f}\n\n"
                    f"**{int(100*conf_display)}% CI for p:** {res['interval_str']}"
                )
                st.caption("Note: This uses the Wald interval. For small samples or extreme p̂, consider Wilson or Agresti–Coull.")

        # 2) Sample Size for Proportion
        elif choice == categories[1]:
            c1, c2 = st.columns(2)
            with c1:
                margin = st.number_input("Desired margin of error (E)", min_value=0.0001, max_value=0.5, value=0.05, step=0.01, format="%.4f")
            with c2:
                p_star = st.number_input("Anticipated proportion p* (use 0.5 if unsure)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            if st.button("Compute n"):
                n_req = ss_proportion(conf_display, margin, p_star)
                st.success(f"**Required sample size:** n = {n_req}")

        # 3) CI for Mean (Known σ)
        elif choice == categories[2]:
            c1, c2, c3 = st.columns(3)
            with c1:
                xbar = st.number_input("Sample mean (x̄)", value=0.0, format="%.6f")
            with c2:
                sigma = st.number_input("Population σ (known)", min_value=0.0000001, value=1.0, format="%.6f")
            with c3:
                n = st.number_input("Sample size (n)", min_value=1, step=1, value=30)
            if st.button("Compute CI"):
                res = ci_mean_known_sigma(conf_display, xbar, sigma, n, decimals)
                st.success(
                    f"x̄ = {res['point']:.{decimals}f}, SE = {res['se']:.{decimals}f}, "
                    f"MOE = {res['moe']:.{decimals}f}\n\n"
                    f"**{int(100*conf_display)}% CI for μ:** {res['interval_str']}"
                )

        # 4) CI for Mean (With Data)
        elif choice == categories[3]:
            st.write("Paste or type data values (comma, space, or newline separated).")
            data_text = st.text_area("Data", value="12, 11, 14, 10, 13, 12")
            if st.button("Compute CI"):
                data = parse_number_list(data_text)
                res = ci_mean_with_data(conf_display, data, decimals)
                st.success(
                    f"n = {res['n']}, x̄ = {res['xbar']:.{decimals}f}, s = {res['s']:.{decimals}f}\n"
                    f"SE = {res['se']:.{decimals}f}, MOE = {res['moe']:.{decimals}f}\n\n"
                    f"**{int(100*conf_display)}% CI for μ:** {res['interval_str']}"
                )

        # 5) Sample Size for Mean
        elif choice == categories[4]:
            c1, c2 = st.columns(2)
            with c1:
                margin = st.number_input("Desired margin of error (E)", min_value=0.000001, value=0.1, format="%.6f")
            with c2:
                sigma = st.number_input("Population σ (or best estimate)", min_value=0.000001, value=1.0, format="%.6f")
            if st.button("Compute n"):
                n_req = ss_mean(conf_display, margin, sigma)
                st.success(f"**Required sample size:** n = {n_req}")

        # 6) CI for Variance (Without Data)
        elif choice == categories[5]:
            c1, c2 = st.columns(2)
            with c1:
                n = st.number_input("Sample size (n)", min_value=2, step=1, value=20)
            with c2:
                s = st.number_input("Sample standard deviation (s)", min_value=0.000001, value=2.0, format="%.6f")
            if st.button("Compute CI for Variance"):
                s2 = s**2
                res = ci_variance_summary(conf_display, n, s2, decimals)
                st.success(
                    f"df = {res['df']}, s² = {res['s2']:.{decimals}f}\n\n"
                    f"**{int(100*conf_display)}% CI for variance (σ²):** {res['interval_str']}"
                )

        # 7) CI for Variance (With Data)
        elif choice == categories[6]:
            st.write("Paste or type data values (comma, space, or newline separated).")
            data_text = st.text_area("Data", value="12, 11, 14, 10, 13, 12, 15, 11")
            if st.button("Compute CI for Variance"):
                data = parse_number_list(data_text)
                res = ci_variance_with_data(conf_display, data, decimals)
                st.success(
                    f"n = {res['n']}, s = {res['s']:.{decimals}f}, s² = {res['s2']:.{decimals}f}\n\n"
                    f"**{int(100*conf_display)}% CI for variance (σ²):** {res['interval_str']}"
                )

        # 8) CI for Standard Deviation (Without Data)
        elif choice == categories[7]:
            c1, c2 = st.columns(2)
            with c1:
                n = st.number_input("Sample size (n)", min_value=2, step=1, value=20)
            with c2:
                s = st.number_input("Sample standard deviation (s)", min_value=0.000001, value=2.0, format="%.6f")
            if st.button("Compute CI for SD"):
                s2 = s**2
                var_res = ci_variance_summary(conf_display, n, s2, decimals)
                sd_low, sd_high, sd_str = ci_sd_from_variance_bounds(var_res["low_var"], var_res["upper_var"], decimals)
                st.success(
                    f"df = {var_res['df']}, s = {s:.{decimals}f}\n\n"
                    f"**{int(100*conf_display)}% CI for standard deviation (σ):** {sd_str}"
                )

        # 9) CI for Standard Deviation (With Data)
        elif choice == categories[8]:
            st.write("Paste or type data values (comma, space, or newline separated).")
            data_text = st.text_area("Data", value="12, 11, 14, 10, 13, 12, 15, 11, 16")
            if st.button("Compute CI for SD"):
                data = parse_number_list(data_text)
                var_res = ci_variance_with_data(conf_display, data, decimals)
                sd_low, sd_high, sd_str = ci_sd_from_variance_bounds(var_res["low_var"], var_res["upper_var"], decimals)
                st.success(
                    f"n = {var_res['n']}, s = {var_res['s']:.{decimals}f}, s² = {var_res['s2']:.{decimals}f}\n\n"
                    f"**{int(100*conf_display)}% CI for standard deviation (σ):** {sd_str}"
                )

    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    run()
