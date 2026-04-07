import streamlit as st
import pandas as pd

st.set_page_config(page_title="Portfolio Tracker", layout="wide")
st.title("📊 Portfolio Tracker")
st.caption("Editable version")

def money(x):
    return f"${x:,.2f}"

def gain_color(x):
    if x > 0:
        return "green"
    elif x < 0:
        return "red"
    return "black"

def compute_df(df):
    df = df.copy()
    df["Buy"] = pd.to_numeric(df["Buy"], errors="coerce")
    df["Have"] = pd.to_numeric(df["Have"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Total"] = df["Buy"] * df["Have"]
    df["Today"] = df["Close"] * df["Have"]
    return df

def section_summary(df):
    buy_total = df["Total"].sum()
    today_total = df["Today"].sum()
    diff = today_total - buy_total
    return buy_total, today_total, diff

def show_section(title, df, key, invest_target=None):
    st.markdown(f"## {title}")

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key=key
    )

    edited = compute_df(edited)
    st.dataframe(edited, use_container_width=True)

    buy_total, today_total, diff = section_summary(edited)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total", money(buy_total))
    c2.metric("Today", money(today_total), delta=f"{diff:,.2f}")
    c3.markdown(
        f"**Gain / Loss:** "
        f"<span style='color:{gain_color(diff)}'>{money(diff)}</span>",
        unsafe_allow_html=True
    )

    if invest_target is not None:
        gap = invest_target - today_total
        st.markdown(
            f"**Invest Target:** {money(invest_target)} &nbsp;&nbsp;&nbsp; "
            f"**Difference to Target:** "
            f"<span style='color:{gain_color(gap)}'>{money(gap)}</span>",
            unsafe_allow_html=True
        )

    return edited, buy_total, today_total

occ_df = pd.DataFrame([
    {"Ticker": "VBAIX", "Buy": 50.92, "Have": 23.0230, "Close": 50.17},
])

csuf_df = pd.DataFrame([
    {"Ticker": "FXAIX", "Buy": 226.59, "Have": 25.6787, "Close": 229.32},
    {"Ticker": "FXNAX", "Buy": 10.42, "Have": 1694.75, "Close": 10.49},
    {"Ticker": "VTIFX", "Buy": 28.70, "Have": 204.75, "Close": 28.76},
])

cypress_df = pd.DataFrame([
    {"Ticker": "FXAIX", "Buy": 226.59, "Have": 91.5840, "Close": 229.32},
    {"Ticker": "FXNAX", "Buy": 10.42, "Have": 2686.407, "Close": 10.49},
    {"Ticker": "VTSNX", "Buy": 160.93, "Have": 127.2164, "Close": 166.38},
])

lacc_df = pd.DataFrame([
    {"Ticker": "FXAIX", "Buy": 226.59, "Have": 51.5224, "Close": 229.32},
    {"Ticker": "FXNAX", "Buy": 10.42, "Have": 3402.867, "Close": 10.49},
    {"Ticker": "VTSNX", "Buy": 160.93, "Have": 71.5681, "Close": 166.38},
])

st.sidebar.header("Targets")
occ_target = st.sidebar.number_input("OCC Invest Target", value=1184.77, step=1.0)
cypress_target = st.sidebar.number_input("Cypress Invest Target", value=100031.60, step=1.0)

occ_df, occ_buy, occ_today = show_section("OCC", occ_df, "occ", invest_target=occ_target)
st.markdown("---")

csuf_df, csuf_buy, csuf_today = show_section("CSUF", csuf_df, "csuf")
st.markdown("---")

cypress_df, cypress_buy, cypress_today = show_section("Cypress", cypress_df, "cypress", invest_target=cypress_target)

grant_buy = csuf_buy + cypress_buy
grant_today = csuf_today + cypress_today
grant_diff = grant_today - grant_buy

st.subheader("Grant Summary")
g1, g2, g3 = st.columns(3)
g1.metric("Grant Total", money(grant_buy))
g2.metric("Grant Today", money(grant_today), delta=f"{grant_diff:,.2f}")
g3.markdown(
    f"**Gain / Loss:** "
    f"<span style='color:{gain_color(grant_diff)}'>{money(grant_diff)}</span>",
    unsafe_allow_html=True
)

st.markdown("---")

lacc_df, lacc_buy, lacc_today = show_section("LACC", lacc_df, "lacc")

overall_buy = occ_buy + csuf_buy + cypress_buy + lacc_buy
overall_today = occ_today + csuf_today + cypress_today + lacc_today
overall_diff = overall_today - overall_buy

st.markdown("---")
st.header("Overall Summary")

o1, o2, o3 = st.columns(3)
o1.metric("Portfolio Total", money(overall_buy))
o2.metric("Portfolio Today", money(overall_today), delta=f"{overall_diff:,.2f}")
o3.markdown(
    f"**Overall Gain / Loss:** "
    f"<span style='color:{gain_color(overall_diff)}'>{money(overall_diff)}</span>",
    unsafe_allow_html=True
)
