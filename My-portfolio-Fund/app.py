import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# ======================================================
# PAGE SETUP
# ======================================================
st.set_page_config(page_title="WealthMonitor Pro", layout="wide")
st.title("📊 WealthMonitor Pro")
st.caption("Portfolio view with entry prices, gain/loss colors, and price alerts")

# ======================================================
# HELPERS
# ======================================================
def money(x):
    return f"${x:,.2f}"

def pct(x):
    return f"{x:,.2f}%"

def gain_color(val):
    if val > 0:
        return "color: green; font-weight: bold;"
    elif val < 0:
        return "color: red; font-weight: bold;"
    return "color: white;"

@st.cache_data(ttl=300)
def load_data(tickers, period="1mo", interval="1d"):
    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    return data

def extract_close_prices(data, tickers):
    if data.empty:
        return pd.DataFrame()

    # Case 1: Multiple tickers usually returns MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        # Expected shape: first level = price field, second level = ticker
        if "Close" in data.columns.get_level_values(0):
            close_df = data["Close"].copy()
        else:
            return pd.DataFrame()

    # Case 2: Single ticker usually returns regular columns
    else:
        if "Close" in data.columns:
            close_df = pd.DataFrame({tickers[0]: data["Close"]})
        else:
            return pd.DataFrame()

    # Keep requested order and only valid columns
    valid_cols = [t for t in tickers if t in close_df.columns]
    if not valid_cols:
        return pd.DataFrame()

    close_df = close_df[valid_cols].copy()
    close_df = close_df.dropna(how="all")
    return close_df

def style_gain_loss(df):
    return df.style.map(
        gain_color, subset=["Change ($)", "Change (%)"]
    ).format({
        "Entry Price": "${:,.2f}",
        "Current Price": "${:,.2f}",
        "Shares": "{:,.4f}",
        "Cost Basis": "${:,.2f}",
        "Market Value": "${:,.2f}",
        "Change ($)": "${:,.2f}",
        "Change (%)": "{:,.2f}%",
        "Alert Below": lambda x: "" if pd.isna(x) else f"${x:,.2f}",
        "Alert Above": lambda x: "" if pd.isna(x) else f"${x:,.2f}",
    })

# ======================================================
# SIDEBAR INPUTS
# ======================================================
with st.sidebar:
    st.header("Portfolio Settings")

    ticker_text = st.text_area(
        "Tickers (comma-separated)",
        value="FXAIX, FXNAX, VTSNX",
        help="Example: FXAIX, FXNAX, VTSNX"
    )

    tickers = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))  # remove duplicates, preserve order

    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    st.markdown("### Entry Prices")
    st.caption("Set your original buy price for each ticker.")

    default_entry_prices = {}
    for t in tickers:
        default_entry_prices[t] = 100.00

    entry_prices = {}
    shares_owned = {}
    alert_below = {}
    alert_above = {}

    for t in tickers:
        entry_prices[t] = st.number_input(
            f"{t} Entry Price",
            min_value=0.0,
            value=float(default_entry_prices[t]),
            step=0.01,
            format="%.2f",
            key=f"entry_{t}"
        )

        shares_owned[t] = st.number_input(
            f"{t} Shares",
            min_value=0.0,
            value=1.0,
            step=0.1,
            format="%.4f",
            key=f"shares_{t}"
        )

        alert_below[t] = st.number_input(
            f"{t} Alert Below",
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.2f",
            key=f"below_{t}",
            help="Set 0 to disable"
        )

        alert_above[t] = st.number_input(
            f"{t} Alert Above",
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.2f",
            key=f"above_{t}",
            help="Set 0 to disable"
        )

    show_raw = st.checkbox("Show raw downloaded data", value=False)

# ======================================================
# DOWNLOAD DATA
# ======================================================
data = load_data(tickers)
close_prices = extract_close_prices(data, tickers)

if close_prices.empty:
    st.error("No data available. Check the tickers, or the market data may be temporarily unavailable.")
    st.stop()

# ======================================================
# BUILD PORTFOLIO TABLE
# ======================================================
rows = []
alert_messages = []

for t in tickers:
    if t not in close_prices.columns:
        continue

    series = close_prices[t].dropna()
    if series.empty:
        continue

    current_price = float(series.iloc[-1])
    entry_price = float(entry_prices[t])
    shares = float(shares_owned[t])

    cost_basis = entry_price * shares
    market_value = current_price * shares
    change_dollar = market_value - cost_basis

    if cost_basis != 0:
        change_percent = ((market_value / cost_basis) - 1) * 100
    else:
        change_percent = 0.0

    below_val = float(alert_below[t]) if float(alert_below[t]) > 0 else float("nan")
    above_val = float(alert_above[t]) if float(alert_above[t]) > 0 else float("nan")

    if pd.notna(below_val) and current_price <= below_val:
        alert_messages.append(f"🔻 {t} is at {money(current_price)}, which is at or below your alert level of {money(below_val)}.")

    if pd.notna(above_val) and current_price >= above_val:
        alert_messages.append(f"🔺 {t} is at {money(current_price)}, which is at or above your alert level of {money(above_val)}.")

    rows.append({
        "Ticker": t,
        "Entry Price": entry_price,
        "Current Price": current_price,
        "Shares": shares,
        "Cost Basis": cost_basis,
        "Market Value": market_value,
        "Change ($)": change_dollar,
        "Change (%)": change_percent,
        "Alert Below": below_val,
        "Alert Above": above_val,
    })

portfolio_df = pd.DataFrame(rows)

if portfolio_df.empty:
    st.error("No valid ticker data was returned.")
    st.stop()

# ======================================================
# TOP SUMMARY
# ======================================================
total_cost = portfolio_df["Cost Basis"].sum()
total_value = portfolio_df["Market Value"].sum()
total_change = total_value - total_cost
total_change_pct = (total_change / total_cost * 100) if total_cost != 0 else 0.0

best_row = portfolio_df.loc[portfolio_df["Change (%)"].idxmax()]
worst_row = portfolio_df.loc[portfolio_df["Change (%)"].idxmin()]

st.subheader("Portfolio Snapshot")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Cost Basis", money(total_cost))
c2.metric("Total Market Value", money(total_value), delta=money(total_change))
c3.metric("Portfolio Return", pct(total_change_pct))
c4.metric("Tickers Tracked", str(len(portfolio_df)))

c5, c6 = st.columns(2)
c5.info(f"Best performer: **{best_row['Ticker']}** ({pct(best_row['Change (%)'])})")
c6.info(f"Worst performer: **{worst_row['Ticker']}** ({pct(worst_row['Change (%)'])})")

# ======================================================
# ALERTS
# ======================================================
st.subheader("Alerts")

if alert_messages:
    for msg in alert_messages:
        st.warning(msg)
else:
    st.success("No alerts triggered right now.")

st.caption("This version includes on-screen alerts. Email alerts are not included here because Gmail sending usually fails on Streamlit Cloud unless secrets and SMTP configuration are set correctly.")

# ======================================================
# PORTFOLIO TABLE
# ======================================================
st.subheader("Portfolio Table")
st.dataframe(
    style_gain_loss(portfolio_df),
    use_container_width=True,
    hide_index=True
)

# ======================================================
# CHART
# ======================================================
st.subheader("Price History")

fig = go.Figure()

for t in tickers:
    if t in close_prices.columns:
        fig.add_trace(
            go.Scatter(
                x=close_prices.index,
                y=close_prices[t],
                mode="lines",
                name=t
            )
        )

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Adjusted Close Price",
    margin=dict(l=20, r=20, t=40, b=20),
    legend_title="Ticker",
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# ======================================================
# ENTRY PRICE STORAGE VIEW
# ======================================================
st.subheader("Stored Entry Prices")

ENTRY_PRICES = {row["Ticker"]: row["Entry Price"] for _, row in portfolio_df.iterrows()}
st.write(ENTRY_PRICES)

# ======================================================
# RAW DATA
# ======================================================
if show_raw:
    st.subheader("Raw Close Price Data")
    st.dataframe(close_prices, use_container_width=True)
