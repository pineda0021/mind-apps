import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ======================================================
# PAGE SETUP
# ======================================================
st.set_page_config(page_title="WealthMonitor Pro", layout="wide")
st.title("WealthMonitor Pro")
st.caption("Market tracker and simple portfolio monitor")

# ======================================================
# DEFAULT PORTFOLIO
# ======================================================
DEFAULT_TICKERS = {
    "FXAIX": 0.20,
    "FXNAX": 0.60,
    "VTIAX": 0.20
}

# ======================================================
# HELPERS
# ======================================================
@st.cache_data(ttl=3600)
def fetch_single_ticker_data(ticker, period="6mo"):
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    return data

@st.cache_data(ttl=3600)
def fetch_portfolio_data(tickers, period="1y"):
    data = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    # yfinance often returns MultiIndex columns for multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close_data = data["Close"].copy()
        else:
            close_data = data.xs("Close", axis=1, level=0, drop_level=True).copy()
    else:
        # single ticker fallback
        close_data = pd.DataFrame(data["Close"])
        close_data.columns = tickers

    return close_data.dropna(how="all")

def compute_portfolio_metrics(price_data, weights_dict):
    weights = np.array([weights_dict[t] for t in price_data.columns], dtype=float)

    returns = price_data.pct_change().dropna()

    latest = price_data.iloc[-1]
    prev = price_data.iloc[-2]

    daily_returns = (latest / prev) - 1
    portfolio_return = float(np.dot(daily_returns.values, weights))
    portfolio_volatility = float(returns.dot(weights).std() * np.sqrt(252))

    cumulative = (1 + returns.dot(weights)).cumprod()

    return {
        "latest": latest,
        "daily_returns": daily_returns,
        "portfolio_return": portfolio_return,
        "portfolio_volatility": portfolio_volatility,
        "cumulative": cumulative
    }

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.header("Settings")

    single_ticker = st.text_input("Single ticker to view", value="VTI").upper()

    st.markdown("### Portfolio Weights")
    fxaix = st.number_input("FXAIX weight", min_value=0.0, max_value=1.0, value=0.20, step=0.05)
    fxnax = st.number_input("FXNAX weight", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
    vtiax = st.number_input("VTIAX weight", min_value=0.0, max_value=1.0, value=0.20, step=0.05)

    weight_sum = fxaix + fxnax + vtiax
    st.write(f"Weight sum: **{weight_sum:.2f}**")

portfolio_tickers = {
    "FXAIX": fxaix,
    "FXNAX": fxnax,
    "VTIAX": vtiax
}

# ======================================================
# VALIDATION
# ======================================================
if abs(weight_sum - 1.0) > 1e-9:
    st.error("Portfolio weights must add up to 1.00.")
    st.stop()

# ======================================================
# SINGLE TICKER SECTION
# ======================================================
st.header("Single Ticker View")

single_data = fetch_single_ticker_data(single_ticker, period="6mo")

if single_data.empty:
    st.warning(f"No data available for {single_ticker}. Check the ticker symbol.")
else:
    current_price = float(single_data["Close"].iloc[-1])
    prev_price = float(single_data["Close"].iloc[-2]) if len(single_data) > 1 else current_price
    change_pct = ((current_price / prev_price) - 1) * 100 if prev_price != 0 else 0.0

    col1, col2 = st.columns(2)
    col1.metric(f"{single_ticker} Current Price", f"${current_price:,.2f}")
    col2.metric("Daily Change", f"{change_pct:.2f}%")

    fig_single = go.Figure()
    fig_single.add_trace(
        go.Scatter(
            x=single_data.index,
            y=single_data["Close"],
            mode="lines",
            name=single_ticker
        )
    )
    fig_single.update_layout(
        title=f"{single_ticker} Price History",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=450
    )
    st.plotly_chart(fig_single, use_container_width=True)

# ======================================================
# PORTFOLIO SECTION
# ======================================================
st.header("Portfolio Monitor")

portfolio_prices = fetch_portfolio_data(list(portfolio_tickers.keys()), period="1y")

if portfolio_prices.empty or len(portfolio_prices) < 2:
    st.warning("Not enough portfolio data available to compute metrics.")
    st.stop()

try:
    results = compute_portfolio_metrics(portfolio_prices, portfolio_tickers)

    latest = results["latest"]
    daily = results["daily_returns"]
    port_ret = results["portfolio_return"]
    vol = results["portfolio_volatility"]
    cumulative = results["cumulative"]

    c1, c2 = st.columns(2)
    c1.metric("Portfolio Daily Return", f"{port_ret * 100:.2f}%")
    c2.metric("Annualized Volatility", f"{vol * 100:.2f}%")

    st.subheader("Latest Prices and Daily Returns")

    summary_df = pd.DataFrame({
        "Ticker": latest.index,
        "Latest Price": latest.values,
        "Daily Return (%)": daily.values * 100,
        "Weight": [portfolio_tickers[t] for t in latest.index]
    })

    summary_df["Latest Price"] = summary_df["Latest Price"].map(lambda x: round(float(x), 2))
    summary_df["Daily Return (%)"] = summary_df["Daily Return (%)"].map(lambda x: round(float(x), 2))
    summary_df["Weight"] = summary_df["Weight"].map(lambda x: round(float(x), 2))

    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Portfolio Growth Index")
    growth_df = pd.DataFrame({
        "Date": cumulative.index,
        "Growth": cumulative.values
    })

    fig_port = go.Figure()
    fig_port.add_trace(
        go.Scatter(
            x=growth_df["Date"],
            y=growth_df["Growth"],
            mode="lines",
            name="Portfolio"
        )
    )
    fig_port.update_layout(
        title="Portfolio Cumulative Growth",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        template="plotly_dark",
        height=450
    )
    st.plotly_chart(fig_port, use_container_width=True)

    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

except Exception as e:
    st.error(f"An error occurred while computing portfolio metrics: {e}")
