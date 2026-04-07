import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# ------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------
st.set_page_config(page_title="WealthMonitor Pro", layout="wide")
st.title("📊 WealthMonitor Pro")

# ------------------------------------------------------
# USER INPUT
# ------------------------------------------------------
with st.sidebar:
    ticker = st.text_input("Ticker", value="VBAIX").upper()

# ------------------------------------------------------
# DATA FETCH FUNCTION (CACHED)
# ------------------------------------------------------
@st.cache_data
def load_data(ticker):
    data = yf.download(
        ticker,
        period="1mo",          # safer for mutual funds
        interval="1d",         # avoid 1m (causes failures)
        auto_adjust=True,
        progress=False
    )
    return data

data = load_data(ticker)

# ------------------------------------------------------
# HANDLE EMPTY DATA
# ------------------------------------------------------
if data.empty:
    st.error("No data available. The market may be closed or the ticker is invalid.")
    st.stop()

# ------------------------------------------------------
# FIX MULTIINDEX (yfinance quirk)
# ------------------------------------------------------
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# ------------------------------------------------------
# PRICE CALCULATIONS
# ------------------------------------------------------
current_price = float(data["Close"].iloc[-1])
entry_price = float(data["Close"].iloc[0])

change_dollar = current_price - entry_price
change_percent = ((current_price / entry_price) - 1) * 100

# ------------------------------------------------------
# METRICS DISPLAY
# ------------------------------------------------------
st.subheader("📈 Price Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Change ($)", f"${change_dollar:.2f}")
col3.metric("Change (%)", f"{change_percent:.2f}%")

# ------------------------------------------------------
# PLOT
# ------------------------------------------------------
st.subheader("📉 Price Chart")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data.index,
    y=data["Close"],
    mode="lines",
    name="Close Price"
))

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Price",
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------
# RAW DATA
# ------------------------------------------------------
with st.expander("🔍 Show Raw Data"):
    st.dataframe(data)
