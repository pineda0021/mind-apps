import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# ======================================================
# PAGE SETUP
# ======================================================
st.set_page_config(page_title="WealthMonitor Pro", layout="wide")
st.title("WealthMonitor Pro")

# ======================================================
# SAFE IMPORT CHECK
# ======================================================
try:
    import yfinance as yf
    import pandas as pd
    import plotly.graph_objects as go
except ModuleNotFoundError as e:
    st.error(f"Setup Error: {e}. Streamlit has not installed your requirements.txt yet.")
    st.stop()

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    ticker = st.text_input("Ticker", value="VTI").upper()

# ======================================================
# DATA FETCH (FAST + CACHED)
# ======================================================
@st.cache_data(ttl=3600)
def get_data(ticker):
    data = yf.download(
        ticker,
        period="1mo",
        interval="1d",
        auto_adjust=True,
        progress=False
    )
    return data

# ======================================================
# FETCH DATA
# ======================================================
with st.spinner("Loading market data..."):
    data = get_data(ticker)

# ======================================================
# HANDLE EMPTY DATA
# ======================================================
if data.empty:
    st.warning("No data available right now. The market may be closed, or the ticker may not support this interval.")
    st.stop()

# ======================================================
# FIX MULTIINDEX COLUMNS IF NEEDED
# ======================================================
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# ======================================================
# CURRENT METRICS
# ======================================================
current_price = float(data["Close"].iloc[-1])

if len(data) > 1:
    previous_price = float(data["Close"].iloc[-2])
    change_pct = ((current_price / previous_price) - 1) * 100
else:
    previous_price = current_price
    change_pct = 0.0

col1, col2 = st.columns(2)
col1.metric(f"{ticker} Current Price", f"${current_price:,.2f}")
col2.metric("Daily Change", f"{change_pct:.2f}%")

# ======================================================
# PRICE CHART
# ======================================================
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Close"],
        mode="lines",
        name=ticker
    )
)

fig.update_layout(
    title=f"{ticker} Price History",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ======================================================
# DATA TABLE
# ======================================================
st.subheader("Recent Data")
st.dataframe(data.tail(), use_container_width=True)
