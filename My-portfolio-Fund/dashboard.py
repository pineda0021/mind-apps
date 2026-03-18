import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

TICKERS = {"FXAIX":0.2,"FXNAX":0.6,"VTIAX":0.2}

st.set_page_config(layout="wide")
st.title("📊 Portfolio Dashboard")

# Load data

data = yf.download(list(TICKERS.keys()), period="1y")["Adj Close"]
returns = data.pct_change().dropna()

# Chart
fig = go.Figure()
for t in data.columns:
    fig.add_trace(go.Scatter(x=data.index, y=data[t], name=t))

st.plotly_chart(fig, use_container_width=True)

# Portfolio
weights = np.array(list(TICKERS.values()))
portfolio = (returns @ weights).cumsum()

st.subheader("Portfolio Performance")
st.line_chart(portfolio)

# Metrics
latest = returns.iloc[-1]
daily_return = (latest @ weights)
vol = returns.std().mean() * np.sqrt(252)

col1, col2 = st.columns(2)
col1.metric("Daily Return", f"{daily_return*100:.2f}%")
col2.metric("Volatility", f"{vol*100:.2f}%")
