import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

TICKERS = {"FXAIX":0.2,"FXNAX":0.6,"VTIAX":0.2}

data = yf.download(list(TICKERS.keys()), period="1y")["Adj Close"]
returns = data.pct_change().dropna()

weights = np.array(list(TICKERS.values()))
portfolio = (returns @ weights).cumsum()

st.title("📊 Portfolio Dashboard")

# Price chart
fig = go.Figure()
for t in data.columns:
    fig.add_trace(go.Scatter(x=data.index, y=data[t], name=t))
st.plotly_chart(fig)

# Portfolio performance
st.subheader("Portfolio Performance")
st.line_chart(portfolio)

# Metrics
latest = returns.iloc[-1]
daily_return = (latest @ weights)
vol = returns.std().mean()*np.sqrt(252)

st.metric("Daily Return", f"{daily_return*100:.2f}%")
st.metric("Volatility", f"{vol*100:.2f}%")
