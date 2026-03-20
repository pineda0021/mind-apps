import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import smtplib
from email.message import EmailMessage

import streamlit as st

# Test imports at the very top
try:
    import yfinance as yf
    import pandas as pd
    import plotly.graph_objects as go
except ModuleNotFoundError as e:
    st.error(f"Setup Error: {e}. Streamlit hasn't installed your requirements.txt yet.")
    st.stop()

st.title("WealthMonitor Pro")

# --- USER SETTINGS ---
with st.sidebar:
    ticker = st.text_input("Ticker", value="VBAIX").upper()
    user_email = st.text_input("Gmail")
    app_pw = st.text_input("App Password", type="password")

# --- DATA FETCH ---
data = yf.download(ticker, period="5d", interval="1m")

if not data.empty:
    # Handle the yfinance MultiIndex column issue
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    current_price = float(data['Close'].iloc[-1])
    st.metric(f"{ticker} Current Price", f"${current_price:.2f}")
    
    # Simple Plot
    fig = go.Figure(go.Scatter(x=data.index, y=data['Close']))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig)
else:
    st.warning("Market is closed or ticker is invalid. No data to show.")
