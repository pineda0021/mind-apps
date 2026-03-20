import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import smtplib
from email.message import EmailMessage

# --- CONFIG ---
st.set_page_config(page_title="WealthMonitor Pro", layout="wide")

# Target weights from your main.py logic
TARGET_WEIGHTS = {"FXAIX": 0.2, "FXNAX": 0.6, "VTIAX": 0.2}

# --- FUNCTIONS ---
def send_email(subject, content, user_email, app_password):
    if not user_email or not app_password:
        st.error("Email settings missing in sidebar!")
        return
    msg = EmailMessage()
    msg.set_content(content)
    msg['Subject'] = subject
    msg['From'] = user_email
    msg['To'] = user_email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(user_email, app_password)
            smtp.send_message(msg)
        st.toast("Email sent!", icon="📧")
    except Exception as e:
        st.error(f"Error: {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Global Settings")
    user_email = st.text_input("Your Email")
    app_password = st.text_input("App Password", type="password")
    st.divider()
    st.info("Switch between 'Single Stock' and 'Portfolio' tabs below.")

# --- TABS ---
tab1, tab2 = st.tabs(["🎯 Real-Time Monitor", "⚖️ Portfolio Rebalance"])

# --- TAB 1: REAL-TIME MONITOR ---
with tab1:
    st.title("Single Stock Alerter")
    ticker = st.text_input("Enter Ticker", value="VBAIX").upper()
    
    col_a, col_b = st.columns(2)
    sell_target = col_a.number_input("Sell Target ($)", value=200.0)
    drop_alert = col_b.number_input("Drop Alert ($)", value=150.0)
    
    data = yf.download(ticker, period="1d", interval="1m")
    
    if not data.empty:
        curr = float(data['Close'].iloc[-1])
        st.metric(f"{ticker} Price", f"${curr:.2f}", f"{(curr - data['Open'].iloc[0]):.2f}")
        
        fig = go.Figure(go.Scatter(x=data.index, y=data['Close'], line=dict(color='#00FFCC')))
        fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Manual Alert Check"):
            if curr >= sell_target:
                send_email(f"SELL: {ticker}", f"{ticker} hit ${curr}", user_email, app_password)
            elif curr <= drop_alert:
                send_email(f"DROP: {ticker}", f"{ticker} dropped to ${curr}", user_email, app_password)
            else:
                st.write("Price is stable. No email sent.")

# --- TAB 2: PORTFOLIO REBALANCE ---
with tab2:
    st.title("Portfolio Health (Annual View)")
    
    # Fetch data for all portfolio tickers
    port_data = yf.download(list(TARGET_WEIGHTS.keys()), period="1y")["Adj Close"]
    
    if not port_data.empty:
        latest = port_data.iloc[-1]
        returns = port_data.pct_change().dropna()
        weights = np.array(list(TARGET_WEIGHTS.values()))
        
        # Calculations
        vol = (returns.dot(weights).std() * np.sqrt(252)) * 100
        
        st.subheader("Current Allocations")
        cols = st.columns(len(TARGET_WEIGHTS))
        for i, (t, w) in enumerate(TARGET_WEIGHTS.items()):
            cols[i].metric(t, f"${latest[t]:.2f}", f"Target: {w*100:.0f}%")
        
        st.divider()
        st.write(f"**Annualized Volatility:** {vol:.2f}%")
        
        # Visualizing Drift (Simplified version of your main.py logic)
        drift_fig = go.Figure(data=[
            go.Bar(name='Target', x=list(TARGET_WEIGHTS.keys()), y=[w*100 for w in TARGET_WEIGHTS.values()]),
            # In a real app, you'd calculate 'Current' based on share counts
        ])
        drift_fig.update_layout(title="Allocation Strategy", template="plotly_dark")
        st.plotly_chart(drift_fig)
