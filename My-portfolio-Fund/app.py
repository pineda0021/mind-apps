import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import smtplib
from email.message import EmailMessage

# --- CONFIG & TARGETS ---
# Your portfolio strategy from main.py
TARGET_WEIGHTS = {"FXAIX": 0.2, "FXNAX": 0.6, "VTIAX": 0.2}

st.set_page_config(page_title="WealthMonitor Pro", layout="wide")

# --- HELPER FUNCTIONS ---
def send_email(subject, content, user_email, app_password):
    if not user_email or not app_password:
        st.error("Please enter Email and App Password in the sidebar.")
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
        st.toast("Alert email sent!", icon="📧")
    except Exception as e:
        st.error(f"Email Error: {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.header("👤 Account Settings")
    user_email = st.text_input("Gmail Address")
    app_password = st.text_input("Gmail App Password", type="password")
    st.caption("Tip: Use Google 'App Passwords' for security.")
    st.divider()
    st.info("This app monitors real-time prices and portfolio drift.")

# --- APP TABS ---
tab1, tab2 = st.tabs(["🎯 Real-Time Monitor", "⚖️ Portfolio Health"])

# --- TAB 1: SINGLE STOCK MONITOR ---
with tab1:
    st.title("Real-Time Ticker Alert")
    ticker_input = st.text_input("Enter Ticker to Watch", value="VBAIX").upper()
    
    c1, c2 = st.columns(2)
    sell_price = c1.number_input("Sell Target Price ($)", value=200.0)
    drop_price = c2.number_input("Email if Drops Below ($)", value=150.0)
    
    # Fetch 1-minute data for the current day
    live_data = yf.download(ticker_input, period="1d", interval="1m")
    
    if not live_data.empty:
        current_val = float(live_data['Close'].iloc[-1])
        open_val = float(live_data['Open'].iloc[0])
        change = current_val - open_val
        
        st.metric(f"{ticker_input} Current", f"${current_val:.2f}", f"{change:.2f}")
        
        # Plotly Chart
        fig = go.Figure(go.Scatter(x=live_data.index, y=live_data['Close'], line=dict(color='#00FFCC')))
        fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,b=0,t=20))
        st.plotly_chart(fig, use_container_width=True)

        # Logic Check
        if st.button(f"Scan {ticker_input} Now"):
            if current_val >= sell_price:
                send_email(f"SELL ALERT: {ticker_input}", f"{ticker_input} hit target: ${current_val:.2f}", user_email, app_password)
            elif current_val <= drop_price:
                send_email(f"DROP ALERT: {ticker_input}", f"{ticker_input} dropped to: ${current_val:.2f}", user_email, app_password)
            else:
                st.info("Price is within range. No email sent.")

# --- TAB 2: PORTFOLIO REBALANCE (Old main.py logic) ---
with tab2:
    st.title("Diversified Portfolio Status")
    
    # Fetch 1y data for the fixed portfolio
    port_tickers = list(TARGET_WEIGHTS.keys())
    port_data = yf.download(port_tickers, period="1y")["Adj Close"]
    
    if not port_data.empty:
        latest_prices = port_data.iloc[-1]
        returns = port_data.pct_change().dropna()
        weights_arr = np.array(list(TARGET_WEIGHTS.values()))
        
        # Statistics
        volatility = (returns.dot(weights_arr).std() * np.sqrt(252)) * 100
        
        st.subheader("Current Asset Snapshot")
        cols = st.columns(len(TARGET_WEIGHTS))
        for i, (t, target_w) in enumerate(TARGET_WEIGHTS.items()):
            cols[i].metric(t, f"${latest_prices[t]:.2f}", f"Target {target_w*100:.0f}%")
        
        st.write(f"**Annualized Portfolio Volatility:** {volatility:.2f}%")

        # Visualizing Allocations
        fig_pie = go.Figure(data=[go.Pie(labels=list(TARGET_WEIGHTS.keys()), 
                                        values=list(TARGET_WEIGHTS.values()), 
                                        hole=.3)])
        fig_pie.update_layout(title="Target Allocation Strategy", template="plotly_dark")
        st.plotly_chart(fig_pie)
