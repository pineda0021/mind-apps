import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import smtplib
from email.message import EmailMessage

# --- CONFIG ---
TARGET_WEIGHTS = {"FXAIX": 0.2, "FXNAX": 0.6, "VTIAX": 0.2}
st.set_page_config(page_title="WealthMonitor Pro", layout="wide")

# --- EMAIL FUNCTION ---
def send_email(subject, content, user_email, app_password):
    if not user_email or not app_password:
        st.error("Missing Email or App Password in Sidebar.")
        return
    msg = EmailMessage()
    msg.set_content(content)
    msg['Subject'] = subject
    msg['From'] = user_email
    msg['To'] = user_email
    try:
        # Standard Gmail SMTP settings
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(user_email, app_password)
            smtp.send_message(msg)
        st.toast("Email sent successfully!", icon="📧")
    except Exception as e:
        st.error(f"Login/SMTP Failed: {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.header("👤 Account Settings")
    user_email = st.text_input("Gmail Address", placeholder="yourname@gmail.com")
    app_password = st.text_input("Gmail App Password", type="password", help="16-character code from Google Security settings.")
    st.divider()
    st.caption("Instructions: Use Google 'App Passwords' (not your login password) for this to work.")

# --- APP TABS ---
tab1, tab2 = st.tabs(["🎯 Real-Time Monitor", "⚖️ Portfolio Health"])

# --- TAB 1: SINGLE TICKER ---
with tab1:
    st.title("Ticker Alert System")
    ticker_input = st.text_input("Stock Ticker", value="VBAIX").upper()
    
    col_a, col_b = st.columns(2)
    sell_target = col_a.number_input("Sell Target ($)", value=200.0)
    drop_alert = col_b.number_input("Drop Alert ($)", value=150.0)
    
    # Fetch Data: Try 1m (Real-Time), fallback to 1h if market is closed
    df = yf.download(ticker_input, period="5d", interval="1m", progress=False)
    if df.empty:
        df = yf.download(ticker_input, period="5d", interval="60m", progress=False)
    
    if not df.empty:
        # Clean MultiIndex columns if present (common in new yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        current_price = float(df['Close'].iloc[-1])
        open_price = float(df['Open'].iloc[0])
        change = current_price - open_price
        
        st.metric(f"{ticker_input} Current", f"${current_price:.2f}", f"{change:.2f}")
        
        # Interactive Plotly Chart
        fig = go.Figure(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color='#00FFCC')))
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Manual Alert Scan"):
            if current_price >= sell_target:
                send_email("📈 SELL ALERT", f"{ticker_input} hit target of ${sell_target}. Current: ${current_price:.2f}", user_email, app_password)
            elif current_price <= drop_alert:
                send_email("📉 DROP ALERT", f"{ticker_input} fell below ${drop_alert}. Current: ${current_price:.2f}", user_email, app_password)
            else:
                st.success("Price is currently between your targets. No alert sent.")
    else:
        st.warning(f"Could not find data for {ticker_input}. Please check the symbol.")

# --- TAB 2: PORTFOLIO ---
with tab2:
    st.title("Portfolio Allocation Status")
    
    # Fetch Data for all portfolio tickers
    port_tickers = list(TARGET_WEIGHTS.keys())
    port_data = yf.download(port_tickers, period="1y", progress=False)["Adj Close"]
    
    if not port_data.empty:
        latest = port_data.iloc[-1]
        
        st.subheader("Current Allocations")
        cols = st.columns(len(TARGET_WEIGHTS))
        for i, (t, w) in enumerate(TARGET_WEIGHTS.items()):
            price = latest[t] if t in latest else 0.0
            cols[i].metric(t, f"${price:.2f}", f"Target: {w*100:.0f}%")
        
        # Pie Chart of Targets
        fig_pie = go.Figure(data=[go.Pie(labels=list(TARGET_WEIGHTS.keys()), 
                                        values=list(TARGET_WEIGHTS.values()), 
                                        hole=.3)])
        fig_pie.update_layout(template="plotly_dark", title="Target Strategy")
        st.plotly_chart(fig_pie)
