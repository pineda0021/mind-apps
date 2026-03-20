import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import smtplib
from email.message import EmailMessage

# --- PAGE SETUP ---
st.set_page_config(page_title="Stock Monitor Pro", layout="wide")

# --- UI: SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("📈 Control Panel")
    ticker = st.text_input("Stock Ticker", value="VBAIX").upper()
    
    st.subheader("Thresholds")
    sell_target = st.number_input("Sell Target (Price High)", value=200.0)
    drop_alert = st.number_input("Drop Alert (Price Low)", value=150.0)
    
    st.divider()
    st.subheader("Email Alerts")
    user_email = st.text_input("Your Email")
    app_password = st.text_input("App Password", type="password", help="Use a Google App Password, not your login password.")
    
    run_monitor = st.button("Check Market Now")

# --- FUNCTIONS ---
def send_email(subject, content):
    if not user_email or not app_password:
        st.error("Email settings missing!")
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
        st.toast("Email alert sent!", icon="📧")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# --- MAIN DASHBOARD ---
st.title(f"Real-Time Monitor: {ticker}")

# Fetch 1-day data with 1-minute intervals
data = yf.download(ticker, period="1d", interval="1m")

if not data.empty:
    # Use the latest "Close" price
    current_price = float(data['Close'].iloc[-1])
    open_price = float(data['Open'].iloc[0])
    price_change = current_price - open_price
    percent_change = (price_change / open_price) * 100

    # 1. Metrics Row
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f} ({percent_change:.2f}%)")
    col2.metric("Target (Sell)", f"${sell_target:.2f}", f"{current_price - sell_target:.2f}", delta_color="inverse")
    col3.metric("Floor (Drop)", f"${drop_alert:.2f}", f"{current_price - drop_alert:.2f}")

    # 2. Interactive Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], line=dict(color='#00FFCC', width=2), name="Price"))
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        margin=dict(l=20, r=20, t=20, b=20),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. Logic Engine
    st.subheader("Alert Status")
    
    if current_price >= sell_target:
        st.error(f"🚨 SELL SIGNAL: {ticker} has hit ${current_price:.2f}")
        if run_monitor:
            send_email(f"SELL ALERT: {ticker}", f"{ticker} is at ${current_price:.2f}. Target of ${sell_target} reached.")
            
    elif current_price <= drop_alert:
        st.warning(f"⚠️ DROP ALERT: {ticker} fell to ${current_price:.2f}")
        if run_monitor:
            send_email(f"DROP ALERT: {ticker}", f"{ticker} fell to ${current_price:.2f}. Floor of ${drop_alert} breached.")
            
    else:
        st.info("Price is within your defined boundaries. No action taken.")

else:
    st.error("Unable to fetch data. Check the ticker symbol or market hours.")

# --- FOOTER ---
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
