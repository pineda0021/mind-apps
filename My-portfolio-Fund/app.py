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
        st.error("Enter Email & App Password in Sidebar.")
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
        st.error(f"Login Failed: {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    user_email = st.text_input("Gmail Address")
    app_password = st.text_input("App Password", type="password")
    st.info("Check your Gmail 'App Passwords' settings if email fails.")

# --- TABS ---
tab1, tab2 = st.tabs(["🎯 Real-Time Monitor", "⚖️ Portfolio Health"])

# --- TAB 1: SINGLE TICKER ---
with tab1:
    st.title("Stock Watch & Alerts")
    ticker_input = st.text_input("Enter Ticker", value="VBAIX").upper()
    
    col1, col2 = st.columns(2)
    sell_target = col1.number_input("Sell Target ($)", value=200.0)
    drop_alert = col2.number_input("Drop Alert ($)", value=150.0)
    
    # Fetch Data with Fallback
    try:
        df = yf.download(ticker_input, period="5d", interval="1m", progress=False)
        if df.empty:
            df = yf.download(ticker_input, period="5d", interval="60m", progress=False)
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            current_p = float(df['Close'].iloc[-1])
            st.metric(f"Current {ticker_input} Price", f"${current_p:.2f}")
            
            fig = go.Figure(go.Scatter(x=df.index, y=df['Close'], line=dict(color='#00FFCC')))
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

            if st.button("Check Alerts"):
                if current_p >= sell_target:
                    send_email("SELL ALERT", f"{ticker_input} hit ${current_p:.2f}", user_email, app_password)
                elif current_p <= drop_alert:
                    send_email("DROP ALERT", f"{ticker_input} fell to ${current_p:.2f}", user_email, app_password)
                else:
                    st.success("Price is currently safe.")
    except Exception as e:
        st.error(f"Data Error: {e}")
