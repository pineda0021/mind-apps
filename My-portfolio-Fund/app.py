import streamlit as st
import yfinance as yf
import pandas as pd
import smtplib
from email.message import EmailMessage

# --- APP CONFIG ---
st.set_page_config(page_title="Stock Monitor & Alerter", layout="wide")
st.title("📈 Stock Real-Time Monitor & Email Alerter")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("Settings")
    ticker_symbol = st.text_input("Enter Ticker", value="AAPL").upper()
    sell_target = st.number_input("Sell Target Price ($)", value=200.0)
    drop_threshold = st.number_input("Email if Price Drops Below ($)", value=150.0)
    
    st.divider()
    user_email = st.text_input("Your Email (for alerts)")
    # For Gmail, you need an "App Password," not your regular password
    email_password = st.text_input("Email App Password", type="password")

# --- DATA FETCHING ---
def get_data(ticker):
    stock = yf.Ticker(ticker)
    # Get 1-day data with 1-minute intervals for "real-time" feel
    df = stock.history(period="1d", interval="1m")
    return df

def send_alert(subject, body, to_email, password):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = to_email
        msg['To'] = to_email

        # Using Gmail SMTP settings as an example
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(to_email, password)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email failed: {e}")
        return False

# --- MAIN UI ---
if ticker_symbol:
    data = get_data(ticker_symbol)
    
    if not data.empty:
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        change = current_price - prev_price

        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:.2f}", f"{change:.2f}")
        col2.metric("Sell Target", f"${sell_target:.2f}")
        col3.metric("Drop Alert", f"${drop_threshold:.2f}")

        # Plotting
        st.line_chart(data['Close'])

        # --- LOGIC ENGINES ---
        st.subheader("Automation Status")
        
        # 1. Sell Trigger
        if current_price >= sell_target:
            st.error(f"!!! SELL SIGNAL !!! {ticker_symbol} reached ${current_price:.2f}")
            if user_email and email_password:
                if send_alert("SELL ALERT", f"{ticker_symbol} hit target: ${current_price}", user_email, email_password):
                    st.success("Sell email sent!")

        # 2. Drop Trigger
        elif current_price <= drop_threshold:
            st.warning(f"Price Drop Detected: ${current_price:.2f}")
            if user_email and email_password:
                if send_alert("DROP ALERT", f"{ticker_symbol} fell to: ${current_price}", user_email, email_password):
                    st.info("Drop notification sent!")
        
        else:
            st.success("Price is within your preferred range. Status: HOLD")
    else:
        st.error("Ticker not found. Please check the symbol.")

# Auto-refresh logic (optional, every 60 seconds)
# st.empty() 
# st.rerun() # Use with caution to avoid API rate limits
