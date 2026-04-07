import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import smtplib
from email.message import EmailMessage

# ======================================================
# PAGE SETUP
# ======================================================
st.set_page_config(page_title="WealthMonitor Pro", layout="wide")
st.title("📊 WealthMonitor Pro")
st.caption("Portfolio view with entry prices, gain/loss colors, and optional email alerts")

# ======================================================
# SAFE ACCESS TO SECRETS
# ======================================================
def get_secret(name, default=""):
    try:
        return st.secrets[name]
    except Exception:
        return default

SMTP_SERVER = get_secret("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(get_secret("SMTP_PORT", 587))
EMAIL_SENDER = get_secret("EMAIL_SENDER", "")
EMAIL_PASSWORD = get_secret("EMAIL_PASSWORD", "")
EMAIL_RECIPIENT_DEFAULT = get_secret("EMAIL_RECIPIENT", "")

# ======================================================
# HELPERS
# ======================================================
def money(x):
    return f"${x:,.2f}"

def pct(x):
    return f"{x:,.2f}%"

def gain_label(x):
    if x > 0:
        return "🟢 Gain"
    elif x < 0:
        return "🔴 Loss"
    return "⚪ Flat"

@st.cache_data(ttl=600, show_spinner=False)
def load_one_ticker(ticker):
    """
    Download one ticker at a time.
    This is usually more stable on Streamlit Cloud than multi-ticker downloads.
    """
    df = yf.download(
        ticker,
        period="6mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=10,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize columns in case yfinance returns MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df = df["Close"].to_frame(name="Close")
        else:
            return pd.DataFrame()

    if "Close" not in df.columns:
        return pd.DataFrame()

    df = df[["Close"]].dropna()
    return df

def send_email_alert(subject, body, recipient):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        return False, "Email secrets are missing."

    if not recipient:
        return False, "No recipient email was provided."

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = recipient
        msg.set_content(body)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)

        return True, "Email alert sent."
    except Exception as e:
        return False, f"Email failed: {e}"

# ======================================================
# SIDEBAR INPUTS
# ======================================================
with st.sidebar:
    st.header("Portfolio Settings")

    ticker_text = st.text_area(
        "Tickers (comma-separated)",
        value="FXAIX, FXNAX, VTSNX"
    )

    tickers = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    if len(tickers) > 8:
        st.warning("Too many tickers can slow the app. Try 8 or fewer.")

    st.markdown("### Entry Prices")
    st.caption("Set your original buy price and shares for each ticker.")

    entry_prices = {}
    shares_owned = {}
    alert_below = {}
    alert_above = {}

    for t in tickers:
        entry_prices[t] = st.number_input(
            f"{t} Entry Price",
            min_value=0.0,
            value=100.00,
            step=0.01,
            format="%.2f",
            key=f"entry_{t}"
        )

        shares_owned[t] = st.number_input(
            f"{t} Shares",
            min_value=0.0,
            value=1.0,
            step=0.1,
            format="%.4f",
            key=f"shares_{t}"
        )

        alert_below[t] = st.number_input(
            f"{t} Alert Below",
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.2f",
            key=f"below_{t}"
        )

        alert_above[t] = st.number_input(
            f"{t} Alert Above",
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.2f",
            key=f"above_{t}"
        )

    st.markdown("### Email Alerts")
    enable_email_alerts = st.checkbox("Enable email alerts", value=False)
    recipient_email = st.text_input(
        "Recipient Email",
        value=EMAIL_RECIPIENT_DEFAULT
    )

    send_test_email = st.button("Send Test Email")

    show_raw = st.checkbox("Show raw downloaded data", value=False)

# ======================================================
# TEST EMAIL
# ======================================================
if send_test_email:
    ok, message = send_email_alert(
        subject="WealthMonitor Pro Test Email",
        body="This is a test email from your Streamlit app.",
        recipient=recipient_email
    )
    if ok:
        st.success(message)
    else:
        st.error(message)

# ======================================================
# DOWNLOAD DATA
# ======================================================
price_history = {}
rows = []
alert_messages = []

with st.spinner("Loading market data..."):
    for t in tickers:
        df_t = load_one_ticker(t)

        if df_t.empty:
            continue

        series = df_t["Close"].dropna()
        if series.empty:
            continue

        price_history[t] = series

        current_price = float(series.iloc[-1])
        entry_price = float(entry_prices[t])
        shares = float(shares_owned[t])

        cost_basis = entry_price * shares
        market_value = current_price * shares
        change_dollar = market_value - cost_basis
        change_percent = ((market_value / cost_basis) - 1) * 100 if cost_basis != 0 else 0.0

        below_val = float(alert_below[t]) if float(alert_below[t]) > 0 else None
        above_val = float(alert_above[t]) if float(alert_above[t]) > 0 else None

        if below_val is not None and current_price <= below_val:
            alert_messages.append(
                f"🔻 {t} is at {money(current_price)}, at or below your alert level of {money(below_val)}."
            )

        if above_val is not None and current_price >= above_val:
            alert_messages.append(
                f"🔺 {t} is at {money(current_price)}, at or above your alert level of {money(above_val)}."
            )

        rows.append({
            "Ticker": t,
            "Entry Price": entry_price,
            "Current Price": current_price,
            "Shares": shares,
            "Cost Basis": cost_basis,
            "Market Value": market_value,
            "Change ($)": change_dollar,
            "Change (%)": change_percent,
            "Status": gain_label(change_dollar),
            "Alert Below": below_val if below_val is not None else "",
            "Alert Above": above_val if above_val is not None else "",
        })

portfolio_df = pd.DataFrame(rows)

if portfolio_df.empty:
    st.error("No valid ticker data was returned. Try fewer tickers, or check the symbols.")
    st.stop()

# ======================================================
# OPTIONAL EMAIL ALERT SEND
# ======================================================
if enable_email_alerts and alert_messages:
    subject = "WealthMonitor Pro Alert"
    body = "\n".join(alert_messages)
    ok, message = send_email_alert(subject, body, recipient_email)
    if ok:
        st.success(message)
    else:
        st.error(message)

# ======================================================
# TOP SUMMARY
# ======================================================
total_cost = portfolio_df["Cost Basis"].sum()
total_value = portfolio_df["Market Value"].sum()
total_change = total_value - total_cost
total_change_pct = (total_change / total_cost * 100) if total_cost != 0 else 0.0

best_row = portfolio_df.loc[portfolio_df["Change (%)"].idxmax()]
worst_row = portfolio_df.loc[portfolio_df["Change (%)"].idxmin()]

st.subheader("Portfolio Snapshot")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Cost Basis", money(total_cost))
c2.metric("Total Market Value", money(total_value), delta=money(total_change))
c3.metric("Portfolio Return", pct(total_change_pct))
c4.metric("Tickers Tracked", str(len(portfolio_df)))

c5, c6 = st.columns(2)
c5.info(f"Best performer: **{best_row['Ticker']}** ({pct(best_row['Change (%)'])})")
c6.info(f"Worst performer: **{worst_row['Ticker']}** ({pct(worst_row['Change (%)'])})")

# ======================================================
# ALERTS
# ======================================================
st.subheader("Alerts")

if alert_messages:
    for msg in alert_messages:
        st.warning(msg)
else:
    st.success("No alerts triggered right now.")

# ======================================================
# PORTFOLIO TABLE
# ======================================================
st.subheader("Portfolio Table")

display_df = portfolio_df.copy()
display_df["Entry Price"] = display_df["Entry Price"].map(money)
display_df["Current Price"] = display_df["Current Price"].map(money)
display_df["Cost Basis"] = display_df["Cost Basis"].map(money)
display_df["Market Value"] = display_df["Market Value"].map(money)
display_df["Change ($)"] = display_df["Change ($)"].map(money)
display_df["Change (%)"] = display_df["Change (%)"].map(pct)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ======================================================
# CHART
# ======================================================
st.subheader("Price History")

fig = go.Figure()

for t, series in price_history.items():
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=t
        )
    )

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Adjusted Close Price",
    margin=dict(l=20, r=20, t=40, b=20),
    legend_title="Ticker",
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# ======================================================
# ENTRY PRICE STORAGE VIEW
# ======================================================
st.subheader("Stored Entry Prices")
ENTRY_PRICES = {row["Ticker"]: row["Entry Price"] for _, row in portfolio_df.iterrows()}
st.write(ENTRY_PRICES)

# ======================================================
# RAW DATA
# ======================================================
if show_raw:
    st.subheader("Raw Close Price Data")
    raw_close = pd.DataFrame(price_history)
    st.dataframe(raw_close, use_container_width=True)
    
