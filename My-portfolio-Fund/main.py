import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ===================== CONFIG =====================
TICKERS = {
    "FXAIX": 0.20,   # Fidelity 500 Index Fund
    "FXNAX": 0.60,   # Fidelity U.S. Bond Index Fund
    "VTIAX": 0.20    # Vanguard Intl Index (proxy)
}

PERIOD = "1y"
THRESHOLD_REBALANCE = 0.05

# ===================== DATA =====================
def fetch_data():
    data = yf.download(list(TICKERS.keys()), period=PERIOD)["Adj Close"]
    return data

# ===================== CALCULATIONS =====================
def compute_metrics(data):
    returns = data.pct_change().dropna()

    weights = np.array(list(TICKERS.values()))

    latest_prices = data.iloc[-1]
    prev_prices = data.iloc[-2]

    daily_returns = (latest_prices / prev_prices - 1)

    portfolio_returns = returns.dot(weights)

    daily_portfolio_return = np.dot(daily_returns, weights)
    ytd_return = (1 + portfolio_returns).prod() - 1
    volatility = portfolio_returns.std() * np.sqrt(252)

    return {
        "returns": returns,
        "latest_prices": latest_prices,
        "daily_returns": daily_returns,
        "portfolio_returns": portfolio_returns,
        "daily_portfolio_return": daily_portfolio_return,
        "ytd_return": ytd_return,
        "volatility": volatility
    }

# ===================== ANALYTICS =====================
def trend_analysis(data):
    ma_50 = data.rolling(50).mean().iloc[-1]
    ma_200 = data.rolling(200).mean().iloc[-1]

    trend = {}
    for ticker in TICKERS:
        if data.iloc[-1][ticker] > ma_50[ticker] > ma_200[ticker]:
            trend[ticker] = "Uptrend"
        elif data.iloc[-1][ticker] < ma_50[ticker] < ma_200[ticker]:
            trend[ticker] = "Downtrend"
        else:
            trend[ticker] = "Mixed"

    return trend, ma_50, ma_200


def drawdown_analysis(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()

    if max_dd < -0.20:
        state = "Bear Market"
    elif max_dd < -0.10:
        state = "Correction"
    else:
        state = "Normal"

    return max_dd, state


def rebalance_check(data):
    latest_prices = data.iloc[-1]
    weights = np.array(list(TICKERS.values()))

    current_weights = (latest_prices * weights) / (latest_prices * weights).sum()
    target_weights = weights

    drift = current_weights - target_weights

    alerts = []
    for i, ticker in enumerate(TICKERS.keys()):
        if abs(drift[i]) > THRESHOLD_REBALANCE:
            alerts.append(f"{ticker} drifted {drift[i]*100:.2f}%")

    return drift, alerts

# ===================== REPORT =====================
def generate_report(metrics, trend, ma_50, ma_200, max_dd, state, alerts):
    report = []

    report.append("=== DAILY PORTFOLIO REPORT ===")
    report.append(f"Date: {datetime.today().strftime('%Y-%m-%d')}\n")

    report.append("-- Fund Performance --")
    for ticker in TICKERS:
        price = metrics["latest_prices"][ticker]
        daily = metrics["daily_returns"][ticker]
        report.append(f"{ticker}: ${price:.2f} ({daily*100:.2f}%)")

    report.append("\n-- Portfolio Metrics --")
    report.append(f"Daily Return: {metrics['daily_portfolio_return']*100:.2f}%")
    report.append(f"YTD Return: {metrics['ytd_return']*100:.2f}%")
    report.append(f"Volatility: {metrics['volatility']*100:.2f}%")

    report.append("\n-- Trend Analysis --")
    for ticker in TICKERS:
        report.append(f"{ticker}: {trend[ticker]} (50MA: {ma_50[ticker]:.2f}, 200MA: {ma_200[ticker]:.2f})")

    report.append("\n-- Risk Signals --")
    report.append(f"Drawdown: {max_dd*100:.2f}% ({state})")

    report.append("\n-- Rebalancing Alerts --")
    if alerts:
        report.extend(alerts)
    else:
        report.append("No rebalancing needed")

    report.append("\n-- Commentary --")
    if metrics['daily_portfolio_return'] > 0:
        report.append("Portfolio gained today, likely driven by equity strength or falling yields.")
    else:
        report.append("Portfolio declined today, possibly due to equity weakness or rising rates.")

    if metrics['volatility'] < 0.08:
        report.append("Low-risk profile (bond-heavy allocation).")
    elif metrics['volatility'] < 0.15:
        report.append("Moderate risk environment.")
    else:
        report.append("Elevated volatility detected.")

    return "\n".join(report)

# ===================== MAIN =====================
def main():
    data = fetch_data()

    metrics = compute_metrics(data)
    trend, ma_50, ma_200 = trend_analysis(data)
    max_dd, state = drawdown_analysis(metrics['returns'])
    drift, alerts = rebalance_check(data)

    report = generate_report(metrics, trend, ma_50, ma_200, max_dd, state, alerts)

    print(report)


if __name__ == "__main__":
    main()
