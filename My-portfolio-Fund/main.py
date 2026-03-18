import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

TICKERS = {"FXAIX":0.2,"FXNAX":0.6,"VTIAX":0.2}


def fetch_data():
    return yf.download(list(TICKERS.keys()), period="1y")["Adj Close"]


def compute(data):
    returns = data.pct_change().dropna()
    weights = np.array(list(TICKERS.values()))

    latest = data.iloc[-1]
    prev = data.iloc[-2]

    daily_returns = (latest/prev - 1)
    portfolio_return = np.dot(daily_returns, weights)

    vol = returns.dot(weights).std() * np.sqrt(252)

    return latest, daily_returns, portfolio_return, vol


def report():
    data = fetch_data()
    latest, daily, port_ret, vol = compute(data)

    print("=== DAILY REPORT ===")
    print(datetime.today().strftime("%Y-%m-%d"))

    for t in TICKERS:
        print(f"{t}: ${latest[t]:.2f} ({daily[t]*100:.2f}%)")

    print(f"\nPortfolio Return: {port_ret*100:.2f}%")
    print(f"Volatility: {vol*100:.2f}%")


if __name__ == "__main__":
    report()
