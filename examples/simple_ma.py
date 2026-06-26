# examples/simple_ma.py
"""Minimal working example: MA crossover strategy using quant-engine."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import yfinance as yf
from metrics import full_report


def build_signals(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    df = df.copy()
    df["fast_ma"] = df["Close"].rolling(fast).mean()
    df["slow_ma"] = df["Close"].rolling(slow).mean()
    df["signal"]  = (df["fast_ma"] > df["slow_ma"]).astype(int).diff()
    return df.dropna()


def run(ticker: str = "QQQ", start: str = "2019-01-01", end: str = "2024-12-31"):
    raw = yf.download(ticker, start=start, end=end, progress=False)
    df  = build_signals(raw)

    capital = 10_000
    equity  = [capital]
    trades  = []

    position = 0
    entry_price = 0.0

    for i, (ts, row) in enumerate(df.iterrows()):
        if row["signal"] == 1 and position == 0:
            position    = capital / row["Close"]
            entry_price = row["Close"]
        elif row["signal"] == -1 and position > 0:
            pnl = position * (row["Close"] - entry_price)
            trades.append({"entry": entry_price, "exit": row["Close"], "pnl": pnl})
            capital    += pnl
            position    = 0
            entry_price = 0.0
        equity.append(capital)

    eq = pd.Series(equity[1:], index=df.index)
    tr = pd.DataFrame(trades)
    report = full_report(eq, tr)

    print(f"\nMA Crossover ({ticker})")
    print(f"  Ann return : {report['ann_return']:.1%}")
    print(f"  Sharpe     : {report['sharpe']:.2f}")
    print(f"  Max DD     : {report['max_drawdown']:.1%}")
    print(f"  Trades     : {report['n_trades']}")
    print(f"  Win rate   : {report['win_rate']:.1%}")


if __name__ == "__main__":
    run()
