# quant-engine

[![CI](https://github.com/JanBoend/quant-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/JanBoend/quant-engine/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Vectorised backtesting engine for multi-instrument systematic strategies. I wrote this instead of wrapping backtrader or zipline because I wanted full control over fill logic and walk-forward validation from day one.

Part of a small set of tools: [market-regime-detector](https://github.com/JanBoend/market-regime-detector), [options-pricer](https://github.com/JanBoend/options-pricer), [factor-backtest](https://github.com/JanBoend/factor-backtest).

## What's in it

- Vectorised position simulation with slippage, commissions, fractional sizing
- Risk-per-trade sizing (fixed % of equity)
- Daily loss limit halting
- Rolling-window walk-forward validation
- Run across 10 instruments, 5+ years of data during development

## Usage

```python
import yfinance as yf
from engine import run_backtest, build_indicators, DEFAULT_PARAMS
from metrics import full_report

data = yf.download("QQQ", start="2019-01-01", end="2024-12-31", interval="1h")
htf  = data.resample("4h").last().dropna()
df   = build_indicators(data, htf)

params = {
    **DEFAULT_PARAMS,
    "risk_pct": 0.005,
    "reward_ratio": 3.0,
}

equity, trades = run_backtest(df, params)
report = full_report(equity, trades)
print(f"Sharpe: {report['sharpe']:.2f}  MaxDD: {report['max_drawdown']:.1%}")
```

## Engine parameters

| Parameter | Default | Description |
|---|---|---|
| `initial_capital` | 10000 | Starting equity |
| `risk_pct` | 0.005 | Fraction of equity risked per trade |
| `reward_ratio` | 4.0 | Take-profit in R-multiples |
| `daily_loss_limit` | 0.06 | Halt new entries after -6% intraday |
| `commission_pct` | 0.0001 | Round-trip commission as fraction of notional |
| `slippage_pts` | 2.0 | Slippage in price points per side |

## Walk-forward validation

```python
from walk_forward import rolling_holdout, wfo_summary
from engine import run_backtest, build_indicators

results = rolling_holdout(df, run_backtest, train_bars=504, test_bars=126, params=params)
summary = wfo_summary(results)
print(f"Pass rate: {summary['pass_rate']:.0%}  Mean OOS Sharpe: {summary['mean_sharpe']:.2f}")
```

13–15 of 16 out-of-sample windows profitable per strategy, across 10 instruments and 5+ years of data.

## Install

```bash
pip install -r requirements.txt
```

`examples/simple_ma.py` is the fastest way to see the API end to end. `examples/run_backtest.py` walks through the full report output.
