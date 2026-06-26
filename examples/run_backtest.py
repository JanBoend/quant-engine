# examples/run_backtest.py
"""Full API walkthrough — shows how to use quant-engine end to end."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yfinance as yf
from engine import load_alpaca_csv, run_backtest, build_indicators, DEFAULT_PARAMS
from metrics import full_report

print("quant-engine — full API example")
print("="*40)

# 1. Load data (yfinance example — swap for load_alpaca_csv with real data)
print("\n1. Loading data...")
raw = yf.download("QQQ", start="2022-01-01", end="2023-12-31", interval="1h", progress=False)
htf = raw.resample("4h").last().dropna()
print(f"   {len(raw)} hourly bars, {len(htf)} 4H bars")

# 2. Build indicators
print("\n2. Building indicators...")
df = build_indicators(raw, htf)
print(f"   {len(df)} bars with indicators")

# 3. Configure params
params = {**DEFAULT_PARAMS, "risk_pct": 0.005, "reward_ratio": 3.0}
print(f"\n3. Params: risk={params['risk_pct']:.1%}, RR={params['reward_ratio']}")

# 4. Run backtest
print("\n4. Running backtest...")
equity, trades = run_backtest(df, params)
print(f"   {len(trades)} trades")

# 5. Print report
print("\n5. Results:")
report = full_report(equity, trades)
for k, v in report.items():
    if isinstance(v, float):
        print(f"   {k:<18}: {v:.3f}")
    else:
        print(f"   {k:<18}: {v}")
