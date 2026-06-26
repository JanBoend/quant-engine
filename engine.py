"""
engine.py — Vectorised backtesting engine for systematic trading strategies.
"""

import os
import pandas as pd
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_ROOT, "data")

# ── DEFAULT PARAMS ────────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    # Capital
    "initial_capital"  : 10_000,
    "risk_pct"         : 0.005,      # 0.5% of equity per trade
    "reward_ratio"     : 4.0,        # 1:4 RR
    "daily_loss_limit" : 0.06,       # halt trading after -6% on the day

    # Indicators
    "htf_ema_span"     : 50,
    "htf_swing_bars"   : 20,
    "ltf_atr_span"     : 14,
    "ltf_swing_bars"   : 10,

    # Signal filters
    "wick_threshold"   : 0.60,       # wick must be >60% of candle range
    "atr_filter_pct"   : 0.003,      # ATR must be >0.3% of price

    # Execution costs (realistic assumptions)
    "commission_pct"   : 0.0001,     # 0.01% of notional per side
    "slippage_pts"     : 2.0,        # price points per side
}


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_alpaca_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV produced by fetch_alpaca.py (Alpaca/QQQ data)."""
    if not os.path.isabs(filepath) and not os.path.exists(filepath):
        filepath = os.path.join(_DATA, filepath)
    df = pd.read_csv(filepath)
    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_localize(None)
    df = df.set_index("time")
    df.columns = [c.capitalize() for c in df.columns]
    return df.sort_index().dropna()


def load_tv_csv(filepath: str) -> pd.DataFrame:
    """Load a TradingView-exported CSV into a clean DataFrame."""
    df = pd.read_csv(filepath)
    df = df.drop(columns=["Plot"], errors="ignore")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df.index = df.index.tz_localize(None)
    df.columns = [c.capitalize() for c in df.columns]
    return df.sort_index().dropna()


# ── INDICATORS ────────────────────────────────────────────────────────────────
def build_indicators(ltf: pd.DataFrame, htf: pd.DataFrame, p: dict = None) -> pd.DataFrame:
    """
    Compute HTF + LTF indicators and merge onto the LTF frame.
    All forward-looking values are shifted by 1 bar before merge.
    Returns a copy — originals are not mutated.
    """
    if p is None:
        p = DEFAULT_PARAMS

    ltf = ltf.copy()
    htf = htf.copy()

    # ── HTF (all shifted 1 bar before forward-fill to LTF) ───────────────────
    htf["ema50"]      = htf["Close"].ewm(span=p["htf_ema_span"], adjust=False).mean()
    htf["swing_high"] = htf["High"].rolling(p["htf_swing_bars"]).max().shift(1)
    htf["swing_low"]  = htf["Low"].rolling(p["htf_swing_bars"]).min().shift(1)
    htf["bias_bull"]  = (htf["Close"] > htf["ema50"]).astype(int)
    htf["bias_bear"]  = (htf["Close"] < htf["ema50"]).astype(int)

    for col in ["ema50", "swing_high", "swing_low", "bias_bull", "bias_bear"]:
        ltf[f"htf_{col}"] = htf[col].reindex(ltf.index, method="ffill")

    # ── LTF ──────────────────────────────────────────────────────────────────
    prev_close = ltf["Close"].shift(1)
    tr = pd.concat([
        ltf["High"] - ltf["Low"],
        (ltf["High"] - prev_close).abs(),
        (ltf["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    ltf["atr"]   = tr.ewm(span=p["ltf_atr_span"], adjust=False).mean()
    ltf["ema20"] = ltf["Close"].ewm(span=20, adjust=False).mean()

    # Swing levels — shifted before use
    ltf["swing_high"] = ltf["High"].rolling(p["ltf_swing_bars"]).max().shift(1)
    ltf["swing_low"]  = ltf["Low"].rolling(p["ltf_swing_bars"]).min().shift(1)

    # Wick ratios
    candle_range           = (ltf["High"] - ltf["Low"]).replace(0, np.nan)
    ltf["lower_wick"]      = (ltf["Close"] - ltf["Low"])   / candle_range
    ltf["upper_wick"]      = (ltf["High"]  - ltf["Close"]) / candle_range
    ltf["lower_wick_prev"] = ltf["lower_wick"].shift(1)
    ltf["upper_wick_prev"] = ltf["upper_wick"].shift(1)

    # Break of structure
    ltf["bos_long"]  = (ltf["Close"] > ltf["swing_high"]).astype(int)
    ltf["bos_short"] = (ltf["Close"] < ltf["swing_low"]).astype(int)

    return ltf.dropna()


# ── BACKTEST ENGINE ───────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, p: dict = None) -> tuple[pd.Series, pd.DataFrame]:
    """
    Event-driven backtest with commissions and slippage.
    Returns (equity_curve, trade_log).
    """
    if p is None:
        p = DEFAULT_PARAMS

    capital     = p["initial_capital"]
    commission  = p.get("commission_pct", 0.0001)
    slippage    = p.get("slippage_pts",   2.0)
    risk_pct    = p["risk_pct"]
    rr          = p["reward_ratio"]
    daily_lim   = p["daily_loss_limit"]
    wick_thresh = p["wick_threshold"]
    atr_filt    = p["atr_filter_pct"]

    position    = 0
    entry_price = sl = tp = pos_size = 0.0
    equity_curve, trade_log = [], []
    daily_pnl   = 0.0
    day_start   = capital
    last_date   = None

    for ts, row in df.iterrows():
        today = ts.date()

        # ── Daily reset ───────────────────────────────────────────────────
        if last_date != today:
            daily_pnl = 0.0
            day_start = capital
            last_date = today

        # ── Circuit breaker ───────────────────────────────────────────────
        if daily_pnl <= -daily_lim * day_start:
            equity_curve.append(capital)
            continue

        # ── EXIT ──────────────────────────────────────────────────────────
        if position != 0:
            pnl, reason = 0, ""
            if position == 1:
                if   row["Low"]  <= sl: pnl, reason = (sl - entry_price) * pos_size, "SL"
                elif row["High"] >= tp: pnl, reason = (tp - entry_price) * pos_size, "TP"
            else:
                if   row["High"] >= sl: pnl, reason = (entry_price - sl) * pos_size, "SL"
                elif row["Low"]  <= tp: pnl, reason = (entry_price - tp) * pos_size, "TP"

            if reason:
                cost       = capital * commission + slippage * pos_size
                pnl       -= cost
                capital   += pnl
                daily_pnl += pnl
                trade_log.append({"date": ts, "type": "EXIT",
                                   "exit_reason": reason,
                                   "pnl": round(pnl, 2), "cost": round(cost, 2)})
                position = 0
                entry_price = sl = tp = pos_size = 0.0

        # ── ENTRY ─────────────────────────────────────────────────────────
        if position == 0:
            vol_ok     = row["atr"] > atr_filt * row["Close"]
            wick_long  = (row["lower_wick"]      > wick_thresh and
                          row["lower_wick_prev"]  > wick_thresh)
            wick_short = (row["upper_wick"]      > wick_thresh and
                          row["upper_wick_prev"]  > wick_thresh)

            if row["htf_bias_bull"] == 1 and vol_ok and wick_long and row["bos_long"]:
                sl_price = row["swing_low"]
                if sl_price < row["Close"]:
                    risk        = row["Close"] - sl_price
                    entry_price = row["Close"] + slippage          # fill above close
                    pos_size    = (capital * risk_pct) / risk
                    cost        = capital * commission + slippage * pos_size
                    capital    -= cost
                    sl          = sl_price
                    tp          = entry_price + rr * risk
                    position    = 1
                    trade_log.append({"date": ts, "type": "ENTER_LONG",
                                       "entry": round(entry_price, 2),
                                       "sl": round(sl, 2), "tp": round(tp, 2),
                                       "cost": round(cost, 2), "pnl": None})

            elif row["htf_bias_bear"] == 1 and vol_ok and wick_short and row["bos_short"]:
                sl_price = row["swing_high"]
                if sl_price > row["Close"]:
                    risk        = sl_price - row["Close"]
                    entry_price = row["Close"] - slippage          # fill below close
                    pos_size    = (capital * risk_pct) / risk
                    cost        = capital * commission + slippage * pos_size
                    capital    -= cost
                    sl          = sl_price
                    tp          = entry_price - rr * risk
                    position    = -1
                    trade_log.append({"date": ts, "type": "ENTER_SHORT",
                                       "entry": round(entry_price, 2),
                                       "sl": round(sl, 2), "tp": round(tp, 2),
                                       "cost": round(cost, 2), "pnl": None})

        equity_curve.append(capital)

    # Close any open position at end of data
    if position != 0:
        last_close = df.iloc[-1]["Close"]
        pnl  = ((last_close - entry_price) if position == 1
                else (entry_price - last_close)) * pos_size
        cost = capital * commission + slippage * pos_size
        pnl -= cost
        capital += pnl
        equity_curve[-1] = capital
        trade_log.append({"date": df.index[-1], "type": "EXIT",
                           "exit_reason": "END_OF_DATA",
                           "pnl": round(pnl, 2), "cost": round(cost, 2)})

    equity = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
    trades = pd.DataFrame(trade_log)
    return equity, trades


# ── METRICS ───────────────────────────────────────────────────────────────────
def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    """Compute standard performance metrics from an equity curve + trade log."""
    exits       = trades[trades["type"] == "EXIT"]
    winners     = exits[exits["pnl"] > 0]
    losers      = exits[exits["pnl"] < 0]
    be_exits    = exits[exits["exit_reason"] == "SL_BE"] if "exit_reason" in exits.columns else exits.iloc[:0]
    partial_tps = trades[trades["type"] == "PARTIAL_TP"] if len(trades) > 0 else exits.iloc[:0]

    ret      = (equity.iloc[-1] / equity.iloc[0]) - 1
    max_dd   = (equity / equity.cummax()).min() - 1
    rets     = equity.pct_change().dropna()
    sharpe   = (rets.mean() / rets.std()) * np.sqrt(252 * 26) if rets.std() > 0 else 0

    win_rate      = len(winners) / len(exits) if len(exits) > 0 else 0
    avg_win       = winners["pnl"].mean() if len(winners) > 0 else 0
    avg_loss      = losers["pnl"].mean()  if len(losers)  > 0 else 0
    actual_rr     = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    total_cost    = exits["cost"].sum() if "cost" in exits.columns else 0
    gross_profit  = winners["pnl"].sum() if len(winners) > 0 else 0
    gross_loss    = abs(losers["pnl"].sum()) if len(losers) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    calmar        = ret / abs(max_dd) if max_dd != 0 else 0

    return {
        "return"        : ret,
        "max_dd"        : max_dd,
        "sharpe"        : sharpe,
        "calmar"        : calmar,
        "profit_factor" : profit_factor,
        "n_trades"      : len(exits),
        "win_rate"      : win_rate,
        "avg_win"       : avg_win,
        "avg_loss"      : avg_loss,
        "actual_rr"     : actual_rr,
        "total_cost"    : total_cost,
        "be_exits"      : len(be_exits),
        "n_partial_tp"  : len(partial_tps),
        "final_capital" : equity.iloc[-1],
    }
