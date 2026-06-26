# metrics.py
import numpy as np
import pandas as pd


def sharpe(equity: pd.Series, risk_free: float = 0.0) -> float:
    returns = equity.pct_change().dropna()
    excess = returns - risk_free / 252
    return float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def calmar(equity: pd.Series) -> float:
    ann = annualised_return(equity)
    mdd = abs(max_drawdown(equity))
    return ann / mdd if mdd > 0 else 0.0


def annualised_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0


def win_rate(trades: pd.DataFrame) -> float:
    closed = trades[trades["pnl"].notna()]
    return float((closed["pnl"] > 0).mean()) if len(closed) > 0 else 0.0


def profit_factor(trades: pd.DataFrame) -> float:
    closed = trades[trades["pnl"].notna()]
    gross_win = closed[closed["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(closed[closed["pnl"] < 0]["pnl"].sum())
    return float(gross_win / gross_loss) if gross_loss > 0 else float("inf")


def avg_r(trades: pd.DataFrame) -> float:
    if "r_multiple" in trades.columns:
        return float(trades["r_multiple"].mean())
    return 0.0


def full_report(equity: pd.Series, trades: pd.DataFrame) -> dict:
    return {
        "ann_return": annualised_return(equity),
        "sharpe": sharpe(equity),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(equity),
        "n_trades": len(trades),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "avg_r": avg_r(trades),
    }
