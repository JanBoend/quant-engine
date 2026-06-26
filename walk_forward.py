# walk_forward.py
"""Rolling and expanding window walk-forward optimisation."""
import pandas as pd
from typing import Callable


def rolling_wfo(
    data: pd.DataFrame,
    run_fn: Callable,
    train_bars: int,
    test_bars: int,
    params: dict,
) -> list[dict]:
    """
    Split data into rolling train/test windows, run run_fn on each.
    run_fn(data, params) -> (equity: pd.Series, trades: pd.DataFrame)
    Returns list of dicts with window dates and out-of-sample metrics.
    """
    from metrics import full_report

    results = []
    n = len(data)
    start = 0

    while start + train_bars + test_bars <= n:
        train = data.iloc[start : start + train_bars]
        test  = data.iloc[start + train_bars : start + train_bars + test_bars]

        equity_oos, trades_oos = run_fn(test, params)

        if len(equity_oos) > 1:
            report = full_report(equity_oos, trades_oos)
            report["oos_start"] = test.index[0]
            report["oos_end"]   = test.index[-1]
            results.append(report)

        start += test_bars

    return results


def wfo_summary(results: list[dict]) -> dict:
    """Summarise WFO results: pass rate, mean OOS Sharpe, mean OOS return."""
    if not results:
        return {}
    sharpes = [r["sharpe"] for r in results]
    returns = [r["ann_return"] for r in results]
    passes  = sum(1 for r in returns if r > 0)
    return {
        "windows":         len(results),
        "pass_rate":       passes / len(results),
        "mean_sharpe":     sum(sharpes) / len(sharpes),
        "mean_ann_return": sum(returns) / len(returns),
    }
