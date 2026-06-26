# tests/test_metrics.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import pytest
from metrics import sharpe, max_drawdown, calmar, annualised_return, win_rate, profit_factor


def flat_equity(n=252, start=10000):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series([start] * n, index=idx)


def growing_equity(n=252, start=10000, end=12000):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    vals = np.linspace(start, end, n)
    return pd.Series(vals, index=idx)


def test_max_drawdown_no_drawdown():
    eq = growing_equity()
    assert max_drawdown(eq) == pytest.approx(0.0, abs=0.001)


def test_max_drawdown_with_drop():
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    eq  = pd.Series([10000, 12000, 9000, 11000], index=idx)
    assert max_drawdown(eq) == pytest.approx(-0.25, abs=0.001)


def test_sharpe_flat_returns_zero():
    eq = flat_equity()
    assert sharpe(eq) == pytest.approx(0.0, abs=0.1)


def test_annualised_return_20pct():
    eq = growing_equity(n=252, start=10000, end=12000)
    result = annualised_return(eq)
    assert 0.18 < result < 0.22


def test_win_rate():
    trades = pd.DataFrame({"pnl": [100, -50, 200, -30, 150]})
    assert win_rate(trades) == pytest.approx(0.6, abs=0.01)


def test_profit_factor():
    trades = pd.DataFrame({"pnl": [100, -50, 200, -50]})
    assert profit_factor(trades) == pytest.approx(3.0, abs=0.01)
