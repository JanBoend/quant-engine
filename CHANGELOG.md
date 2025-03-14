# Changelog

## [1.2.0] - 2025-03-14
- Add `rolling_holdout` walk-forward evaluator with configurable window/step
- Fix Sharpe annualisation factor (was using bar count, now correctly uses 252)
- Add `profit_factor` and `avg_r` to metrics module

## [1.1.0] - 2024-11-08
- Add `load_tv_csv` loader for TradingView exports
- Add `max_drawdown` and `calmar` ratio to `compute_metrics`
- Improve position sizing: switch from fixed-lot to risk-pct based sizing

## [1.0.1] - 2024-08-22
- Fix timezone handling in `load_alpaca_csv` for tz-aware indices
- Add `win_rate` metric

## [1.0.0] - 2024-06-01
- Initial release: vectorised backtesting engine with full trade accounting
