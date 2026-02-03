# Walk-Forward Backtesting

## Overview

The walk-forward backtester evaluates the ACO portfolio optimizer using out-of-sample testing. This methodology ensures that portfolio performance is measured on data not used for optimization.

## Methodology

### Walk-Forward Process

```
Time ──────────────────────────────────────────────────────►

     │────── Train ──────│── Test ──│
                         │────── Train ──────│── Test ──│
                                             │────── Train ──────│── Test ──│
```

For each rebalance date:

1. **Train Window**: Use past N days (default: 252 trading days ≈ 1 year)
2. **Optimize**: Run operator + agents to gather features, then ACO to get weights
3. **Test Window**: Hold portfolio for M days (default: 21 trading days ≈ 1 month)
4. **Record**: Capture returns, turnover, and portfolio details
5. **Roll Forward**: Move to next rebalance date and repeat

### Configuration

```yaml
backtest:
  train_window_days: 252    # Training lookback
  test_window_days: 21      # Holding period
  rebalance_frequency: monthly  # Rebalance timing
  risk_free_rate: 0.0       # For Sharpe calculation
  transaction_cost: 0.001   # 0.1% per trade
```

## Metrics

### Performance Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| Cumulative Return | Total return over backtest | ∏(1 + rₜ) - 1 |
| Annualized Return | Yearly equivalent return | (1 + cum)^(252/n) - 1 |
| Annualized Volatility | Yearly risk measure | σ_daily × √252 |
| Sharpe Ratio | Risk-adjusted return | (r - rf) / σ |
| Max Drawdown | Largest peak-to-trough | max((peak - trough) / peak) |
| Turnover | Portfolio change per rebalance | Σ|wₜ - wₜ₋₁| / 2 |

### Baseline Comparisons

1. **Equal Weight**: Same tickers, equal allocation
2. **SPY Benchmark**: S&P 500 ETF (if available in universe)

## Output

### Console Summary

```
============================================================
BACKTEST RESULTS SUMMARY
============================================================

Windows: 12
Period: 2023-01-31 to 2024-01-31

--- ACO Portfolio ---
  cumulative_return: 0.2534
  annualized_return: 0.2534
  annualized_volatility: 0.2215
  sharpe_ratio: 1.1440
  max_drawdown: 0.0823
  avg_turnover: 0.2341

--- Baseline Comparison ---
  equal_weight:
    cumulative_return: 0.1823
    sharpe_ratio: 0.8234
    max_drawdown: 0.1045
```

### JSON Output

Saved to `data/outputs/backtest_results.json`:

```json
{
  "metrics": {
    "cumulative_return": 0.2534,
    "annualized_return": 0.2534,
    "sharpe_ratio": 1.144,
    "max_drawdown": 0.0823,
    "avg_turnover": 0.2341
  },
  "baseline_metrics": {
    "equal_weight": {
      "cumulative_return": 0.1823,
      "sharpe_ratio": 0.8234
    }
  },
  "windows": [
    {
      "train_end": "2023-01-31",
      "test_start": "2023-02-01",
      "test_end": "2023-02-28",
      "test_return": 0.0234,
      "turnover": 0.15,
      "num_holdings": 8,
      "top_holdings": [["AAPL", 0.10], ["MSFT", 0.10]]
    }
  ]
}
```

## Usage Examples

### Basic Backtest

```bash
python -m src.app.main --mode backtest \
  --tickers AAPL,MSFT,GOOGL,AMZN,META \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --seed 42
```

### Extended Backtest

```bash
python -m src.app.main --mode backtest \
  --tickers AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,V,UNH \
  --start 2022-01-01 \
  --end 2024-01-01 \
  --seed 42 \
  --output results/extended_backtest/
```

## Interpreting Results

### Positive Indicators

- **Sharpe > 1.0**: Good risk-adjusted returns
- **Outperformance vs equal-weight**: ACO adding value
- **Reasonable turnover (20-40%)**: Not over-trading
- **Lower max drawdown vs baseline**: Better risk management

### Warning Signs

- **High turnover (>75%)**: Excessive trading
- **Underperformance vs equal-weight**: Consider simpler approach
- **Sharpe < 0**: Losing money after risk adjustment
- **Extreme max drawdown (>30%)**: Poor risk control

## Limitations

1. **Transaction Costs**: Simplified model (flat %)
2. **Slippage**: Not modeled
3. **Market Impact**: Assumes unlimited liquidity
4. **Look-Ahead Bias**: Carefully avoided via walk-forward
5. **Survivorship Bias**: Uses current S&P 500 constituents
