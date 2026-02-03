# Usage Guide

## Installation

```bash
# Clone repository
git clone https://github.com/GabrielEzerzer/portfolio-allocation-optimisation.git
cd portfolio-allocation-optimisation

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### Configuration Files

- `config/settings.example.yaml` - Template with all options
- `config/settings.local.yaml` - Your local config (gitignored)

### API Key Setup

Set your Alpha Vantage API key:

```bash
# Windows
set ALPHAVANTAGE_API_KEY=your_key_here

# Linux/Mac
export ALPHAVANTAGE_API_KEY=your_key_here
```

Or add to `config/settings.local.yaml`:
```yaml
providers:
  alphavantage:
    api_key: "your_key_here"
```

### Key Configuration Options

```yaml
# Execution mode
mode: live  # live | cached | backtest

# ACO parameters
aco:
  num_ants: 20
  num_iterations: 50
  weight_granularity: 0.05  # 5% chunks
  random_seed: 42  # For reproducibility

# Portfolio constraints
constraints:
  max_weight_per_ticker: 0.10  # 10% max
  min_holdings: 5
  max_holdings: 50

# Backtest settings
backtest:
  train_window_days: 252  # 1 year
  test_window_days: 21    # 1 month
  rebalance_frequency: monthly
```

## Running the Optimizer

### LIVE Mode

Fetches real-time data from APIs:

```bash
# Basic run with specific tickers
python -m src.app.main --mode live --tickers AAPL,MSFT,GOOGL,AMZN,META

# With custom config
python -m src.app.main --mode live --tickers AAPL,MSFT,GOOGL --config config/settings.local.yaml

# Deterministic run
python -m src.app.main --mode live --tickers AAPL,MSFT,GOOGL --seed 42

# Full S&P 500 (rate limits apply!)
python -m src.app.main --mode live --universe-size 50
```

### CACHED Mode

Uses cached data, no API calls:

```bash
# Run from cache
python -m src.app.main --mode cached --tickers AAPL,MSFT,GOOGL

# Useful for development/testing
python -m src.app.main --mode cached --tickers AAPL,MSFT,GOOGL,AMZN,META --seed 42
```

### BACKTEST Mode

Walk-forward historical analysis:

```bash
# Basic backtest
python -m src.app.main --mode backtest --tickers AAPL,MSFT,GOOGL,AMZN,META --start 2023-01-01 --end 2024-01-01

# With specific seed
python -m src.app.main --mode backtest --tickers AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,V,UNH --start 2022-01-01 --end 2024-01-01 --seed 42

# Custom output directory
python -m src.app.main --mode backtest --tickers AAPL,MSFT,GOOGL --start 2023-01-01 --end 2024-01-01 --output results/
```

## CLI Reference

```
usage: main.py [-h] --mode {live,cached,backtest} [--config CONFIG]
               [--tickers TICKERS] [--universe-size UNIVERSE_SIZE]
               [--start START] [--end END] [--seed SEED] [--output OUTPUT]
               [--verbose]

ACO Portfolio Optimizer

optional arguments:
  --mode {live,cached,backtest}
                        Execution mode
  --config CONFIG       Path to configuration file
  --tickers TICKERS     Comma-separated ticker list
  --universe-size N     Limit universe to N tickers
  --start YYYY-MM-DD    Start date for backtest/data
  --end YYYY-MM-DD      End date for backtest/data
  --seed N              Random seed for determinism
  --output PATH         Output directory
  --verbose, -v         Enable verbose logging
```

## Output

### Portfolio Output (LIVE/CACHED)

```
============================================================
PORTFOLIO OPTIMIZATION RESULT
============================================================

Fitness Score: 0.8542
Number of Holdings: 8
Total Weight: 100.00%
Max Weight: 10.00%

Top 10 Holdings:
----------------------------------------
  NVDA       10.00%
  AAPL       10.00%
  MSFT       10.00%
  GOOGL      10.00%
  META       10.00%
  AMZN       10.00%
  TSLA       10.00%
  JPM        10.00%

Expected Annual Return: 25.34%
Expected Volatility: 22.15%
Sharpe Ratio: 1.14

============================================================
```

### Backtest Output

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

============================================================
```

## Troubleshooting

### "No price data available"

- Check API key is set correctly
- Verify ticker symbols are valid
- Check rate limits (Alpha Vantage: 5/min free tier)
- Try running in cached mode with existing cache

### "Timeout" errors

- Increase timeout in config
- Reduce universe size
- Check network connectivity

### Tests fail with network errors

- Tests should not make network calls
- Check that conftest.py socket blocking is in place
- Ensure all providers are mocked in tests
