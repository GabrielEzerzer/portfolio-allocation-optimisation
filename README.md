# ACO Portfolio Optimizer

An agent-orchestrated Ant Colony Optimization (ACO) system for S&P 500 portfolio construction with walk-forward backtesting.

## Overview

This system uses multiple specialized runtime agents to gather financial data (prices, fundamentals, technicals, sentiment) for S&P 500 stocks, merges them into a unified feature set, and applies Ant Colony Optimization to construct an optimized portfolio with constraints.

## Features

- **Multi-Agent Architecture**: Parallel data gathering with specialized agents
- **ACO Optimization**: Portfolio construction using Ant Colony Optimization
- **Walk-Forward Backtesting**: Out-of-sample performance evaluation
- **Pluggable Data Providers**: Alpha Vantage, yfinance fallback
- **Three Modes**: LIVE (API), CACHED (offline), BACKTEST (historical)
- **Comprehensive Constraints**: Max weight, min/max holdings, sector caps

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/GabrielEzerzer/portfolio-allocation-optimisation.git
cd portfolio-allocation-optimisation

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy the example config:
   ```bash
   copy config\settings.example.yaml config\settings.local.yaml
   ```

2. Set your API key (optional for cached mode):
   ```bash
   set ALPHAVANTAGE_API_KEY=your_key_here
   ```

### Running the Application

The **highly recommended** way to interact with the ACO Optimizer is through the Web Dashboard. The beautiful Streamlit UI lets you configure everything interactively without dealing with CLI flags.

```bash
# Launch the Web Dashboard
python -m streamlit run streamlit_app.py
```

This will automatically open the UI in your default web browser at `http://localhost:8501`.

*(Alternatively, you can still run the engine via the CLI using `python -m src.app.main --mode live ...`)*

### Running Tests

```bash
# Run all tests (offline, no API needed)
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/app
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI (main.py)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Operator                               │
│  - Orchestrates agents concurrently                         │
│  - Merges outputs into unified feature table                │
│  - Handles missing values and normalization                 │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  UniverseAgent  │ │ PriceAgent      │ │ TechnicalAgent  │
│  (ticker list)  │ │ (OHLCV data)    │ │ (indicators)    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ FundamentalsAgent│ │ SentimentAgent │ │  Data Providers │
│ (market cap, PE) │ │ (synthetic)    │ │ (AV, yfinance)  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ACO Optimizer                             │
│  - Discretized weight allocation                            │
│  - Pheromone trails + heuristic desirability                │
│  - Constraint enforcement (weights, holdings, sectors)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Walk-Forward Backtester                    │
│  - Rolling train/test windows                               │
│  - Baseline comparisons (equal-weight, SPY)                 │
│  - Metrics: Sharpe, drawdown, turnover                      │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
repo/
├── src/app/
│   ├── main.py              # CLI entrypoint
│   ├── config.py            # Configuration management
│   ├── context.py           # RunContext shared state
│   ├── operator.py          # Agent orchestration
│   ├── agents/              # Data agents
│   │   ├── universe_agent.py
│   │   ├── price_agent.py
│   │   ├── technical_agent.py
│   │   ├── fundamentals_agent.py
│   │   └── sentiment_agent.py
│   ├── data_providers/      # API providers
│   │   ├── alphavantage.py
│   │   └── yfinance_fallback.py
│   ├── optimization/        # ACO engine
│   │   ├── aco.py
│   │   ├── fitness.py
│   │   ├── constraints.py
│   │   └── portfolio.py
│   ├── backtesting/         # Walk-forward
│   │   ├── walk_forward.py
│   │   └── metrics.py
│   └── utils/               # Utilities
├── tests/                   # Test suite
├── config/                  # Configuration files
├── data/
│   ├── universe/            # Seed files
│   ├── cache/               # Cached API responses
│   └── outputs/             # Results
└── docs/                    # Documentation
```

## Configuration

See `config/settings.example.yaml` for all options:

- **Providers**: API endpoints, rate limits, timeouts
- **Agents**: Enable/disable, timeouts, parameters
- **ACO**: Ants, iterations, evaporation, alpha/beta
- **Constraints**: Max weight, holdings limits
- **Backtest**: Train/test windows, rebalance frequency

## Documentation

- [PRD](docs/PRD.md) - Product requirements and acceptance criteria
- [Architecture](docs/Architecture.md) - Technical design
- [Usage](docs/Usage.md) - Detailed usage guide
- [Backtesting](docs/Backtesting.md) - Walk-forward methodology
- [Test Plan](docs/TestPlan.md) - Test coverage
- [Test Report](docs/TestReport.md) - Test results

## License

MIT
