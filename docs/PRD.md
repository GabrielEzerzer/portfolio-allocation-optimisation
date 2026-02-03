# Product Requirements Document (PRD)

## Agent-Orchestrated ACO Portfolio Optimizer

### Overview

Build a working Python system where a runtime Operator launches multiple specialized Data Agents in parallel to gather/compute financial features for the S&P 500, merges outputs into a unified per-ticker dataset, and feeds it into an Ant Colony Optimization (ACO) engine that outputs a constrained portfolio, with a walk-forward backtester to evaluate out-of-sample performance.

---

## Acceptance Criteria

### Core Functionality

- [x] Program runs in **LIVE**, **CACHED**, and **BACKTEST** modes
- [x] In LIVE/CACHED mode, produces a portfolio (ticker → weight) and logs diagnostics
- [x] In BACKTEST mode, produces walk-forward metrics and baseline comparison
- [x] `pytest` passes in TEST mode with **no internet** and **no API keys**
- [x] No secrets (API keys, tokens) are committed - placeholders only

### Mode Requirements

#### LIVE Mode
- [x] Uses real API calls to fetch prices and fundamentals
- [x] Implements rate limiting, timeouts, retries/backoff
- [x] Caches responses

#### CACHED Mode  
- [x] Uses cached data snapshots
- [x] Can run without API keys if cache exists

#### TEST Mode
- [x] Tests do not call the internet
- [x] Providers are mocked
- [x] Deterministic runs via fixed seeds

### Agent Requirements

- [x] **UniverseAgent**: Produces ticker list with fallback to seed file
- [x] **PriceHistoryAgent**: Fetches historical OHLCV
- [x] **TechnicalIndicatorAgent**: Computes returns, volatility, momentum
- [x] **FundamentalsAgent**: Fetches market cap, PE, EPS, sector
- [x] **SentimentAgent**: Pluggable interface (stubbed for MVP)

### Operator Requirements

- [x] Launches agents concurrently with `asyncio.gather`
- [x] Enforces per-agent timeouts
- [x] Collects per-agent timings and coverage
- [x] Merges outputs into unified feature table
- [x] Handles missing values (configurable strategy)
- [x] Applies feature normalization

### ACO Requirements

- [x] Discretized weight allocation (5% chunks)
- [x] Pheromone + heuristic-based probability selection
- [x] Risk-adjusted fitness function (Sharpe-like)
- [x] Constraint enforcement (sum=1, max weight, holdings count)
- [x] Deterministic with random seed

### Backtest Requirements

- [x] Walk-forward with configurable train/test windows
- [x] Metrics: cumulative return, volatility, Sharpe, max drawdown, turnover
- [x] Baseline comparisons: equal-weight, SPY (if available)
- [x] Output artifacts saved to `data/outputs/`

### CLI Requirements

- [x] `--mode live|cached|backtest`
- [x] `--config` path to config file
- [x] `--tickers` comma-separated list
- [x] `--start` / `--end` date range
- [x] `--seed` for determinism

### Testing Requirements

- [x] Operator tests: concurrent execution, merge, failure handling
- [x] Agent tests: computation validation, mocked providers
- [x] ACO tests: weights sum to 1, constraints respected, determinism
- [x] Backtest tests: metrics accuracy

---

## Definition of Done

- [x] `pytest` passes offline (no keys, no internet)
- [x] `python -m src.app.main --mode live --tickers AAPL,MSFT` runs with configured API key
- [x] `python -m src.app.main --mode cached --tickers AAPL,MSFT` runs from cache
- [x] `python -m src.app.main --mode backtest --tickers AAPL,MSFT` runs walk-forward
- [x] Repo contains documentation matching actual behavior
- [x] No secrets committed; `.gitignore` is correct
- [x] Codebase is modular and aligned with Architecture
