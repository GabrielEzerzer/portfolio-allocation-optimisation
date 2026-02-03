# Test Plan

## Overview

This document enumerates the test coverage for the ACO Portfolio Optimizer. All tests run **offline** with no network calls.

## Test Categories

### 1. Operator Tests (`test_operator.py`)

| Test | Description | Status |
|------|-------------|--------|
| `test_operator_runs_agents_concurrently` | Verify agents execute in parallel | ✅ |
| `test_operator_merges_outputs_correctly` | Check feature table construction | ✅ |
| `test_operator_handles_missing_values_drop` | Drop strategy for NaN | ✅ |
| `test_operator_handles_missing_values_fill_median` | Fill strategy for NaN | ✅ |
| `test_operator_handles_agent_failure_gracefully` | Exception handling | ✅ |
| `test_operator_result_properties` | Result accessors | ✅ |

### 2. Agent Tests (`test_agents.py`)

| Test | Description | Status |
|------|-------------|--------|
| **AgentResult** | | |
| `test_agent_result_creation` | Basic instantiation | ✅ |
| `test_agent_result_coverage` | Coverage property | ✅ |
| `test_agent_result_errors` | Error tracking | ✅ |
| `test_agent_result_add_error` | Adding errors | ✅ |
| **UniverseAgent** | | |
| `test_universe_agent_with_provided_tickers` | Returns provided list | ✅ |
| `test_universe_agent_respects_size_limit` | Size limiting | ✅ |
| **TechnicalIndicatorAgent** | | |
| `test_technical_agent_computes_indicators` | Expected columns | ✅ |
| `test_technical_agent_handles_empty_price_data` | Empty input handling | ✅ |
| `test_compute_returns` | Return calculation | ✅ |
| `test_compute_volatility` | Volatility calculation | ✅ |
| `test_compute_momentum` | Momentum calculation | ✅ |
| `test_compute_rsi` | RSI calculation | ✅ |
| **SentimentAgent** | | |
| `test_sentiment_agent_generates_synthetic_data` | Synthetic generation | ✅ |
| `test_sentiment_agent_is_deterministic_with_seed` | Seed determinism | ✅ |
| `test_sentiment_agent_returns_empty_when_disabled` | Disabled handling | ✅ |
| `test_compute_sentiment_score` | Composite score | ✅ |
| **FundamentalsAgent** | | |
| `test_compute_fundamentals_score` | Composite score | ✅ |
| **PriceHistoryAgent** | | |
| `test_get_close_prices` | Pivot extraction | ✅ |
| `test_get_prices_for_ticker` | Single ticker | ✅ |

### 3. ACO Tests (`test_aco.py`)

| Test | Description | Status |
|------|-------------|--------|
| **Portfolio** | | |
| `test_portfolio_creation` | Basic instantiation | ✅ |
| `test_portfolio_filters_negligible_weights` | Weight filtering | ✅ |
| `test_portfolio_normalization` | Sum to 1 | ✅ |
| `test_portfolio_min_threshold` | Min threshold | ✅ |
| `test_portfolio_top_holdings` | Top N extraction | ✅ |
| `test_portfolio_to_dict` | Serialization | ✅ |
| **ConstraintChecker** | | |
| `test_sum_weights_constraint_pass` | Valid sum | ✅ |
| `test_sum_weights_constraint_fail` | Invalid sum | ✅ |
| `test_max_weight_constraint_pass` | Within limits | ✅ |
| `test_holdings_count_constraint_pass` | Valid count | ✅ |
| `test_holdings_count_too_few` | Too few holdings | ✅ |
| `test_is_feasible` | Feasibility check | ✅ |
| **FitnessCalculator** | | |
| `test_calculate_fitness` | Fitness computation | ✅ |
| `test_portfolio_returns_calculation` | Weighted returns | ✅ |
| **ACOOptimizer** | | |
| `test_optimize_produces_valid_portfolio` | Valid output | ✅ |
| `test_optimize_respects_max_weight` | Max weight constraint | ✅ |
| `test_optimize_is_deterministic_with_seed` | Seed determinism | ✅ |
| `test_compute_heuristic` | Heuristic values | ✅ |
| `test_construct_portfolio` | Single construction | ✅ |
| `test_optimizer_handles_empty_features` | Empty input | ✅ |

### 4. Backtest Tests (`test_backtest.py`)

| Test | Description | Status |
|------|-------------|--------|
| **Metrics** | | |
| `test_cumulative_return` | Cumulative calc | ✅ |
| `test_cumulative_return_empty` | Empty series | ✅ |
| `test_volatility` | Volatility calc | ✅ |
| `test_volatility_is_annualized` | Annualization | ✅ |
| `test_sharpe_ratio` | Sharpe calc | ✅ |
| `test_sharpe_ratio_with_risk_free` | With rf rate | ✅ |
| `test_max_drawdown` | Drawdown calc | ✅ |
| `test_max_drawdown_with_known_values` | Known values | ✅ |
| `test_turnover_complete_change` | 100% turnover | ✅ |
| `test_turnover_no_change` | 0% turnover | ✅ |
| `test_turnover_partial_change` | Partial change | ✅ |
| `test_calculate_all_metrics` | All metrics | ✅ |
| `test_calculate_all_metrics_with_benchmark` | With benchmark | ✅ |
| **Integration** | | |
| `test_metrics_are_reasonable` | Sanity checks | ✅ |
| `test_metrics_detect_poor_performance` | Bad performance | ✅ |

## Test Infrastructure

### Fixtures (`conftest.py`)

| Fixture | Description |
|---------|-------------|
| `test_config` | Complete test configuration |
| `mock_logger` | Mocked logger |
| `test_context` | RunContext for testing |
| `sample_tickers` | 10 sample tickers |
| `sample_price_data` | Synthetic OHLCV data |
| `sample_close_prices` | Pivot of close prices |
| `sample_returns` | Calculated returns |
| `sample_features` | Synthetic features |
| `sample_agent_result` | Example AgentResult |
| `mock_price_response` | Mock API response |
| `mock_fundamentals_response` | Mock API response |
| `no_network` | Blocks all socket calls |

### Network Blocking

```python
@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Block all network calls in tests."""
    def guard(*args, **kwargs):
        raise RuntimeError("Network calls are not allowed in tests!")
    monkeypatch.setattr(socket, 'socket', guard)
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_aco.py -v

# With coverage
pytest tests/ -v --cov=src/app --cov-report=html

# Specific test
pytest tests/test_aco.py::TestACOOptimizer::test_optimize_is_deterministic_with_seed -v
```

## Coverage Target

- **Operator**: >90%
- **Agents**: >85%
- **ACO**: >90%
- **Metrics**: >95%
- **Overall**: >85%
