# Test Report

## Summary

| Category | Tests | Passed | Failed | Skipped |
|----------|-------|--------|--------|---------|
| Operator | 6 | TBD | TBD | 0 |
| Agents | 18 | TBD | TBD | 0 |
| ACO | 18 | TBD | TBD | 0 |
| Backtest | 15 | TBD | TBD | 0 |
| **Total** | **57** | **TBD** | **TBD** | **0** |

## Execution

```
Date: [Pending execution]
Python: 3.11+
Platform: Windows
Command: pytest tests/ -v
```

## Known Limitations

1. **Walk-Forward Integration Test**: The full walk-forward backtest integration test is not included due to complexity of mocking the complete data pipeline. The individual components (metrics, operators, ACO) are tested thoroughly.

2. **Provider Integration**: Real API providers are not tested against live endpoints. Mocking validates interface compliance.

3. **Large Universe Performance**: Performance testing with 500+ tickers not included in unit tests.

## Coverage Report

```
[Pending coverage run]

Expected coverage breakdown:
- src/app/operator.py: >90%
- src/app/agents/: >85%
- src/app/optimization/: >90%
- src/app/backtesting/metrics.py: >95%
```

## Running Tests

To generate a fresh test report:

```bash
# Run tests with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/app --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ -v --cov=src/app --cov-report=html
```

## Test Infrastructure Validation

- [x] Network calls blocked (`no_network` fixture)
- [x] Deterministic fixtures with fixed seeds
- [x] All providers mocked
- [x] No external dependencies in test execution
