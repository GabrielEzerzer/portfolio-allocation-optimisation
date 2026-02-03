"""
Pytest configuration and fixtures.
All tests run offline - no network calls allowed.
"""

import asyncio
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.app.agents import AgentResult
from src.app.config import (
    ACOConfig,
    AgentConfig,
    BacktestConfig,
    CacheConfig,
    Config,
    ConstraintsConfig,
    FeaturesConfig,
    LoggingConfig,
    OutputConfig,
    ProviderConfig,
)
from src.app.context import RunContext


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config() -> Config:
    """Create a test configuration with sensible defaults."""
    config = Config(
        mode='test',
        primary_provider='alphavantage',
        providers={
            'alphavantage': ProviderConfig(
                base_url='https://www.alphavantage.co/query',
                api_key='TEST_KEY',
                rate_limit_per_minute=100,
                timeout_seconds=5,
                retries=1,
                backoff_base=1.0,
                enabled=True
            )
        },
        agents={
            'universe': AgentConfig(
                enabled=True,
                timeout_seconds=10,
                fallback_file='data/universe/sp500_seed.csv'
            ),
            'price': AgentConfig(
                enabled=True,
                timeout_seconds=30,
                history_days=100
            ),
            'technical': AgentConfig(
                enabled=True,
                timeout_seconds=10,
                indicators=['returns_1m', 'volatility_21d', 'momentum_200d']
            ),
            'fundamentals': AgentConfig(
                enabled=True,
                timeout_seconds=30,
                fields=['market_cap', 'pe_ratio', 'eps', 'sector']
            ),
            'sentiment': AgentConfig(
                enabled=False,
                timeout_seconds=10
            )
        },
        cache=CacheConfig(
            enabled=False,
            path='data/cache',
            refresh_days=1,
            format='parquet'
        ),
        aco=ACOConfig(
            num_ants=5,
            num_iterations=10,
            evaporation_rate=0.3,
            alpha=1.0,
            beta=2.0,
            weight_granularity=0.10,
            random_seed=42,
            heuristic_weights={
                'momentum': 0.3,
                'returns': 0.3,
                'inverse_volatility': 0.2,
                'fundamentals_score': 0.2
            }
        ),
        constraints=ConstraintsConfig(
            max_weight_per_ticker=0.20,
            min_weight_threshold=0.05,
            min_holdings=3,
            max_holdings=10,
            sum_weights_tolerance=0.01,
            sector_cap=None
        ),
        backtest=BacktestConfig(
            train_window_days=60,
            test_window_days=20,
            rebalance_frequency='monthly',
            risk_free_rate=0.0,
            initial_capital=100000.0,
            transaction_cost=0.001,
            baselines=['equal_weight']
        ),
        features=FeaturesConfig(
            missing_value_strategy='fill_median',
            required_features=['returns_1m'],
            normalize=True,
            normalization_method='zscore'
        ),
        logging=LoggingConfig(
            level='WARNING',
            file='logs/test.log',
            format='%(message)s',
            max_bytes=1000000,
            backup_count=1
        ),
        output=OutputConfig(
            path='data/outputs/test',
            save_weights=False,
            save_metrics=False,
            save_diagnostics=False,
            format='json'
        )
    )
    return config


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    logger = MagicMock()
    return logger


@pytest.fixture
def test_context(test_config, mock_logger) -> RunContext:
    """Create a test RunContext with mocked components."""
    ctx = RunContext(
        config=test_config,
        logger=mock_logger,
        mode='test',
        session=None,  # No network in tests
        rate_limiter=None,
        cache_manager=None,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        random_seed=42,
        run_id='test-run-001'
    )
    return ctx


@pytest.fixture
def sample_tickers() -> list[str]:
    """Sample ticker list for testing."""
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'UNH']


@pytest.fixture
def sample_price_data(sample_tickers) -> pd.DataFrame:
    """Generate synthetic price data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    
    records = []
    for ticker in sample_tickers:
        # Generate random walk prices
        base_price = np.random.uniform(50, 500)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = base_price * np.cumprod(1 + returns)
        
        for i, d in enumerate(dates):
            records.append({
                'ticker': ticker,
                'date': d.date(),
                'open': prices[i] * np.random.uniform(0.99, 1.01),
                'high': prices[i] * np.random.uniform(1.0, 1.02),
                'low': prices[i] * np.random.uniform(0.98, 1.0),
                'close': prices[i],
                'volume': int(np.random.uniform(1e6, 1e8))
            })
    
    df = pd.DataFrame(records)
    df = df.set_index(['ticker', 'date']).sort_index()
    return df


@pytest.fixture
def sample_close_prices(sample_price_data) -> pd.DataFrame:
    """Extract close prices as pivot table."""
    df = sample_price_data.reset_index()
    return df.pivot(index='date', columns='ticker', values='close')


@pytest.fixture
def sample_returns(sample_close_prices) -> pd.DataFrame:
    """Calculate returns from prices."""
    return sample_close_prices.pct_change().dropna()


@pytest.fixture
def sample_features(sample_tickers) -> pd.DataFrame:
    """Generate synthetic features for testing."""
    np.random.seed(42)
    
    data = {
        'returns_1m': np.random.uniform(-0.1, 0.2, len(sample_tickers)),
        'returns_3m': np.random.uniform(-0.2, 0.3, len(sample_tickers)),
        'volatility_21d': np.random.uniform(0.1, 0.4, len(sample_tickers)),
        'momentum_200d': np.random.uniform(-0.1, 0.1, len(sample_tickers)),
        'fund_market_cap': np.random.uniform(1e9, 1e12, len(sample_tickers)),
        'fund_pe_ratio': np.random.uniform(10, 50, len(sample_tickers)),
        'fund_sector': np.random.choice(
            ['Technology', 'Financials', 'Healthcare', 'Consumer'],
            len(sample_tickers)
        )
    }
    
    df = pd.DataFrame(data, index=sample_tickers)
    df.index.name = 'ticker'
    return df


@pytest.fixture
def sample_agent_result(sample_features) -> AgentResult:
    """Create a sample AgentResult."""
    return AgentResult(
        name='test_agent',
        data=sample_features,
        metadata={
            'timestamp': '2023-12-31T00:00:00',
            'coverage': len(sample_features),
            'coverage_ratio': 1.0,
            'requested': len(sample_features),
            'latency_ms': 100,
            'errors': []
        }
    )


@pytest.fixture
def mock_price_response():
    """Mock response for price data API call."""
    return {
        'Time Series (Daily)': {
            '2023-12-29': {
                '1. open': '100.00',
                '2. high': '102.00',
                '3. low': '99.00',
                '4. close': '101.50',
                '5. volume': '10000000'
            },
            '2023-12-28': {
                '1. open': '99.00',
                '2. high': '101.00',
                '3. low': '98.50',
                '4. close': '100.00',
                '5. volume': '9000000'
            }
        }
    }


@pytest.fixture
def mock_fundamentals_response():
    """Mock response for fundamentals API call."""
    return {
        'Symbol': 'AAPL',
        'MarketCapitalization': '3000000000000',
        'PERatio': '30.5',
        'EPS': '6.10',
        'Sector': 'Technology',
        'Industry': 'Consumer Electronics',
        'DividendYield': '0.005',
        'BookValue': '4.00',
        'ProfitMargin': '0.25'
    }


# Network blocking is handled at the test level via mocked sessions.
# All async tests use mocked sessions/providers so no actual network calls are made.
# If needed, use pytest-socket plugin with:
#   pytest --disable-socket --allow-hosts=localhost,fd00::/8

# No autouse fixture needed - tests use mocked data


