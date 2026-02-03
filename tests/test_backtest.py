"""
Tests for backtesting functionality.
"""

import numpy as np
import pandas as pd
import pytest

from src.app.backtesting import (
    calculate_all_metrics,
    calculate_cumulative_return,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_turnover,
    calculate_volatility,
)


class TestMetrics:
    """Tests for individual metric calculations."""
    
    @pytest.fixture
    def sample_return_series(self):
        """Create a sample return series."""
        np.random.seed(42)
        returns = pd.Series(
            np.random.normal(0.001, 0.02, 252),  # ~25% annual return, 32% vol
            index=pd.date_range('2023-01-01', periods=252, freq='B')
        )
        return returns
    
    def test_cumulative_return(self, sample_return_series):
        """Test cumulative return calculation."""
        cum_ret = calculate_cumulative_return(sample_return_series)
        
        # Should match prod of (1+r) - 1
        expected = (1 + sample_return_series).prod() - 1
        
        assert abs(cum_ret - expected) < 1e-10
    
    def test_cumulative_return_empty(self):
        """Test cumulative return with empty series."""
        result = calculate_cumulative_return(pd.Series(dtype=float))
        assert result == 0.0
    
    def test_volatility(self, sample_return_series):
        """Test volatility calculation."""
        vol = calculate_volatility(sample_return_series)
        
        # Should be annualized (daily std * sqrt(252))
        expected = sample_return_series.std() * np.sqrt(252)
        
        assert abs(vol - expected) < 1e-10
    
    def test_volatility_is_annualized(self, sample_return_series):
        """Test that volatility is properly annualized."""
        vol = calculate_volatility(sample_return_series, periods_per_year=252)
        
        # Annualized vol should be approximately sqrt(252) * daily vol
        daily_vol = sample_return_series.std()
        expected = daily_vol * np.sqrt(252)
        
        assert abs(vol - expected) < 1e-6
    
    def test_sharpe_ratio(self, sample_return_series):
        """Test Sharpe ratio calculation."""
        sharpe = calculate_sharpe_ratio(sample_return_series, risk_free_rate=0.0)
        
        assert isinstance(sharpe, float)
        # Sharpe should be reasonable (between -5 and 5 typically)
        assert -5 < sharpe < 5
    
    def test_sharpe_ratio_with_risk_free(self, sample_return_series):
        """Test Sharpe ratio with non-zero risk-free rate."""
        sharpe_0 = calculate_sharpe_ratio(sample_return_series, risk_free_rate=0.0)
        sharpe_5 = calculate_sharpe_ratio(sample_return_series, risk_free_rate=0.05)
        
        # Higher risk-free rate should lower Sharpe
        assert sharpe_5 < sharpe_0
    
    def test_max_drawdown(self, sample_return_series):
        """Test max drawdown calculation."""
        mdd = calculate_max_drawdown(sample_return_series)
        
        assert mdd >= 0  # Drawdown is positive
        assert mdd <= 1  # Cannot lose more than 100%
    
    def test_max_drawdown_with_known_values(self):
        """Test max drawdown with known values."""
        # Create a series that goes up 50%, then down 30%
        returns = pd.Series([0.5, -0.3])
        
        mdd = calculate_max_drawdown(returns)
        
        # Peak is 1.5, trough is 1.5 * 0.7 = 1.05
        # Drawdown = (1.5 - 1.05) / 1.5 = 0.3
        assert abs(mdd - 0.3) < 0.001
    
    def test_turnover_complete_change(self):
        """Test turnover with complete portfolio change."""
        prev = {'AAPL': 0.5, 'MSFT': 0.5}
        curr = {'GOOGL': 0.5, 'AMZN': 0.5}
        
        turnover = calculate_turnover(prev, curr)
        
        # Complete change = 100% turnover
        assert abs(turnover - 1.0) < 0.001
    
    def test_turnover_no_change(self):
        """Test turnover with no change."""
        weights = {'AAPL': 0.5, 'MSFT': 0.5}
        
        turnover = calculate_turnover(weights, weights)
        
        assert turnover == 0.0
    
    def test_turnover_partial_change(self):
        """Test turnover with partial change."""
        prev = {'AAPL': 0.5, 'MSFT': 0.5}
        curr = {'AAPL': 0.5, 'GOOGL': 0.5}  # MSFT replaced by GOOGL
        
        turnover = calculate_turnover(prev, curr)
        
        # MSFT: -0.5, GOOGL: +0.5, total change = 1.0, turnover = 0.5
        assert abs(turnover - 0.5) < 0.001
    
    def test_calculate_all_metrics(self, sample_return_series):
        """Test calculating all metrics at once."""
        metrics = calculate_all_metrics(sample_return_series)
        
        assert 'cumulative_return' in metrics
        assert 'annualized_return' in metrics
        assert 'annualized_volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'num_periods' in metrics
    
    def test_calculate_all_metrics_with_benchmark(self, sample_return_series):
        """Test calculating metrics with benchmark comparison."""
        np.random.seed(43)
        benchmark = pd.Series(
            np.random.normal(0.0005, 0.015, 252),
            index=sample_return_series.index
        )
        
        metrics = calculate_all_metrics(sample_return_series, benchmark)
        
        assert 'information_ratio' in metrics
        assert 'benchmark_cumulative_return' in metrics
        assert 'excess_return' in metrics


class TestBacktestIntegration:
    """Integration tests for backtesting."""
    
    def test_metrics_are_reasonable(self):
        """Test that metrics produce reasonable values for known data."""
        # Create a steadily increasing return series
        np.random.seed(42)
        good_returns = pd.Series(
            np.random.normal(0.002, 0.01, 252),  # 50% annual, low vol
            index=pd.date_range('2023-01-01', periods=252, freq='B')
        )
        
        metrics = calculate_all_metrics(good_returns)
        
        # Good returns should have positive Sharpe
        assert metrics['sharpe_ratio'] > 0
        
        # Cumulative return should be significantly positive
        assert metrics['cumulative_return'] > 0
    
    def test_metrics_detect_poor_performance(self):
        """Test that metrics correctly identify poor performance."""
        # Create a declining return series
        np.random.seed(42)
        bad_returns = pd.Series(
            np.random.normal(-0.002, 0.01, 252),  # -50% annual
            index=pd.date_range('2023-01-01', periods=252, freq='B')
        )
        
        metrics = calculate_all_metrics(bad_returns)
        
        # Bad returns should have negative Sharpe
        assert metrics['sharpe_ratio'] < 0
        
        # Cumulative return should be negative
        assert metrics['cumulative_return'] < 0
        
        # Should have meaningful drawdown
        assert metrics['max_drawdown'] > 0
