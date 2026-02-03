"""
Tests for ACO optimization.
"""

import numpy as np
import pandas as pd
import pytest

from src.app.optimization import (
    ACOOptimizer,
    ConstraintChecker,
    FitnessCalculator,
    Portfolio,
)


class TestPortfolio:
    """Tests for Portfolio class."""
    
    def test_portfolio_creation(self):
        """Test creating a portfolio."""
        weights = {'AAPL': 0.3, 'MSFT': 0.3, 'GOOGL': 0.4}
        portfolio = Portfolio(weights, fitness=1.5)
        
        assert portfolio.num_holdings == 3
        assert abs(portfolio.total_weight - 1.0) < 0.001
        assert portfolio.max_weight == 0.4
    
    def test_portfolio_filters_negligible_weights(self):
        """Test that weights below threshold are filtered."""
        weights = {'AAPL': 0.5, 'MSFT': 0.0005, 'GOOGL': 0.499}
        portfolio = Portfolio(weights)
        
        # 0.0005 should be filtered out (< 0.001)
        assert 'MSFT' not in portfolio.weights
        assert portfolio.num_holdings == 2
    
    def test_portfolio_normalization(self):
        """Test portfolio normalization."""
        weights = {'AAPL': 0.3, 'MSFT': 0.3}  # Total = 0.6
        portfolio = Portfolio(weights)
        
        normalized = portfolio.normalize()
        
        assert abs(normalized.total_weight - 1.0) < 0.001
        assert abs(normalized.weights['AAPL'] - 0.5) < 0.001
    
    def test_portfolio_min_threshold(self):
        """Test applying minimum weight threshold."""
        weights = {'AAPL': 0.5, 'MSFT': 0.03, 'GOOGL': 0.47}
        portfolio = Portfolio(weights)
        
        filtered = portfolio.apply_min_threshold(0.05)
        
        assert 'MSFT' not in filtered.weights
        assert filtered.num_holdings == 2
    
    def test_portfolio_top_holdings(self):
        """Test top holdings property."""
        weights = {
            'AAPL': 0.3, 'MSFT': 0.25, 'GOOGL': 0.2,
            'AMZN': 0.15, 'META': 0.1
        }
        portfolio = Portfolio(weights)
        
        top = portfolio.top_holdings
        
        assert len(top) == 5
        assert top[0][0] == 'AAPL'
        assert top[0][1] == 0.3
    
    def test_portfolio_to_dict(self):
        """Test portfolio serialization."""
        weights = {'AAPL': 0.6, 'MSFT': 0.4}
        portfolio = Portfolio(weights, fitness=1.0, diagnostics={'test': True})
        
        d = portfolio.to_dict()
        
        assert 'weights' in d
        assert 'fitness' in d
        assert 'num_holdings' in d
        assert d['num_holdings'] == 2


class TestConstraintChecker:
    """Tests for ConstraintChecker."""
    
    @pytest.fixture
    def constraint_checker(self, test_config):
        """Create a constraint checker."""
        return ConstraintChecker(test_config.constraints)
    
    def test_sum_weights_constraint_pass(self, constraint_checker):
        """Test sum weights constraint - passing."""
        weights = {'AAPL': 0.5, 'MSFT': 0.5}
        
        penalty, violation = constraint_checker.check_sum_weights(weights)
        
        assert penalty == 0.0
        assert violation is None
    
    def test_sum_weights_constraint_fail(self, constraint_checker):
        """Test sum weights constraint - failing."""
        weights = {'AAPL': 0.3, 'MSFT': 0.3}  # Total = 0.6
        
        penalty, violation = constraint_checker.check_sum_weights(weights)
        
        assert penalty > 0
        assert violation is not None
        assert 'sum' in violation.message.lower()
    
    def test_max_weight_constraint_pass(self, constraint_checker):
        """Test max weight constraint - passing."""
        weights = {'AAPL': 0.15, 'MSFT': 0.15, 'GOOGL': 0.70}
        # max_weight is 0.20, but GOOGL exceeds it
        
        penalty, violations = constraint_checker.check_max_weights(weights)
        
        # GOOGL should violate
        assert penalty > 0
        assert len(violations) > 0
    
    def test_holdings_count_constraint_pass(self, constraint_checker):
        """Test holdings count constraint - passing."""
        weights = {'AAPL': 0.2, 'MSFT': 0.2, 'GOOGL': 0.2, 'AMZN': 0.2, 'META': 0.2}
        
        penalty, violation = constraint_checker.check_holdings_count(weights)
        
        assert penalty == 0.0
        assert violation is None
    
    def test_holdings_count_too_few(self, constraint_checker):
        """Test holdings count constraint - too few."""
        weights = {'AAPL': 0.5, 'MSFT': 0.5}  # Only 2, min is 3
        
        penalty, violation = constraint_checker.check_holdings_count(weights)
        
        assert penalty > 0
        assert violation is not None
    
    def test_is_feasible(self, constraint_checker):
        """Test feasibility check."""
        # Feasible portfolio
        weights = {'AAPL': 0.2, 'MSFT': 0.2, 'GOOGL': 0.2, 'AMZN': 0.2, 'META': 0.2}
        assert constraint_checker.is_feasible(weights)
        
        # Infeasible (doesn't sum to 1)
        bad_weights = {'AAPL': 0.3, 'MSFT': 0.3}
        assert not constraint_checker.is_feasible(bad_weights)


class TestFitnessCalculator:
    """Tests for FitnessCalculator."""
    
    @pytest.fixture
    def fitness_calculator(self, test_config):
        """Create a fitness calculator."""
        checker = ConstraintChecker(test_config.constraints)
        return FitnessCalculator(checker)
    
    def test_calculate_fitness(self, fitness_calculator, sample_returns):
        """Test fitness calculation."""
        weights = {t: 0.1 for t in sample_returns.columns}
        
        fitness, diagnostics = fitness_calculator.calculate(weights, sample_returns)
        
        assert isinstance(fitness, float)
        assert 'annualized_return' in diagnostics
        assert 'annualized_volatility' in diagnostics
        assert 'sharpe_ratio' in diagnostics
    
    def test_portfolio_returns_calculation(
        self,
        fitness_calculator,
        sample_returns
    ):
        """Test weighted portfolio returns."""
        weights = {'AAPL': 0.5, 'MSFT': 0.5}
        
        returns = fitness_calculator._calculate_portfolio_returns(
            weights, sample_returns
        )
        
        assert not returns.empty
        # Should be approximately average of two tickers
        expected = (sample_returns['AAPL'] + sample_returns['MSFT']) / 2
        pd.testing.assert_series_equal(
            returns.round(6),
            expected.dropna().round(6),
            check_names=False
        )


class TestACOOptimizer:
    """Tests for ACOOptimizer."""
    
    @pytest.fixture
    def aco_optimizer(self, test_config):
        """Create an ACO optimizer."""
        return ACOOptimizer(
            test_config.aco,
            test_config.constraints,
            rng=np.random.default_rng(42)
        )
    
    def test_optimize_produces_valid_portfolio(
        self,
        aco_optimizer,
        sample_features,
        sample_returns
    ):
        """Test that optimization produces a valid portfolio."""
        portfolio = aco_optimizer.optimize(sample_features, sample_returns)
        
        assert isinstance(portfolio, Portfolio)
        assert portfolio.num_holdings >= 0
        
        # Weights should sum to approximately 1
        if portfolio.num_holdings > 0:
            assert abs(portfolio.total_weight - 1.0) < 0.05
    
    def test_optimize_respects_max_weight(
        self,
        aco_optimizer,
        sample_features,
        sample_returns
    ):
        """Test that no weight exceeds maximum."""
        portfolio = aco_optimizer.optimize(sample_features, sample_returns)
        
        max_allowed = aco_optimizer.constraints.max_weight_per_ticker
        
        for ticker, weight in portfolio.weights.items():
            assert weight <= max_allowed + 0.01  # Small tolerance
    
    def test_optimize_is_deterministic_with_seed(
        self,
        test_config,
        sample_features,
        sample_returns
    ):
        """Test that same seed produces same results."""
        optimizer1 = ACOOptimizer(
            test_config.aco,
            test_config.constraints,
            rng=np.random.default_rng(42)
        )
        
        optimizer2 = ACOOptimizer(
            test_config.aco,
            test_config.constraints,
            rng=np.random.default_rng(42)
        )
        
        portfolio1 = optimizer1.optimize(sample_features, sample_returns)
        portfolio2 = optimizer2.optimize(sample_features, sample_returns)
        
        # Same weights (with tolerance for floating point)
        assert set(portfolio1.weights.keys()) == set(portfolio2.weights.keys())
        for ticker in portfolio1.weights:
            assert abs(portfolio1.weights[ticker] - portfolio2.weights[ticker]) < 0.001
    
    def test_compute_heuristic(self, aco_optimizer, sample_features):
        """Test heuristic computation."""
        aco_optimizer.tickers = list(sample_features.index)
        
        heuristic = aco_optimizer._compute_heuristic(sample_features)
        
        assert len(heuristic) == len(sample_features)
        assert all(h > 0 for h in heuristic)
    
    def test_construct_portfolio(self, aco_optimizer, sample_features, sample_returns):
        """Test single portfolio construction."""
        aco_optimizer.tickers = list(sample_features.index)
        aco_optimizer.pheromone = np.ones(
            (len(sample_features), aco_optimizer.max_units + 1)
        )
        aco_optimizer.heuristic = np.ones(len(sample_features))
        
        weights = aco_optimizer._construct_portfolio()
        
        assert isinstance(weights, dict)
        assert sum(weights.values()) <= 1.0 + 0.01
    
    def test_optimizer_handles_empty_features(self, aco_optimizer, sample_returns):
        """Test handling of empty features."""
        portfolio = aco_optimizer.optimize(pd.DataFrame(), sample_returns)
        
        assert portfolio.num_holdings == 0
        assert 'error' in portfolio.diagnostics
