"""
Walk-forward backtesting engine.
"""

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..context import RunContext
from ..operator import Operator
from ..optimization import ACOOptimizer, Portfolio
from ..utils import format_date, get_trading_days, get_trading_days_back, iter_monthly_dates
from .metrics import (
    calculate_all_metrics,
    calculate_cumulative_return,
    calculate_turnover,
)


@dataclass
class WindowResult:
    """Result from a single backtest window."""
    train_end: date
    test_start: date
    test_end: date
    portfolio: Portfolio
    test_return: float
    benchmark_return: float
    turnover: float
    metadata: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Complete backtest results."""
    window_results: list[WindowResult]
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    equal_weight_returns: pd.Series
    metrics: dict
    baseline_metrics: dict
    config: dict
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            'metrics': self.metrics,
            'baseline_metrics': self.baseline_metrics,
            'num_windows': len(self.window_results),
            'config': self.config,
            'windows': [
                {
                    'train_end': format_date(w.train_end),
                    'test_start': format_date(w.test_start),
                    'test_end': format_date(w.test_end),
                    'test_return': w.test_return,
                    'benchmark_return': w.benchmark_return,
                    'turnover': w.turnover,
                    'num_holdings': w.portfolio.num_holdings,
                    'top_holdings': w.portfolio.top_holdings[:5],
                }
                for w in self.window_results
            ]
        }
    
    def save(self, output_dir: str, filename: str = 'backtest_results.json') -> Path:
        """Save results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / filename
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        return filepath
    
    def print_summary(self) -> None:
        """Print a summary of backtest results."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"\nWindows: {len(self.window_results)}")
        print(f"Period: {format_date(self.window_results[0].test_start)} to "
              f"{format_date(self.window_results[-1].test_end)}")
        
        print("\n--- ACO Portfolio ---")
        for key, value in self.metrics.items():
            if isinstance(value, float):
                if 'return' in key or 'ratio' in key:
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value:.2%}" if value < 1 else f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n--- Baseline Comparison ---")
        for baseline_name, baseline_metrics in self.baseline_metrics.items():
            print(f"\n  {baseline_name}:")
            for key in ['cumulative_return', 'sharpe_ratio', 'max_drawdown']:
                if key in baseline_metrics:
                    value = baseline_metrics[key]
                    print(f"    {key}: {value:.4f}")
        
        print("\n" + "=" * 60)


class WalkForwardBacktester:
    """
    Walk-forward backtesting engine.
    
    Performs rolling optimization with train/test windows:
    1. Train on historical data up to train_end
    2. Optimize portfolio using ACO
    3. Test on subsequent test_window period
    4. Roll forward and repeat
    """
    
    def __init__(self, ctx: RunContext, config: BacktestConfig):
        self.ctx = ctx
        self.config = config
        self.logger = ctx.logger
    
    async def run(
        self,
        universe: list[str],
        start_date: date,
        end_date: date
    ) -> BacktestResult:
        """
        Run walk-forward backtest.
        
        Args:
            universe: List of tickers to include
            start_date: Backtest start date
            end_date: Backtest end date
        
        Returns:
            BacktestResult with all metrics and window details
        """
        self.logger.info(
            f"Starting walk-forward backtest from {start_date} to {end_date} "
            f"with {len(universe)} tickers"
        )
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(start_date, end_date)
        self.logger.info(f"Generated {len(rebalance_dates)} rebalance dates")
        
        if len(rebalance_dates) < 2:
            raise ValueError("Insufficient date range for backtesting")
        
        # Initialize operator
        operator = Operator(self.ctx)
        
        # Fetch all historical data upfront for the full period
        # This is more efficient than fetching per-window
        self.logger.info("Fetching historical data for full backtest period...")
        
        # Extend start to include training window
        data_start = start_date - timedelta(days=self.config.train_window_days + 50)
        
        self.ctx.start_date = data_start
        self.ctx.end_date = end_date
        
        operator_result = await operator.run_cycle(universe)
        
        if operator_result.price_data.empty:
            raise ValueError("No price data available for backtesting")
        
        # Get close prices as pivot table
        from ..agents import PriceHistoryAgent
        price_agent = PriceHistoryAgent()
        close_prices = price_agent.get_close_prices(operator_result.price_data)
        
        if close_prices.empty:
            raise ValueError("Could not extract close prices")
        
        # Calculate returns
        returns = close_prices.pct_change().dropna()
        
        # Run walk-forward
        window_results = []
        prev_weights: dict[str, float] = {}
        
        for i in range(len(rebalance_dates) - 1):
            train_end = rebalance_dates[i]
            test_start = train_end + timedelta(days=1)
            test_end = rebalance_dates[i + 1]
            
            self.logger.info(
                f"Window {i + 1}/{len(rebalance_dates) - 1}: "
                f"train ends {train_end}, test {test_start} to {test_end}"
            )
            
            # Get training data — use trading days, not calendar days
            train_start = get_trading_days_back(train_end, self.config.train_window_days)
            train_returns = self._filter_returns(returns, train_start, train_end)
            
            if train_returns.empty or len(train_returns) < 20:
                self.logger.warning(f"Insufficient training data for window {i + 1}")
                continue
            
            # Filter to tickers with training data
            available_tickers = [
                t for t in universe
                if t in train_returns.columns and train_returns[t].notna().sum() > 20
            ]
            train_returns = train_returns[available_tickers]
            
            # Recompute features from ONLY train-period prices to avoid look-ahead bias
            features = self._compute_window_features(
                operator_result, available_tickers, train_start, train_end
            )
            
            if len(features) < self.config.min_holdings_for_backtest:
                self.logger.warning(
                    f"Only {len(features)} tickers available, skipping window"
                )
                continue
            
            # Optimize portfolio
            portfolio = await self._optimize_for_window(
                features, train_returns, available_tickers
            )
            
            # Get test period returns
            test_returns = self._filter_returns(returns, test_start, test_end)
            
            if test_returns.empty:
                self.logger.warning(f"No test returns for window {i + 1}")
                continue
            
            # Calculate portfolio return for test period
            portfolio_test_returns = self._calculate_portfolio_returns(
                portfolio.weights, test_returns
            )
            test_return = calculate_cumulative_return(portfolio_test_returns)
            
            # Calculate benchmark return (equal weight)
            equal_weight = {t: 1.0 / len(available_tickers) for t in available_tickers}
            benchmark_returns = self._calculate_portfolio_returns(
                equal_weight, test_returns
            )
            benchmark_return = calculate_cumulative_return(benchmark_returns)
            
            # Calculate turnover
            turnover = calculate_turnover(prev_weights, portfolio.weights)
            prev_weights = portfolio.weights.copy()
            
            window_results.append(WindowResult(
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                portfolio=portfolio,
                test_return=test_return,
                benchmark_return=benchmark_return,
                turnover=turnover,
                metadata={'num_holdings': portfolio.num_holdings}
            ))
        
        if not window_results:
            raise ValueError("No valid backtest windows completed")
        
        # Aggregate results
        return self._aggregate_results(window_results, returns, universe)
    
    def _get_rebalance_dates(
        self,
        start_date: date,
        end_date: date
    ) -> list[date]:
        """Get rebalance dates based on frequency."""
        if self.config.rebalance_frequency == 'monthly':
            return list(iter_monthly_dates(start_date, end_date))
        else:
            # Default to monthly
            return list(iter_monthly_dates(start_date, end_date))
    
    def _filter_returns(
        self,
        returns: pd.DataFrame,
        start: date,
        end: date
    ) -> pd.DataFrame:
        """Filter returns DataFrame to date range."""
        mask = (returns.index >= start) & (returns.index <= end)
        return returns.loc[mask]
    
    def _compute_window_features(
        self,
        operator_result,
        tickers: list[str],
        train_start: date,
        train_end: date
    ) -> 'pd.DataFrame':
        """
        Recompute features from train-period data only to avoid look-ahead bias.
        
        Technical indicators are recomputed from the train slice of prices.
        Fundamentals (which are point-in-time) are taken from the operator result.
        """
        from ..agents import TechnicalIndicatorAgent
        
        # Start with fundamentals from the operator (point-in-time, no look-ahead issue)
        base_features = operator_result.features.copy()
        # Keep only fundamental columns
        fund_cols = [c for c in base_features.columns if c.startswith('fund_')]
        features = base_features.loc[
            base_features.index.isin(tickers), fund_cols
        ].copy()
        
        # Recompute technical indicators from train-period prices only
        price_data = operator_result.price_data
        if not price_data.empty:
            tech_agent = TechnicalIndicatorAgent()
            for ticker in tickers:
                if ticker not in price_data.index.get_level_values('ticker'):
                    continue
                ticker_prices = price_data.loc[ticker].copy()
                # Filter to train period only
                mask = (ticker_prices.index >= train_start) & (ticker_prices.index <= train_end)
                ticker_train = ticker_prices.loc[mask]
                if len(ticker_train) < 21:
                    continue
                close = ticker_train['close'].sort_index()
                # Compute each indicator from train data
                for indicator in ['returns_1m', 'returns_3m', 'volatility_21d', 'momentum_200d', 'rsi_14']:
                    val = tech_agent._compute_indicator(indicator, close, ticker_train)
                    if val is not None:
                        features.loc[ticker, indicator] = val
        
        # Drop tickers that ended up with no indicators
        if 'returns_1m' in features.columns:
            features = features.dropna(subset=['returns_1m'])
        
        return features

    async def _optimize_for_window(
        self,
        features: pd.DataFrame,
        train_returns: pd.DataFrame,
        tickers: list[str]
    ) -> Portfolio:
        """Run ACO optimization for a single window."""
        optimizer = ACOOptimizer(
            self.ctx.config.aco,
            self.ctx.config.constraints,
            rng=np.random.default_rng(self.ctx.random_seed)
        )
        
        # Get sectors if available
        sectors = {}
        if 'fund_sector' in features.columns:
            sectors = features['fund_sector'].to_dict()
        
        portfolio = optimizer.optimize(features, train_returns, sectors)
        
        return portfolio
    
    def _calculate_portfolio_returns(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame
    ) -> pd.Series:
        """Calculate weighted portfolio returns."""
        common_tickers = [t for t in weights.keys() if t in returns.columns]
        
        if not common_tickers:
            return pd.Series(dtype=float)
        
        # Normalize weights
        total = sum(weights[t] for t in common_tickers)
        if total == 0:
            return pd.Series(dtype=float)
        
        norm_weights = {t: weights[t] / total for t in common_tickers}
        
        # Calculate weighted sum
        portfolio_returns = sum(
            returns[t] * w for t, w in norm_weights.items()
        )
        
        return portfolio_returns.dropna()
    
    def _aggregate_results(
        self,
        window_results: list[WindowResult],
        returns: pd.DataFrame,
        universe: list[str]
    ) -> BacktestResult:
        """Aggregate window results into final backtest result."""
        # Combine all test period returns
        all_portfolio_returns = []
        all_benchmark_returns = []
        all_equal_weight_returns = []
        
        for window in window_results:
            test_returns = self._filter_returns(
                returns, window.test_start, window.test_end
            )
            
            # Portfolio returns
            portfolio_rets = self._calculate_portfolio_returns(
                window.portfolio.weights, test_returns
            )
            all_portfolio_returns.append(portfolio_rets)
            
            # Equal weight returns
            available = [t for t in universe if t in test_returns.columns]
            if available:
                equal_weights = {t: 1.0 / len(available) for t in available}
                eq_rets = self._calculate_portfolio_returns(equal_weights, test_returns)
                all_equal_weight_returns.append(eq_rets)
        
        # Concatenate return series
        portfolio_returns = pd.concat(all_portfolio_returns) if all_portfolio_returns else pd.Series(dtype=float)
        equal_weight_returns = pd.concat(all_equal_weight_returns) if all_equal_weight_returns else pd.Series(dtype=float)
        
        # For SPY benchmark, use equal weight as proxy if SPY not in universe
        if 'SPY' in returns.columns:
            spy_mask = (returns.index >= window_results[0].test_start) & \
                       (returns.index <= window_results[-1].test_end)
            benchmark_returns = returns.loc[spy_mask, 'SPY']
        else:
            benchmark_returns = equal_weight_returns
        
        # Calculate metrics
        metrics = calculate_all_metrics(
            portfolio_returns,
            benchmark_returns,
            self.config.risk_free_rate
        )
        
        # Add average turnover
        turnovers = [w.turnover for w in window_results]
        metrics['avg_turnover'] = np.mean(turnovers) if turnovers else 0.0
        
        # Calculate baseline metrics
        baseline_metrics = {
            'equal_weight': calculate_all_metrics(
                equal_weight_returns,
                benchmark_returns,
                self.config.risk_free_rate
            )
        }
        
        if 'SPY' in returns.columns:
            baseline_metrics['spy_benchmark'] = {
                'cumulative_return': calculate_cumulative_return(benchmark_returns),
                'sharpe_ratio': 0.0,  # Would need to calculate
                'max_drawdown': 0.0
            }
        
        return BacktestResult(
            window_results=window_results,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            equal_weight_returns=equal_weight_returns,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            config={
                'train_window_days': self.config.train_window_days,
                'test_window_days': self.config.test_window_days,
                'rebalance_frequency': self.config.rebalance_frequency,
                'num_windows': len(window_results)
            }
        )
    
    @property
    def min_holdings_for_backtest(self) -> int:
        """Minimum holdings needed for a valid backtest window."""
        return getattr(self.config, 'min_holdings_for_backtest', 5)


# Add property to BacktestConfig
BacktestConfig.min_holdings_for_backtest = 5
