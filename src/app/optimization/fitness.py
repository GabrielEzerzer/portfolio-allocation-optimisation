"""
Fitness function for portfolio optimization.
"""

import numpy as np
import pandas as pd

from .constraints import ConstraintChecker


class FitnessCalculator:
    """
    Calculates portfolio fitness for ACO optimization.
    
    Fitness is based on:
    - Risk-adjusted return (Sharpe-like ratio)
    - Constraint penalties
    """
    
    def __init__(
        self,
        constraint_checker: ConstraintChecker,
        risk_free_rate: float = 0.0
    ):
        self.constraint_checker = constraint_checker
        self.risk_free_rate = risk_free_rate
    
    def calculate(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame,
        sectors: dict[str, str] | None = None
    ) -> tuple[float, dict]:
        """
        Calculate portfolio fitness.
        
        Args:
            weights: Dict of ticker -> weight
            returns: DataFrame of daily returns (columns = tickers)
            sectors: Optional dict of ticker -> sector for sector constraints
        
        Returns:
            Tuple of (fitness_score, diagnostics)
        """
        diagnostics = {}
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(weights, returns)
        
        if portfolio_returns.empty or len(portfolio_returns) < 10:
            diagnostics['error'] = 'Insufficient return data'
            return -100.0, diagnostics
        
        # Calculate risk-adjusted metrics
        mean_return = portfolio_returns.mean()
        volatility = portfolio_returns.std()
        
        # Annualize
        annualized_return = mean_return * 252
        annualized_vol = volatility * np.sqrt(252)
        
        # Sharpe ratio (avoid division by zero)
        if annualized_vol > 0:
            sharpe = (annualized_return - self.risk_free_rate) / annualized_vol
        else:
            sharpe = 0.0
        
        diagnostics['annualized_return'] = annualized_return
        diagnostics['annualized_volatility'] = annualized_vol
        diagnostics['sharpe_ratio'] = sharpe
        
        # Check constraints
        penalty, violations = self.constraint_checker.check_all(weights, sectors)
        diagnostics['constraint_penalty'] = penalty
        diagnostics['violations'] = [v.message for v in violations]
        
        # Final fitness = Sharpe - penalty
        # Sharpe can be negative, so we add a base to keep fitness positive-ish
        fitness = sharpe - penalty
        
        diagnostics['raw_sharpe'] = sharpe
        diagnostics['final_fitness'] = fitness
        
        return fitness, diagnostics
    
    def _calculate_portfolio_returns(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame
    ) -> pd.Series:
        """Calculate weighted portfolio returns."""
        # Filter to tickers that exist in both weights and returns
        common_tickers = [t for t in weights.keys() if t in returns.columns]
        
        if not common_tickers:
            return pd.Series(dtype=float)
        
        # Normalize weights for available tickers
        total_weight = sum(weights[t] for t in common_tickers)
        if total_weight == 0:
            return pd.Series(dtype=float)
        
        normalized_weights = {t: weights[t] / total_weight for t in common_tickers}
        
        # Calculate weighted returns
        portfolio_returns = sum(
            returns[ticker] * weight
            for ticker, weight in normalized_weights.items()
        )
        
        return portfolio_returns.dropna()
    
    def calculate_heuristic_fitness(
        self,
        weights: dict[str, float],
        features: pd.DataFrame,
        heuristic_weights: dict[str, float]
    ) -> float:
        """
        Calculate a heuristic fitness based on features (no returns needed).
        
        Useful when historical returns are limited.
        
        Args:
            weights: Portfolio weights
            features: Feature DataFrame indexed by ticker
            heuristic_weights: Weights for each feature in the score
        
        Returns:
            Heuristic fitness score
        """
        if features.empty:
            return 0.0
        
        score = 0.0
        total_heuristic_weight = 0.0
        
        for feature, h_weight in heuristic_weights.items():
            # Try different feature name patterns
            for col in [feature, f'{feature}_norm', feature.replace('_', '')]:
                if col in features.columns:
                    # Weighted average of feature values by portfolio weights
                    feature_contribution = sum(
                        weights.get(ticker, 0) * features.loc[ticker, col]
                        for ticker in features.index
                        if ticker in weights
                    )
                    score += feature_contribution * h_weight
                    total_heuristic_weight += h_weight
                    break
        
        if total_heuristic_weight > 0:
            score /= total_heuristic_weight
        
        return score
