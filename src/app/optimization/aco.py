"""
Ant Colony Optimization for portfolio construction.
"""

import numpy as np
import pandas as pd

from ..config import ACOConfig, ConstraintsConfig
from .constraints import ConstraintChecker
from .fitness import FitnessCalculator
from .portfolio import Portfolio


class ACOOptimizer:
    """
    Ant Colony Optimization for portfolio weight allocation.
    
    The ACO algorithm:
    1. Discretizes weights into chunks (e.g., 5% units)
    2. Each ant constructs a portfolio by allocating weight units
    3. Allocation probability is based on pheromone and heuristic desirability
    4. Best portfolios deposit more pheromone
    5. Pheromone evaporates over time
    """
    
    def __init__(
        self,
        aco_config: ACOConfig,
        constraints_config: ConstraintsConfig,
        rng: np.random.Generator | None = None
    ):
        self.config = aco_config
        self.constraints = constraints_config
        self.rng = rng or np.random.default_rng(aco_config.random_seed)
        
        # ACO parameters
        self.num_ants = aco_config.num_ants
        self.num_iterations = aco_config.num_iterations
        self.evaporation_rate = aco_config.evaporation_rate
        self.alpha = aco_config.alpha  # Pheromone importance
        self.beta = aco_config.beta    # Heuristic importance
        self.weight_granularity = aco_config.weight_granularity
        
        # Derived
        self.max_units = int(self.constraints.max_weight_per_ticker / self.weight_granularity)
        self.total_units = int(1.0 / self.weight_granularity)
        
        # State (initialized during optimize)
        self.pheromone: np.ndarray | None = None
        self.heuristic: np.ndarray | None = None
        self.tickers: list[str] = []
        
        # Constraint checking
        self.constraint_checker = ConstraintChecker(constraints_config)
        self.fitness_calculator = FitnessCalculator(self.constraint_checker)
    
    def optimize(
        self,
        features: pd.DataFrame,
        returns: pd.DataFrame,
        sectors: dict[str, str] | None = None
    ) -> Portfolio:
        """
        Run ACO optimization to find optimal portfolio weights.
        
        Args:
            features: Feature DataFrame indexed by ticker
            returns: Historical returns DataFrame (columns = tickers)
            sectors: Optional dict of ticker -> sector
        
        Returns:
            Best Portfolio found
        """
        if features.empty:
            return Portfolio({}, 0.0, {'error': 'No features provided'})
        
        self.tickers = list(features.index)
        n_tickers = len(self.tickers)
        
        # Initialize pheromone uniformly
        self.pheromone = np.ones((n_tickers, self.max_units + 1))
        
        # Compute heuristic desirability from features
        self.heuristic = self._compute_heuristic(features)
        
        # Track best solution
        best_portfolio: Portfolio | None = None
        best_fitness = float('-inf')
        
        # Iteration history for diagnostics
        fitness_history = []
        
        for iteration in range(self.num_iterations):
            # Each ant constructs a portfolio
            ant_portfolios = []
            
            for ant in range(self.num_ants):
                weights = self._construct_portfolio()
                
                # Calculate fitness
                fitness, diagnostics = self.fitness_calculator.calculate(
                    weights, returns, sectors
                )
                
                portfolio = Portfolio(weights, fitness, diagnostics)
                ant_portfolios.append(portfolio)
                
                # Update best
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_portfolio = portfolio
            
            # Evaporate pheromone
            self.pheromone *= (1 - self.evaporation_rate)
            
            # Deposit pheromone from best ants
            self._update_pheromone(ant_portfolios)
            
            # Track progress
            avg_fitness = np.mean([p.fitness for p in ant_portfolios])
            fitness_history.append({
                'iteration': iteration,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness
            })
            
            # Early stopping if converged
            if iteration > 10:
                recent_best = [h['best_fitness'] for h in fitness_history[-10:]]
                if max(recent_best) - min(recent_best) < 0.001:
                    break
        
        # Finalize best portfolio
        if best_portfolio is None:
            best_portfolio = Portfolio({}, 0.0, {'error': 'No valid portfolio found'})
        
        # Normalize and clean up
        best_portfolio = best_portfolio.normalize()
        best_portfolio = best_portfolio.apply_min_threshold(
            self.constraints.min_weight_threshold
        )
        best_portfolio = best_portfolio.normalize()  # Re-normalize after threshold
        
        # Update diagnostics
        best_portfolio.diagnostics['iterations'] = len(fitness_history)
        best_portfolio.diagnostics['fitness_history'] = fitness_history
        best_portfolio.diagnostics['top_pheromone'] = self._get_top_pheromone(5)
        
        return best_portfolio
    
    def _compute_heuristic(self, features: pd.DataFrame) -> np.ndarray:
        """
        Compute heuristic desirability for each ticker.
        
        Higher values = more desirable.
        """
        n_tickers = len(features)
        heuristic = np.ones(n_tickers)
        
        weights = self.config.heuristic_weights
        
        # Momentum contribution
        if 'momentum' in weights:
            for col in ['momentum_200d', 'momentum_200d_norm']:
                if col in features.columns:
                    # Normalize to 0-1 range
                    vals = features[col].values
                    normalized = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)
                    heuristic += weights['momentum'] * normalized
                    break
        
        # Returns contribution
        if 'returns' in weights:
            for col in ['returns_1m', 'returns_1m_norm', 'returns_3m', 'returns_3m_norm']:
                if col in features.columns:
                    vals = features[col].values
                    normalized = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)
                    heuristic += weights['returns'] * normalized
                    break
        
        # Inverse volatility (lower vol = higher score)
        if 'inverse_volatility' in weights:
            for col in ['volatility_21d', 'volatility_21d_norm']:
                if col in features.columns:
                    vals = features[col].values
                    # Inverse and normalize
                    inv_vals = 1.0 / (vals + 0.01)
                    normalized = (inv_vals - inv_vals.min()) / (inv_vals.max() - inv_vals.min() + 1e-10)
                    heuristic += weights['inverse_volatility'] * normalized
                    break
        
        # Ensure positive values
        heuristic = np.maximum(heuristic, 0.1)
        
        return heuristic
    
    def _construct_portfolio(self) -> dict[str, float]:
        """
        Construct a portfolio as an ant.
        
        Allocates weight units probabilistically based on
        pheromone and heuristic values.
        """
        n_tickers = len(self.tickers)
        allocation = np.zeros(n_tickers, dtype=int)  # Units per ticker
        remaining_units = self.total_units
        
        while remaining_units > 0:
            # Calculate probabilities for each ticker
            probabilities = np.zeros(n_tickers)
            
            for i in range(n_tickers):
                if allocation[i] < self.max_units:
                    # Consider allocating one more unit to ticker i
                    next_level = allocation[i] + 1
                    pheromone = self.pheromone[i, next_level]
                    heuristic = self.heuristic[i]
                    
                    probabilities[i] = (pheromone ** self.alpha) * (heuristic ** self.beta)
            
            # If no valid moves, break
            if probabilities.sum() == 0:
                break
            
            # Normalize probabilities
            probabilities /= probabilities.sum()
            
            # Select ticker
            selected = self.rng.choice(n_tickers, p=probabilities)
            allocation[selected] += 1
            remaining_units -= 1
        
        # Convert to weights
        weights = {
            self.tickers[i]: allocation[i] * self.weight_granularity
            for i in range(n_tickers)
            if allocation[i] > 0
        }
        
        return weights
    
    def _update_pheromone(self, portfolios: list[Portfolio]) -> None:
        """Update pheromone based on portfolio fitness."""
        # Sort by fitness
        sorted_portfolios = sorted(portfolios, key=lambda p: p.fitness, reverse=True)
        
        # Top performers deposit pheromone
        n_elite = max(1, len(portfolios) // 5)  # Top 20%
        
        for rank, portfolio in enumerate(sorted_portfolios[:n_elite]):
            if portfolio.fitness <= 0:
                continue
            
            # Deposit amount decreases with rank
            deposit = portfolio.fitness / (rank + 1)
            
            for ticker, weight in portfolio.weights.items():
                if ticker not in self.tickers:
                    continue
                
                ticker_idx = self.tickers.index(ticker)
                units = int(weight / self.weight_granularity)
                
                if 0 <= units <= self.max_units:
                    self.pheromone[ticker_idx, units] += deposit
    
    def _get_top_pheromone(self, n: int) -> list[tuple[str, float]]:
        """Get tickers with highest average pheromone."""
        avg_pheromone = self.pheromone.mean(axis=1)
        top_indices = np.argsort(avg_pheromone)[-n:][::-1]
        
        return [
            (self.tickers[i], float(avg_pheromone[i]))
            for i in top_indices
        ]
