"""
Portfolio representation for optimization.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Portfolio:
    """
    Represents a portfolio allocation.
    
    Attributes:
        weights: Dict mapping ticker to weight (0-1)
        fitness: Fitness score from optimization
        diagnostics: Additional information about the portfolio
    """
    weights: dict[str, float]
    fitness: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure weights are properly initialized."""
        # Only filter out truly zero weights (not small positive ones)
        # Real cleanup is done explicitly via apply_min_threshold()
        self.weights = {
            k: v for k, v in self.weights.items()
            if v > 0
        }
    
    @property
    def tickers(self) -> list[str]:
        """Get list of tickers in portfolio."""
        return list(self.weights.keys())
    
    @property
    def num_holdings(self) -> int:
        """Get number of holdings."""
        return len(self.weights)
    
    @property
    def total_weight(self) -> float:
        """Get sum of all weights."""
        return sum(self.weights.values())
    
    @property
    def max_weight(self) -> float:
        """Get maximum single weight."""
        return max(self.weights.values()) if self.weights else 0.0
    
    @property
    def top_holdings(self) -> list[tuple[str, float]]:
        """Get top 10 holdings by weight."""
        sorted_weights = sorted(
            self.weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_weights[:10]
    
    def normalize(self) -> 'Portfolio':
        """Return a new portfolio with weights normalized to sum to 1."""
        total = self.total_weight
        if total == 0:
            return Portfolio({}, self.fitness, self.diagnostics.copy())
        
        normalized_weights = {
            k: v / total for k, v in self.weights.items()
        }
        return Portfolio(normalized_weights, self.fitness, self.diagnostics.copy())
    
    def apply_min_threshold(self, threshold: float) -> 'Portfolio':
        """Return a new portfolio with weights below threshold removed."""
        filtered_weights = {
            k: v for k, v in self.weights.items()
            if v >= threshold
        }
        return Portfolio(filtered_weights, self.fitness, self.diagnostics.copy())
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            'weights': self.weights,
            'fitness': self.fitness,
            'num_holdings': self.num_holdings,
            'total_weight': self.total_weight,
            'max_weight': self.max_weight,
            'top_holdings': self.top_holdings,
            'diagnostics': self.diagnostics
        }
    
    def __repr__(self) -> str:
        return (
            f"Portfolio(holdings={self.num_holdings}, "
            f"fitness={self.fitness:.4f}, "
            f"max_weight={self.max_weight:.2%})"
        )
