"""
Portfolio constraints for ACO optimization.
"""

from dataclasses import dataclass

from ..config import ConstraintsConfig


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""
    name: str
    penalty: float
    message: str


class ConstraintChecker:
    """
    Checks portfolio constraints and computes penalties.
    
    Constraints:
    - Sum of weights = 1
    - Max weight per ticker
    - Min/max number of holdings
    - Optional sector caps
    """
    
    def __init__(self, config: ConstraintsConfig):
        self.max_weight = config.max_weight_per_ticker
        self.min_weight = config.min_weight_threshold
        self.min_holdings = config.min_holdings
        self.max_holdings = config.max_holdings
        self.sum_tolerance = config.sum_weights_tolerance
        self.sector_cap = config.sector_cap
    
    def check_all(
        self,
        weights: dict[str, float],
        sectors: dict[str, str] | None = None
    ) -> tuple[float, list[ConstraintViolation]]:
        """
        Check all constraints and return total penalty.
        
        Args:
            weights: Dict of ticker -> weight
            sectors: Optional dict of ticker -> sector
        
        Returns:
            Tuple of (total_penalty, list of violations)
        """
        violations = []
        
        # Check sum of weights
        penalty, violation = self.check_sum_weights(weights)
        if violation:
            violations.append(violation)
        
        # Check max weight per ticker
        p, v = self.check_max_weights(weights)
        penalty += p
        violations.extend(v)
        
        # Check holdings count
        p, v = self.check_holdings_count(weights)
        penalty += p
        if v:
            violations.append(v)
        
        # Check sector caps if applicable
        if sectors and self.sector_cap:
            p, v = self.check_sector_caps(weights, sectors)
            penalty += p
            violations.extend(v)
        
        return penalty, violations
    
    def check_sum_weights(
        self,
        weights: dict[str, float]
    ) -> tuple[float, ConstraintViolation | None]:
        """Check that weights sum to 1."""
        total = sum(weights.values())
        deviation = abs(total - 1.0)
        
        if deviation <= self.sum_tolerance:
            return 0.0, None
        
        # Penalty proportional to deviation
        penalty = deviation * 10.0  # Heavy penalty for sum violation
        return penalty, ConstraintViolation(
            name='sum_weights',
            penalty=penalty,
            message=f"Weights sum to {total:.4f}, expected 1.0"
        )
    
    def check_max_weights(
        self,
        weights: dict[str, float]
    ) -> tuple[float, list[ConstraintViolation]]:
        """Check that no weight exceeds maximum."""
        total_penalty = 0.0
        violations = []
        
        for ticker, weight in weights.items():
            if weight > self.max_weight:
                excess = weight - self.max_weight
                penalty = excess * 5.0  # Penalty per excess percentage
                total_penalty += penalty
                violations.append(ConstraintViolation(
                    name='max_weight',
                    penalty=penalty,
                    message=f"{ticker} weight {weight:.2%} exceeds max {self.max_weight:.2%}"
                ))
        
        return total_penalty, violations
    
    def check_holdings_count(
        self,
        weights: dict[str, float]
    ) -> tuple[float, ConstraintViolation | None]:
        """Check that holdings count is within bounds."""
        # Count only meaningful weights
        count = sum(1 for w in weights.values() if w >= self.min_weight)
        
        if count < self.min_holdings:
            penalty = (self.min_holdings - count) * 1.0
            return penalty, ConstraintViolation(
                name='min_holdings',
                penalty=penalty,
                message=f"Only {count} holdings, minimum is {self.min_holdings}"
            )
        
        if count > self.max_holdings:
            penalty = (count - self.max_holdings) * 0.5
            return penalty, ConstraintViolation(
                name='max_holdings',
                penalty=penalty,
                message=f"{count} holdings exceeds maximum of {self.max_holdings}"
            )
        
        return 0.0, None
    
    def check_sector_caps(
        self,
        weights: dict[str, float],
        sectors: dict[str, str]
    ) -> tuple[float, list[ConstraintViolation]]:
        """Check that no sector exceeds the cap."""
        if not self.sector_cap:
            return 0.0, []
        
        # Aggregate weights by sector
        sector_weights: dict[str, float] = {}
        for ticker, weight in weights.items():
            sector = sectors.get(ticker, 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        total_penalty = 0.0
        violations = []
        
        for sector, weight in sector_weights.items():
            if weight > self.sector_cap:
                excess = weight - self.sector_cap
                penalty = excess * 3.0
                total_penalty += penalty
                violations.append(ConstraintViolation(
                    name='sector_cap',
                    penalty=penalty,
                    message=f"Sector '{sector}' weight {weight:.2%} exceeds cap {self.sector_cap:.2%}"
                ))
        
        return total_penalty, violations
    
    def is_feasible(
        self,
        weights: dict[str, float],
        sectors: dict[str, str] | None = None
    ) -> bool:
        """Check if a portfolio is feasible (no hard violations)."""
        penalty, _ = self.check_all(weights, sectors)
        return penalty < 0.1  # Small tolerance for numerical issues
