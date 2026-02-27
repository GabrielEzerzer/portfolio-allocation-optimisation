"""
Base agent interface and result types.
All runtime agents must conform to this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..context import RunContext


@dataclass
class AgentResult:
    """
    Standard result container returned by all agents.
    
    Attributes:
        name: Agent name identifier
        data: DataFrame indexed by ticker with named columns
        metadata: Additional run information
    """
    name: str
    data: pd.DataFrame
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure required metadata fields exist."""
        if 'timestamp' not in self.metadata:
            self.metadata['timestamp'] = datetime.now().isoformat()
        if 'coverage' not in self.metadata:
            self.metadata['coverage'] = len(self.data) if not self.data.empty else 0
        if 'latency_ms' not in self.metadata:
            self.metadata['latency_ms'] = 0
        if 'errors' not in self.metadata:
            self.metadata['errors'] = []
    
    @property
    def coverage(self) -> float:
        """Get coverage as a ratio (0-1)."""
        return self.metadata.get('coverage_ratio', 0.0)
    
    @property
    def has_errors(self) -> bool:
        """Check if the agent encountered errors."""
        return len(self.metadata.get('errors', [])) > 0
    
    @property
    def errors(self) -> list[str]:
        """Get list of errors."""
        return self.metadata.get('errors', [])
    
    def add_error(self, error: str) -> None:
        """Add an error to the result."""
        if 'errors' not in self.metadata:
            self.metadata['errors'] = []
        self.metadata['errors'].append(error)


class BaseAgent(ABC):
    """
    Abstract base class for all runtime agents.
    
    Agents are responsible for fetching or computing specific data domains.
    Each agent must implement the async run() method.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifier for this agent."""
        pass
    
    @abstractmethod
    async def run(
        self,
        universe: list[str],
        ctx: 'RunContext'
    ) -> AgentResult:
        """
        Execute the agent's data gathering/computation.
        
        Args:
            universe: List of tickers to process
            ctx: Shared run context with configuration and resources
        
        Returns:
            AgentResult with data indexed by ticker
        """
        pass
    
    def _create_result(
        self,
        data: pd.DataFrame,
        requested_count: int,
        latency_ms: float,
        errors: list[str] | None = None,
        override_coverage: int | None = None
    ) -> AgentResult:
        """
        Helper to create a properly formatted AgentResult.
        
        Args:
            data: DataFrame indexed by ticker
            requested_count: Number of tickers requested
            latency_ms: Time taken in milliseconds
            errors: List of error messages
            override_coverage: If set, use this as the returned count
                              (useful for multi-indexed data like prices)
        
        Returns:
            Formatted AgentResult
        """
        if override_coverage is not None:
            returned_count = override_coverage
        else:
            returned_count = len(data) if not data.empty else 0
        coverage_ratio = returned_count / requested_count if requested_count > 0 else 0.0
        
        return AgentResult(
            name=self.name,
            data=data,
            metadata={
                'timestamp': datetime.now().isoformat(),
                'coverage': returned_count,
                'coverage_ratio': coverage_ratio,
                'requested': requested_count,
                'latency_ms': latency_ms,
                'errors': errors or []
            }
        )
