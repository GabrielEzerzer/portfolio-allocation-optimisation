"""
RunContext - shared execution context for all agents and operators.
"""

from dataclasses import dataclass, field
from datetime import date
from logging import Logger
from typing import TYPE_CHECKING

import aiohttp
from aiolimiter import AsyncLimiter

if TYPE_CHECKING:
    from .config import Config
    from .utils.cache import CacheManager


@dataclass
class RunContext:
    """
    Shared context passed to all agents during a run cycle.
    
    Attributes:
        config: Application configuration
        logger: Logger instance
        mode: Current run mode (live, cached, test)
        session: aiohttp session for HTTP requests (may be None in test mode)
        rate_limiter: Rate limiter for API calls
        cache_manager: Cache manager for storing/retrieving data
        start_date: Start of date range for data
        end_date: End of date range for data
        random_seed: Random seed for deterministic behavior
        run_id: Unique identifier for this run
    """
    config: 'Config'
    logger: Logger
    mode: str
    session: aiohttp.ClientSession | None = None
    rate_limiter: AsyncLimiter | None = None
    cache_manager: 'CacheManager | None' = None
    start_date: date | None = None
    end_date: date | None = None
    random_seed: int | None = None
    run_id: str = ""
    
    # Runtime state
    _errors: list[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """Record an error encountered during the run."""
        self._errors.append(error)
        self.logger.error(error)
    
    @property
    def errors(self) -> list[str]:
        """Get list of errors from this run."""
        return self._errors.copy()
    
    def is_live_mode(self) -> bool:
        """Check if running in live mode (API calls allowed)."""
        return self.mode == 'live'
    
    def is_cached_mode(self) -> bool:
        """Check if running in cached mode."""
        return self.mode == 'cached'
    
    def is_test_mode(self) -> bool:
        """Check if running in test mode (no network)."""
        return self.mode == 'test'
    
    def is_backtest_mode(self) -> bool:
        """Check if running a backtest."""
        return self.mode == 'backtest'
