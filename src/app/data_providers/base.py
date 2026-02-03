"""
Base data provider interface and factory.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..context import RunContext


@dataclass
class ProviderResponse:
    """Standard response from a provider call."""
    success: bool
    data: pd.DataFrame | dict | None
    error: str | None = None
    cached: bool = False


class DataProvider(ABC):
    """
    Abstract base class for data providers.
    
    Providers are responsible for fetching data from external sources
    (APIs, files, etc.) with proper rate limiting and error handling.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass
    
    @abstractmethod
    async def get_price_history(
        self,
        ticker: str,
        start: date,
        end: date,
        ctx: 'RunContext'
    ) -> ProviderResponse:
        """
        Fetch historical OHLCV data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start: Start date
            end: End date
            ctx: Run context
        
        Returns:
            ProviderResponse with DataFrame containing columns:
            date (index), open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    async def get_fundamentals(
        self,
        ticker: str,
        ctx: 'RunContext'
    ) -> ProviderResponse:
        """
        Fetch fundamental data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            ctx: Run context
        
        Returns:
            ProviderResponse with dict containing:
            market_cap, pe_ratio, eps, sector, etc.
        """
        pass
    
    async def get_universe(
        self,
        ctx: 'RunContext'
    ) -> ProviderResponse:
        """
        Fetch the S&P 500 ticker list.
        
        Default implementation returns empty (not supported).
        Override in providers that support this.
        
        Returns:
            ProviderResponse with list of tickers
        """
        return ProviderResponse(
            success=False,
            data=None,
            error="Universe fetch not supported by this provider"
        )


def get_provider(name: str, ctx: 'RunContext') -> DataProvider:
    """
    Factory function to get a data provider by name.
    
    Args:
        name: Provider name (alphavantage, yfinance, etc.)
        ctx: Run context
    
    Returns:
        DataProvider instance
    
    Raises:
        ValueError: If provider name is unknown
    """
    from .alphavantage import AlphaVantageProvider
    from .yfinance_fallback import YFinanceProvider
    
    providers = {
        'alphavantage': AlphaVantageProvider,
        'yfinance': YFinanceProvider,
    }
    
    if name not in providers:
        raise ValueError(f"Unknown provider: {name}. Available: {list(providers.keys())}")
    
    return providers[name](ctx)
