"""
yfinance fallback data provider.
Uses the yfinance library for price data when primary provider is unavailable.
"""

from datetime import date

import pandas as pd

from ..context import RunContext
from .base import DataProvider, ProviderResponse


class YFinanceProvider(DataProvider):
    """
    yfinance fallback data provider.
    
    Note: yfinance uses web scraping and may have reliability issues.
    Use as fallback when primary API provider is unavailable.
    """
    
    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self._yf = None
    
    @property
    def name(self) -> str:
        return "yfinance"
    
    def _get_yf(self):
        """Lazy load yfinance to avoid import errors if not installed."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                raise ImportError(
                    "yfinance is not installed. Install with: pip install yfinance"
                )
        return self._yf
    
    async def get_price_history(
        self,
        ticker: str,
        start: date,
        end: date,
        ctx: RunContext
    ) -> ProviderResponse:
        """
        Fetch historical price data using yfinance.
        
        Note: yfinance calls are synchronous, wrapped for async interface.
        """
        # Check cache first
        cache_key = f"{ticker}_{start.isoformat()}_{end.isoformat()}"
        if ctx.cache_manager:
            cached = ctx.cache_manager.get_dataframe('prices', cache_key)
            if cached is not None:
                ctx.logger.debug(f"Cache hit for {ticker} prices")
                return ProviderResponse(success=True, data=cached, cached=True)
        
        try:
            yf = self._get_yf()
            
            # yfinance is synchronous
            stock = yf.Ticker(ticker)
            df = stock.history(start=start.isoformat(), end=end.isoformat())
            
            if df.empty:
                return ProviderResponse(
                    success=False,
                    data=None,
                    error=f"No data returned for {ticker}"
                )
            
            # Normalize column names
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Keep only needed columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Convert index to date (remove timezone)
            df.index = pd.to_datetime(df.index).date
            df.index.name = 'date'
            
            # Cache result
            if ctx.cache_manager:
                ctx.cache_manager.set_dataframe('prices', cache_key, df)
            
            return ProviderResponse(success=True, data=df)
            
        except Exception as e:
            ctx.logger.error(f"yfinance error for {ticker}: {e}")
            return ProviderResponse(
                success=False,
                data=None,
                error=str(e)
            )
    
    async def get_fundamentals(
        self,
        ticker: str,
        ctx: RunContext
    ) -> ProviderResponse:
        """
        Fetch fundamental data using yfinance.
        """
        # Check cache first
        if ctx.cache_manager:
            cached = ctx.cache_manager.get_json('fundamentals', ticker)
            if cached is not None:
                ctx.logger.debug(f"Cache hit for {ticker} fundamentals")
                return ProviderResponse(success=True, data=cached, cached=True)
        
        try:
            yf = self._get_yf()
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                return ProviderResponse(
                    success=False,
                    data=None,
                    error=f"No info returned for {ticker}"
                )
            
            fundamentals = {
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('forwardPE') or info.get('trailingPE'),
                'eps': info.get('trailingEps'),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'dividend_yield': info.get('dividendYield'),
                'book_value': info.get('bookValue'),
                'profit_margin': info.get('profitMargins'),
            }
            
            # Cache result
            if ctx.cache_manager:
                ctx.cache_manager.set_json('fundamentals', ticker, fundamentals)
            
            return ProviderResponse(success=True, data=fundamentals)
            
        except Exception as e:
            ctx.logger.error(f"yfinance fundamentals error for {ticker}: {e}")
            return ProviderResponse(
                success=False,
                data=None,
                error=str(e)
            )
