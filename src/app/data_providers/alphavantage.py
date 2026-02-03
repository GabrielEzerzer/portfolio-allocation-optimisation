"""
Alpha Vantage data provider implementation.
"""

import asyncio
from datetime import date
from typing import Any

import pandas as pd

from ..context import RunContext
from .base import DataProvider, ProviderResponse


class AlphaVantageProvider(DataProvider):
    """
    Alpha Vantage API data provider.
    
    Supports:
    - TIME_SERIES_DAILY for OHLCV data
    - OVERVIEW for fundamentals
    
    Rate limiting and caching are handled through the RunContext.
    """
    
    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.config = ctx.config.providers.get('alphavantage')
        if not self.config:
            raise ValueError("Alpha Vantage provider not configured")
        
        self.base_url = self.config.base_url
        self.api_key = self.config.api_key
        self.timeout = self.config.timeout_seconds
        self.retries = self.config.retries
        self.backoff_base = self.config.backoff_base
    
    @property
    def name(self) -> str:
        return "alphavantage"
    
    def _is_valid_api_key(self) -> bool:
        """Check if API key is configured (not a placeholder)."""
        invalid_keys = ['YOUR_API_KEY_HERE', '', None]
        return self.api_key not in invalid_keys
    
    async def _make_request(
        self,
        params: dict[str, str]
    ) -> dict[str, Any] | None:
        """
        Make a rate-limited API request with retries.
        
        Args:
            params: Query parameters
        
        Returns:
            JSON response or None on failure
        """
        if not self._is_valid_api_key():
            self.ctx.logger.warning("Alpha Vantage API key not configured")
            return None
        
        if not self.ctx.session:
            self.ctx.logger.error("No HTTP session available")
            return None
        
        params['apikey'] = self.api_key
        
        for attempt in range(self.retries):
            try:
                # Wait for rate limit
                if self.ctx.rate_limiter:
                    await self.ctx.rate_limiter.acquire()
                
                async with self.ctx.session.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for API error messages
                        if 'Error Message' in data:
                            self.ctx.logger.error(f"API error: {data['Error Message']}")
                            return None
                        if 'Note' in data:
                            self.ctx.logger.warning(f"API note (rate limit?): {data['Note']}")
                            # Wait and retry on rate limit
                            await asyncio.sleep(60)
                            continue
                        
                        return data
                    
                    elif response.status == 429:
                        # Rate limited
                        wait_time = self.backoff_base ** attempt * 60
                        self.ctx.logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    
                    else:
                        self.ctx.logger.error(f"API returned status {response.status}")
                        
            except asyncio.TimeoutError:
                self.ctx.logger.warning(f"Request timeout (attempt {attempt + 1}/{self.retries})")
            except Exception as e:
                self.ctx.logger.error(f"Request error: {e}")
            
            # Backoff before retry
            if attempt < self.retries - 1:
                await asyncio.sleep(self.backoff_base ** attempt)
        
        return None
    
    async def get_price_history(
        self,
        ticker: str,
        start: date,
        end: date,
        ctx: RunContext
    ) -> ProviderResponse:
        """
        Fetch historical price data from Alpha Vantage.
        
        Uses TIME_SERIES_DAILY_ADJUSTED for full history.
        """
        # Check cache first
        cache_key = f"{ticker}_{start.isoformat()}_{end.isoformat()}"
        if ctx.cache_manager:
            cached = ctx.cache_manager.get_dataframe('prices', cache_key)
            if cached is not None:
                ctx.logger.debug(f"Cache hit for {ticker} prices")
                return ProviderResponse(success=True, data=cached, cached=True)
        
        # Make API request
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': ticker,
            'outputsize': 'full'
        }
        
        data = await self._make_request(params)
        if not data:
            return ProviderResponse(
                success=False,
                data=None,
                error=f"Failed to fetch price data for {ticker}"
            )
        
        # Parse response
        time_series = data.get('Time Series (Daily)', {})
        if not time_series:
            return ProviderResponse(
                success=False,
                data=None,
                error=f"No time series data for {ticker}"
            )
        
        # Convert to DataFrame
        records = []
        for date_str, values in time_series.items():
            try:
                record_date = date.fromisoformat(date_str)
                if start <= record_date <= end:
                    records.append({
                        'date': record_date,
                        'open': float(values.get('1. open', 0)),
                        'high': float(values.get('2. high', 0)),
                        'low': float(values.get('3. low', 0)),
                        'close': float(values.get('4. close', 0)),
                        'volume': int(values.get('5. volume', 0))
                    })
            except (ValueError, KeyError) as e:
                ctx.logger.warning(f"Error parsing date {date_str}: {e}")
        
        if not records:
            return ProviderResponse(
                success=False,
                data=None,
                error=f"No data in date range for {ticker}"
            )
        
        df = pd.DataFrame(records)
        df = df.set_index('date').sort_index()
        
        # Cache result
        if ctx.cache_manager:
            ctx.cache_manager.set_dataframe('prices', cache_key, df)
        
        return ProviderResponse(success=True, data=df)
    
    async def get_fundamentals(
        self,
        ticker: str,
        ctx: RunContext
    ) -> ProviderResponse:
        """
        Fetch fundamental data from Alpha Vantage OVERVIEW endpoint.
        """
        # Check cache first
        if ctx.cache_manager:
            cached = ctx.cache_manager.get_json('fundamentals', ticker)
            if cached is not None:
                ctx.logger.debug(f"Cache hit for {ticker} fundamentals")
                return ProviderResponse(success=True, data=cached, cached=True)
        
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker
        }
        
        data = await self._make_request(params)
        if not data or 'Symbol' not in data:
            return ProviderResponse(
                success=False,
                data=None,
                error=f"Failed to fetch fundamentals for {ticker}"
            )
        
        # Extract relevant fields
        fundamentals = {
            'market_cap': self._parse_float(data.get('MarketCapitalization')),
            'pe_ratio': self._parse_float(data.get('PERatio')),
            'eps': self._parse_float(data.get('EPS')),
            'sector': data.get('Sector', ''),
            'industry': data.get('Industry', ''),
            'dividend_yield': self._parse_float(data.get('DividendYield')),
            'book_value': self._parse_float(data.get('BookValue')),
            'profit_margin': self._parse_float(data.get('ProfitMargin')),
        }
        
        # Cache result
        if ctx.cache_manager:
            ctx.cache_manager.set_json('fundamentals', ticker, fundamentals)
        
        return ProviderResponse(success=True, data=fundamentals)
    
    def _parse_float(self, value: Any) -> float | None:
        """Safely parse a float value."""
        if value is None or value == 'None' or value == '-':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
