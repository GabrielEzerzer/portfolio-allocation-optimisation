"""
Price History Agent - fetches historical OHLCV data for tickers.
"""

import asyncio
import time
from datetime import date, timedelta

import pandas as pd

from ..context import RunContext
from ..data_providers import get_provider
from .base import AgentResult, BaseAgent


class PriceHistoryAgent(BaseAgent):
    """
    Agent responsible for fetching historical price data.
    
    Fetches OHLCV data for each ticker in the universe using the
    configured data provider with fallback support.
    """
    
    @property
    def name(self) -> str:
        return "price_history"
    
    async def run(
        self,
        universe: list[str],
        ctx: 'RunContext'
    ) -> AgentResult:
        """
        Fetch historical price data for all tickers.
        
        Args:
            universe: List of tickers to fetch
            ctx: Run context
        
        Returns:
            AgentResult with multi-indexed DataFrame (ticker, date)
            containing columns: open, high, low, close, volume
        """
        start_time = time.time()
        errors = []
        
        # Determine date range
        end_date = ctx.end_date or date.today()
        agent_config = ctx.config.agents.get('price')
        history_days = agent_config.history_days if agent_config else 365
        start_date = ctx.start_date or (end_date - timedelta(days=history_days))
        
        ctx.logger.info(
            f"Fetching price history for {len(universe)} tickers "
            f"from {start_date} to {end_date}"
        )
        
        # Get provider
        try:
            provider = get_provider(ctx.config.primary_provider, ctx)
        except ValueError as e:
            errors.append(str(e))
            return self._create_result(pd.DataFrame(), len(universe), 0, errors)
        
        # Fetch data for all tickers concurrently (with semaphore to limit concurrency)
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        async def fetch_ticker(ticker: str) -> tuple[str, pd.DataFrame | None, str | None]:
            async with semaphore:
                response = await provider.get_price_history(
                    ticker, start_date, end_date, ctx
                )
                if response.success and response.data is not None:
                    return (ticker, response.data, None)
                else:
                    return (ticker, None, response.error)
        
        # Run all fetches concurrently
        tasks = [fetch_ticker(ticker) for ticker in universe]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        all_data = []
        for result in results:
            if isinstance(result, Exception):
                errors.append(f"Exception: {result}")
                continue
            
            ticker, df, error = result
            if error:
                errors.append(f"{ticker}: {error}")
            elif df is not None and not df.empty:
                # Add ticker column for multi-index
                df = df.copy()
                df['ticker'] = ticker
                all_data.append(df)
        
        # Combine all data
        if all_data:
            combined = pd.concat(all_data, ignore_index=False)
            combined = combined.reset_index()
            combined = combined.set_index(['ticker', 'date']).sort_index()
        else:
            combined = pd.DataFrame()
        
        latency_ms = (time.time() - start_time) * 1000
        ctx.logger.info(
            f"Price history: fetched {len(all_data)}/{len(universe)} tickers "
            f"in {latency_ms:.0f}ms"
        )
        
        return self._create_result(combined, len(universe), latency_ms, errors)
    
    def get_prices_for_ticker(
        self,
        data: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """Extract price data for a single ticker from combined result."""
        if data.empty:
            return pd.DataFrame()
        try:
            return data.loc[ticker]
        except KeyError:
            return pd.DataFrame()
    
    def get_close_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract close prices as a pivot table (date x ticker).
        
        Useful for computing returns and portfolio values.
        """
        if data.empty:
            return pd.DataFrame()
        
        # Reset index to get ticker and date as columns
        df = data.reset_index()
        
        # Pivot to get date as index, tickers as columns
        pivot = df.pivot(index='date', columns='ticker', values='close')
        return pivot.sort_index()
