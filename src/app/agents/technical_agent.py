"""
Technical Indicator Agent - computes technical indicators from price data.
Purely computational - no network calls.
"""

import time

import numpy as np
import pandas as pd

from ..context import RunContext
from .base import AgentResult, BaseAgent


class TechnicalIndicatorAgent(BaseAgent):
    """
    Agent responsible for computing technical indicators.
    
    This agent is purely computational and does not make any network calls.
    It takes price history data and computes various technical indicators.
    """
    
    @property
    def name(self) -> str:
        return "technical_indicators"
    
    async def run(
        self,
        universe: list[str],
        ctx: 'RunContext',
        price_data: pd.DataFrame | None = None
    ) -> AgentResult:
        """
        Compute technical indicators for all tickers.
        
        Args:
            universe: List of tickers (for validation)
            ctx: Run context
            price_data: Multi-indexed DataFrame with price history
                       (passed from operator after price agent runs)
        
        Returns:
            AgentResult with DataFrame indexed by ticker containing
            indicator columns
        """
        start_time = time.time()
        errors = []
        
        if price_data is None or price_data.empty:
            errors.append("No price data provided")
            return self._create_result(pd.DataFrame(), len(universe), 0, errors)
        
        ctx.logger.info(f"Computing technical indicators for {len(universe)} tickers")
        
        # Get configured indicators
        agent_config = ctx.config.agents.get('technical')
        indicators_to_compute = (
            agent_config.indicators if agent_config and agent_config.indicators
            else ['returns_1m', 'returns_3m', 'volatility_21d', 'momentum_200d']
        )
        
        # Process each ticker
        results = []
        for ticker in universe:
            try:
                # Get ticker's price data
                if ticker not in price_data.index.get_level_values('ticker'):
                    errors.append(f"{ticker}: No price data available")
                    continue
                
                ticker_data = price_data.loc[ticker].copy()
                if ticker_data.empty or len(ticker_data) < 21:
                    errors.append(f"{ticker}: Insufficient price history")
                    continue
                
                # Ensure data is sorted by date
                ticker_data = ticker_data.sort_index()
                close = ticker_data['close']
                
                # Compute indicators
                row = {'ticker': ticker}
                
                for indicator in indicators_to_compute:
                    value = self._compute_indicator(indicator, close, ticker_data)
                    if value is not None:
                        row[indicator] = value
                
                results.append(row)
                
            except Exception as e:
                errors.append(f"{ticker}: {str(e)}")
        
        # Create result DataFrame
        if results:
            df = pd.DataFrame(results).set_index('ticker')
        else:
            df = pd.DataFrame()
        
        latency_ms = (time.time() - start_time) * 1000
        ctx.logger.info(
            f"Technical indicators: computed for {len(results)}/{len(universe)} tickers "
            f"in {latency_ms:.0f}ms"
        )
        
        return self._create_result(df, len(universe), latency_ms, errors)
    
    def _compute_indicator(
        self,
        indicator: str,
        close: pd.Series,
        full_data: pd.DataFrame
    ) -> float | None:
        """
        Compute a single indicator.
        
        Args:
            indicator: Indicator name
            close: Close price series
            full_data: Full OHLCV data
        
        Returns:
            Computed indicator value or None
        """
        try:
            if indicator == 'returns_1m':
                return self._compute_returns(close, 21)
            elif indicator == 'returns_3m':
                return self._compute_returns(close, 63)
            elif indicator == 'returns_6m':
                return self._compute_returns(close, 126)
            elif indicator == 'returns_12m':
                return self._compute_returns(close, 252)
            elif indicator == 'volatility_21d':
                return self._compute_volatility(close, 21)
            elif indicator == 'volatility_63d':
                return self._compute_volatility(close, 63)
            elif indicator == 'momentum_200d':
                return self._compute_momentum(close, 200)
            elif indicator == 'rsi_14':
                return self._compute_rsi(close, 14)
            else:
                return None
        except Exception:
            return None
    
    def _compute_returns(self, close: pd.Series, days: int) -> float | None:
        """Compute period returns."""
        if len(close) < days + 1:
            return None
        current = close.iloc[-1]
        past = close.iloc[-days - 1]
        if past == 0:
            return None
        return (current - past) / past
    
    def _compute_volatility(self, close: pd.Series, days: int) -> float | None:
        """Compute annualized volatility from daily returns."""
        if len(close) < days + 1:
            return None
        returns = close.pct_change().dropna().tail(days)
        if len(returns) < days // 2:
            return None
        return returns.std() * np.sqrt(252)  # Annualized
    
    def _compute_momentum(self, close: pd.Series, days: int) -> float | None:
        """Compute momentum as price vs moving average."""
        if len(close) < days:
            return None
        ma = close.tail(days).mean()
        if ma == 0:
            return None
        return (close.iloc[-1] - ma) / ma
    
    def _compute_rsi(self, close: pd.Series, period: int = 14) -> float | None:
        """Compute Relative Strength Index."""
        if len(close) < period + 1:
            return None
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).tail(period).mean()
        loss = (-delta.where(delta < 0, 0)).tail(period).mean()
        
        if loss == 0:
            return 100.0
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
