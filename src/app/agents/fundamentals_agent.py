"""
Fundamentals Agent - fetches fundamental data for tickers.
"""

import asyncio
import time

import numpy as np
import pandas as pd

from ..context import RunContext
from ..data_providers import get_provider
from .base import AgentResult, BaseAgent


class FundamentalsAgent(BaseAgent):
    """
    Agent responsible for fetching fundamental data.
    
    Fetches market cap, PE ratio, EPS, sector classification, etc.
    Gracefully degrades if data is unavailable.
    """
    
    @property
    def name(self) -> str:
        return "fundamentals"
    
    async def run(
        self,
        universe: list[str],
        ctx: 'RunContext'
    ) -> AgentResult:
        """
        Fetch fundamental data for all tickers.
        
        Args:
            universe: List of tickers to fetch
            ctx: Run context
        
        Returns:
            AgentResult with DataFrame indexed by ticker containing
            fundamental data columns
        """
        start_time = time.time()
        errors = []
        
        ctx.logger.info(f"Fetching fundamentals for {len(universe)} tickers")
        
        # Get provider
        try:
            provider = get_provider(ctx.config.primary_provider, ctx)
        except ValueError as e:
            errors.append(str(e))
            return self._create_result(pd.DataFrame(), len(universe), 0, errors)
        
        # Fetch data for all tickers concurrently
        semaphore = asyncio.Semaphore(3)  # Limit concurrency more for fundamentals
        
        async def fetch_ticker(ticker: str) -> tuple[str, dict | None, str | None]:
            async with semaphore:
                response = await provider.get_fundamentals(ticker, ctx)
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
            
            ticker, data, error = result
            if error:
                errors.append(f"{ticker}: {error}")
                # Add empty row for missing tickers (graceful degradation)
                all_data.append({'ticker': ticker})
            elif data is not None:
                row = {'ticker': ticker, **data}
                all_data.append(row)
        
        # Create DataFrame
        if all_data:
            df = pd.DataFrame(all_data).set_index('ticker')
            # Ensure expected columns exist (fill with None if missing)
            expected_cols = ['market_cap', 'pe_ratio', 'eps', 'sector']
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = None
        else:
            df = pd.DataFrame()
        
        latency_ms = (time.time() - start_time) * 1000
        successful = len([r for r in results if not isinstance(r, Exception) and r[1] is not None])
        ctx.logger.info(
            f"Fundamentals: fetched {successful}/{len(universe)} tickers "
            f"in {latency_ms:.0f}ms"
        )
        
        return self._create_result(df, len(universe), latency_ms, errors)
    
    def compute_fundamentals_score(
        self,
        data: pd.DataFrame,
        weights: dict[str, float] | None = None
    ) -> pd.Series:
        """
        Compute a composite fundamentals score for each ticker.
        
        Higher score = more attractive fundamentals.
        
        Args:
            data: Fundamentals DataFrame
            weights: Optional weights for each factor
        
        Returns:
            Series of scores indexed by ticker
        """
        if data.empty:
            return pd.Series(dtype=float)
        
        weights = weights or {
            'pe_score': 0.4,
            'eps_score': 0.3,
            'size_score': 0.3
        }
        
        scores = pd.DataFrame(index=data.index)
        
        # PE Score (lower is better, inverse and normalize)
        if 'pe_ratio' in data.columns:
            pe = pd.to_numeric(data['pe_ratio'], errors='coerce')
            # Cap PE at reasonable bounds
            pe = pe.clip(0, 100)
            # Inverse (lower PE = higher score)
            pe_score = 1 - (pe / pe.max())
            pe_score = pe_score.fillna(0.5)  # Neutral for missing
            scores['pe_score'] = pe_score
        else:
            scores['pe_score'] = 0.5
        
        # EPS Score (higher is better)
        if 'eps' in data.columns:
            eps = pd.to_numeric(data['eps'], errors='coerce')
            # Normalize
            eps_min, eps_max = eps.min(), eps.max()
            if eps_max > eps_min:
                eps_score = (eps - eps_min) / (eps_max - eps_min)
            else:
                eps_score = 0.5
            eps_score = eps_score.fillna(0.5)
            scores['eps_score'] = eps_score
        else:
            scores['eps_score'] = 0.5
        
        # Size Score (log of market cap, normalized)
        if 'market_cap' in data.columns:
            mc = pd.to_numeric(data['market_cap'], errors='coerce')
            log_mc = mc.apply(lambda x: np.log10(x) if x and x > 0 else 0)
            mc_min, mc_max = log_mc.min(), log_mc.max()
            if mc_max > mc_min:
                size_score = (log_mc - mc_min) / (mc_max - mc_min)
            else:
                size_score = 0.5
            size_score = size_score.fillna(0.5)
            scores['size_score'] = size_score
        else:
            scores['size_score'] = 0.5
        
        # Weighted composite
        composite = sum(
            scores[col] * weight
            for col, weight in weights.items()
            if col in scores.columns
        )
        
        return composite
