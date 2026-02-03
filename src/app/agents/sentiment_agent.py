"""
Sentiment Agent - provides sentiment data for tickers.
Stubbed for MVP - returns synthetic/cached sentiment.
"""

import time
from hashlib import md5

import numpy as np
import pandas as pd

from ..context import RunContext
from .base import AgentResult, BaseAgent


class SentimentAgent(BaseAgent):
    """
    Agent responsible for providing sentiment data.
    
    In MVP: Returns synthetic sentiment scores based on ticker hash
    for deterministic behavior. Pluggable for future real implementation.
    """
    
    @property
    def name(self) -> str:
        return "sentiment"
    
    async def run(
        self,
        universe: list[str],
        ctx: 'RunContext'
    ) -> AgentResult:
        """
        Get sentiment data for all tickers.
        
        In MVP, this returns synthetic sentiment scores that are
        deterministic based on ticker name and optional random seed.
        
        Args:
            universe: List of tickers
            ctx: Run context
        
        Returns:
            AgentResult with DataFrame indexed by ticker containing
            sentiment columns
        """
        start_time = time.time()
        errors = []
        
        ctx.logger.info(f"Generating sentiment data for {len(universe)} tickers")
        
        # Check if sentiment is enabled
        agent_config = ctx.config.agents.get('sentiment')
        if agent_config and not agent_config.enabled:
            ctx.logger.info("Sentiment agent is disabled, returning empty result")
            return self._create_result(pd.DataFrame(), len(universe), 0, errors)
        
        # Generate synthetic sentiment
        rng = np.random.default_rng(ctx.random_seed or 42)
        
        results = []
        for ticker in universe:
            # Use ticker hash + seed for deterministic but varied values
            ticker_seed = int(md5(ticker.encode()).hexdigest()[:8], 16)
            ticker_rng = np.random.default_rng(ticker_seed + (ctx.random_seed or 0))
            
            # Generate sentiment scores (-1 to 1, neutral-biased)
            raw_sentiment = ticker_rng.normal(0, 0.3)
            sentiment_score = max(-1, min(1, raw_sentiment))
            
            # Generate news volume (log-normal)
            news_volume = int(ticker_rng.lognormal(2, 1))
            
            # Generate social mentions (log-normal)
            social_mentions = int(ticker_rng.lognormal(3, 1.5))
            
            # Analyst rating (1-5 scale, normally distributed around 3)
            analyst_rating = max(1, min(5, ticker_rng.normal(3, 0.8)))
            
            results.append({
                'ticker': ticker,
                'sentiment_score': sentiment_score,
                'news_volume': news_volume,
                'social_mentions': social_mentions,
                'analyst_rating': analyst_rating,
                'sentiment_source': 'synthetic'
            })
        
        df = pd.DataFrame(results).set_index('ticker')
        
        latency_ms = (time.time() - start_time) * 1000
        ctx.logger.info(
            f"Sentiment: generated for {len(universe)} tickers in {latency_ms:.0f}ms"
        )
        
        return self._create_result(df, len(universe), latency_ms, errors)
    
    def compute_sentiment_score(
        self,
        data: pd.DataFrame,
        weights: dict[str, float] | None = None
    ) -> pd.Series:
        """
        Compute a composite sentiment score for optimization.
        
        Args:
            data: Sentiment DataFrame
            weights: Optional weights for each factor
        
        Returns:
            Series of normalized scores (0-1) indexed by ticker
        """
        if data.empty:
            return pd.Series(dtype=float)
        
        weights = weights or {
            'sentiment_score': 0.5,
            'analyst_rating': 0.3,
            'news_volume': 0.1,
            'social_mentions': 0.1
        }
        
        scores = pd.DataFrame(index=data.index)
        
        # Normalize sentiment (-1 to 1) to (0 to 1)
        if 'sentiment_score' in data.columns:
            scores['sentiment_score'] = (data['sentiment_score'] + 1) / 2
        else:
            scores['sentiment_score'] = 0.5
        
        # Normalize analyst rating (1-5) to (0-1)
        if 'analyst_rating' in data.columns:
            scores['analyst_rating'] = (data['analyst_rating'] - 1) / 4
        else:
            scores['analyst_rating'] = 0.5
        
        # Normalize news volume (log-scale)
        if 'news_volume' in data.columns:
            log_vol = np.log1p(data['news_volume'])
            vol_max = log_vol.max() if log_vol.max() > 0 else 1
            scores['news_volume'] = log_vol / vol_max
        else:
            scores['news_volume'] = 0.5
        
        # Normalize social mentions (log-scale)
        if 'social_mentions' in data.columns:
            log_social = np.log1p(data['social_mentions'])
            social_max = log_social.max() if log_social.max() > 0 else 1
            scores['social_mentions'] = log_social / social_max
        else:
            scores['social_mentions'] = 0.5
        
        # Weighted composite
        composite = sum(
            scores[col] * weight
            for col, weight in weights.items()
            if col in scores.columns
        )
        
        return composite.fillna(0.5)
