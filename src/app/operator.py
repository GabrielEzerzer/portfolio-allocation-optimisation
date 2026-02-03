"""
Operator - orchestrates runtime agents and merges their outputs.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .agents import (
    AgentResult,
    FundamentalsAgent,
    PriceHistoryAgent,
    SentimentAgent,
    TechnicalIndicatorAgent,
    UniverseAgent,
)
from .context import RunContext


@dataclass
class OperatorResult:
    """Result from a complete operator run cycle."""
    features: pd.DataFrame  # Unified feature table indexed by ticker
    price_data: pd.DataFrame  # Raw price data (multi-indexed)
    metadata: dict = field(default_factory=dict)
    
    @property
    def tickers(self) -> list[str]:
        """Get list of tickers in the result."""
        return list(self.features.index)
    
    @property
    def num_tickers(self) -> int:
        """Get number of tickers."""
        return len(self.features)


class Operator:
    """
    Orchestrates runtime agents and merges their outputs.
    
    The Operator:
    1. Launches all agents concurrently
    2. Enforces per-agent timeouts
    3. Collects and logs per-agent timings and coverage
    4. Merges outputs into a unified feature table
    5. Applies feature normalization
    6. Handles missing values according to config
    """
    
    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.logger = ctx.logger
        
        # Initialize agents
        self.universe_agent = UniverseAgent()
        self.price_agent = PriceHistoryAgent()
        self.technical_agent = TechnicalIndicatorAgent()
        self.fundamentals_agent = FundamentalsAgent()
        self.sentiment_agent = SentimentAgent()
    
    async def run_cycle(
        self,
        universe: list[str] | None = None
    ) -> OperatorResult:
        """
        Run a complete data gathering and feature computation cycle.
        
        Args:
            universe: Optional pre-specified universe of tickers.
                     If None, the universe agent will provide it.
        
        Returns:
            OperatorResult with unified feature table and metadata
        """
        start_time = time.time()
        agent_results: dict[str, AgentResult] = {}
        
        # Step 1: Get universe
        self.logger.info("Step 1: Getting universe...")
        universe_result = await self._run_with_timeout(
            self.universe_agent.run(universe or [], self.ctx),
            'universe'
        )
        agent_results['universe'] = universe_result
        
        if universe_result.data.empty and not universe:
            raise ValueError("No universe available - cannot proceed")
        
        # Use provided universe or agent result
        if universe:
            active_universe = universe
        else:
            active_universe = list(universe_result.data.index)
        
        self.logger.info(f"Active universe: {len(active_universe)} tickers")
        
        # Step 2: Fetch data in parallel (price + fundamentals + sentiment)
        self.logger.info("Step 2: Fetching data from agents in parallel...")
        
        agent_tasks = {
            'price': self._run_with_timeout(
                self.price_agent.run(active_universe, self.ctx),
                'price'
            ),
            'fundamentals': self._run_with_timeout(
                self.fundamentals_agent.run(active_universe, self.ctx),
                'fundamentals'
            ) if self._is_agent_enabled('fundamentals') else None,
            'sentiment': self._run_with_timeout(
                self.sentiment_agent.run(active_universe, self.ctx),
                'sentiment'
            ) if self._is_agent_enabled('sentiment') else None,
        }
        
        # Filter out disabled agents
        active_tasks = {k: v for k, v in agent_tasks.items() if v is not None}
        
        # Run all tasks concurrently
        results = await asyncio.gather(
            *active_tasks.values(),
            return_exceptions=True
        )
        
        # Map results back to agent names
        for name, result in zip(active_tasks.keys(), results):
            if isinstance(result, Exception):
                self.logger.error(f"Agent {name} failed: {result}")
                agent_results[name] = AgentResult(
                    name=name,
                    data=pd.DataFrame(),
                    metadata={'errors': [str(result)]}
                )
            else:
                agent_results[name] = result
        
        # Step 3: Compute technical indicators (after prices are available)
        price_result = agent_results.get('price')
        if price_result and not price_result.data.empty:
            self.logger.info("Step 3: Computing technical indicators...")
            technical_result = await self._run_with_timeout(
                self.technical_agent.run(active_universe, self.ctx, price_result.data),
                'technical'
            )
            agent_results['technical'] = technical_result
        else:
            self.logger.warning("Skipping technical indicators - no price data")
            agent_results['technical'] = AgentResult(
                name='technical',
                data=pd.DataFrame(),
                metadata={'errors': ['No price data available']}
            )
        
        # Step 4: Merge all agent outputs
        self.logger.info("Step 4: Merging agent outputs...")
        features = self._merge_results(agent_results, active_universe)
        
        # Step 5: Handle missing values
        features = self._handle_missing_values(features)
        
        # Step 6: Normalize features
        if self.ctx.config.features.normalize:
            features = self._normalize_features(features)
        
        # Collect metadata
        total_time = (time.time() - start_time) * 1000
        metadata = {
            'total_latency_ms': total_time,
            'num_tickers_input': len(active_universe),
            'num_tickers_output': len(features),
            'agent_stats': {
                name: {
                    'coverage': result.metadata.get('coverage_ratio', 0),
                    'latency_ms': result.metadata.get('latency_ms', 0),
                    'errors': len(result.errors)
                }
                for name, result in agent_results.items()
            }
        }
        
        self.logger.info(
            f"Operator cycle complete: {len(features)} tickers, "
            f"{total_time:.0f}ms total"
        )
        
        # Log agent stats
        for name, stats in metadata['agent_stats'].items():
            self.logger.debug(
                f"  {name}: coverage={stats['coverage']:.1%}, "
                f"latency={stats['latency_ms']:.0f}ms, "
                f"errors={stats['errors']}"
            )
        
        price_data = agent_results.get('price', AgentResult('price', pd.DataFrame())).data
        
        return OperatorResult(
            features=features,
            price_data=price_data,
            metadata=metadata
        )
    
    def _is_agent_enabled(self, agent_name: str) -> bool:
        """Check if an agent is enabled in config."""
        agent_config = self.ctx.config.agents.get(agent_name)
        return agent_config.enabled if agent_config else True
    
    async def _run_with_timeout(
        self,
        coro,
        agent_name: str
    ) -> AgentResult:
        """Run an agent coroutine with timeout."""
        agent_config = self.ctx.config.agents.get(agent_name)
        timeout = agent_config.timeout_seconds if agent_config else 60
        
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"Agent {agent_name} timed out after {timeout}s")
            return AgentResult(
                name=agent_name,
                data=pd.DataFrame(),
                metadata={'errors': [f'Timeout after {timeout}s']}
            )
    
    def _merge_results(
        self,
        results: dict[str, AgentResult],
        universe: list[str]
    ) -> pd.DataFrame:
        """
        Merge all agent results into a unified feature table.
        
        Args:
            results: Dict of agent name -> AgentResult
            universe: List of tickers
        
        Returns:
            DataFrame indexed by ticker with all features
        """
        # Start with universe as base
        merged = pd.DataFrame(index=universe)
        merged.index.name = 'ticker'
        
        # Add technical indicators
        technical = results.get('technical')
        if technical and not technical.data.empty:
            for col in technical.data.columns:
                merged[col] = technical.data[col]
        
        # Add fundamentals
        fundamentals = results.get('fundamentals')
        if fundamentals and not fundamentals.data.empty:
            for col in ['market_cap', 'pe_ratio', 'eps', 'sector']:
                if col in fundamentals.data.columns:
                    merged[f'fund_{col}'] = fundamentals.data[col]
        
        # Add sentiment
        sentiment = results.get('sentiment')
        if sentiment and not sentiment.data.empty:
            for col in ['sentiment_score', 'analyst_rating']:
                if col in sentiment.data.columns:
                    merged[f'sent_{col}'] = sentiment.data[col]
        
        return merged
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to configuration."""
        strategy = self.ctx.config.features.missing_value_strategy
        required = self.ctx.config.features.required_features
        
        # First, drop rows missing required features
        if required:
            for feat in required:
                if feat in features.columns:
                    features = features.dropna(subset=[feat])
        
        # Handle remaining missing values
        if strategy == 'drop':
            features = features.dropna()
        elif strategy == 'fill_median':
            for col in features.select_dtypes(include=[np.number]).columns:
                features[col] = features[col].fillna(features[col].median())
        elif strategy == 'fill_zero':
            features = features.fillna(0)
        # 'use_cache' would require additional implementation
        
        return features
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric features."""
        method = self.ctx.config.features.normalization_method
        
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'zscore':
                mean = features[col].mean()
                std = features[col].std()
                if std > 0:
                    features[f'{col}_norm'] = (features[col] - mean) / std
            elif method == 'minmax':
                min_val = features[col].min()
                max_val = features[col].max()
                if max_val > min_val:
                    features[f'{col}_norm'] = (features[col] - min_val) / (max_val - min_val)
        
        return features
