"""
Universe Agent - provides the list of tickers to process.
Supports S&P 500 fetching with fallback to seed file.
"""

import time
from pathlib import Path

import pandas as pd

from ..context import RunContext
from .base import AgentResult, BaseAgent


class UniverseAgent(BaseAgent):
    """
    Agent responsible for providing the universe of tickers.
    
    Priority:
    1. CLI override (--tickers)
    2. API fetch (if available and in live mode)
    3. Fallback seed file
    """
    
    @property
    def name(self) -> str:
        return "universe"
    
    async def run(
        self,
        universe: list[str],
        ctx: 'RunContext'
    ) -> AgentResult:
        """
        Get the ticker universe.
        
        If universe is already provided (from CLI), validate and return it.
        Otherwise, load from seed file or fetch from API.
        
        Args:
            universe: Pre-provided ticker list (may be empty)
            ctx: Run context
        
        Returns:
            AgentResult with tickers as index and name/sector columns
        """
        start_time = time.time()
        errors = []
        
        # If universe already provided (from CLI), use it
        if universe:
            ctx.logger.info(f"Using provided universe of {len(universe)} tickers")
            data = pd.DataFrame({
                'ticker': universe,
                'name': [''] * len(universe),
                'sector': [''] * len(universe)
            }).set_index('ticker')
            
            # Apply universe size limit if configured
            if ctx.config.universe_size:
                data = data.head(ctx.config.universe_size)
            
            latency_ms = (time.time() - start_time) * 1000
            return self._create_result(data, len(universe), latency_ms, errors)
        
        # Try to load from seed file
        data = await self._load_from_seed(ctx)
        
        if data.empty:
            errors.append("Failed to load universe from any source")
        else:
            # Apply universe size limit if configured
            if ctx.config.universe_size:
                data = data.head(ctx.config.universe_size)
            ctx.logger.info(f"Loaded universe of {len(data)} tickers")
        
        latency_ms = (time.time() - start_time) * 1000
        return self._create_result(data, len(data), latency_ms, errors)
    
    async def _load_from_seed(self, ctx: 'RunContext') -> pd.DataFrame:
        """Load universe from seed CSV file."""
        agent_config = ctx.config.agents.get('universe')
        if not agent_config or not agent_config.fallback_file:
            ctx.logger.warning("No seed file configured")
            return pd.DataFrame()
        
        seed_path = Path(agent_config.fallback_file)
        if not seed_path.exists():
            ctx.logger.warning(f"Seed file not found: {seed_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(seed_path)
            
            # Ensure required columns exist
            if 'ticker' not in df.columns:
                ctx.logger.error("Seed file missing 'ticker' column")
                return pd.DataFrame()
            
            # Set ticker as index
            df = df.set_index('ticker')
            
            # Ensure name and sector columns exist
            if 'name' not in df.columns:
                df['name'] = ''
            if 'sector' not in df.columns:
                df['sector'] = ''
            
            ctx.logger.info(f"Loaded {len(df)} tickers from seed file")
            return df
            
        except Exception as e:
            ctx.logger.error(f"Error loading seed file: {e}")
            return pd.DataFrame()
