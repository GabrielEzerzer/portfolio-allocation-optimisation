"""
Tests for agent implementations.
"""

import numpy as np
import pandas as pd
import pytest

from src.app.agents import (
    AgentResult,
    FundamentalsAgent,
    PriceHistoryAgent,
    SentimentAgent,
    TechnicalIndicatorAgent,
    UniverseAgent,
)


class TestAgentResult:
    """Tests for AgentResult dataclass."""
    
    def test_agent_result_creation(self, sample_features):
        """Test creating an AgentResult."""
        result = AgentResult(
            name='test',
            data=sample_features,
            metadata={'coverage_ratio': 1.0}
        )
        
        assert result.name == 'test'
        assert not result.data.empty
        assert 'timestamp' in result.metadata
    
    def test_agent_result_coverage(self, sample_features):
        """Test coverage property."""
        result = AgentResult(
            name='test',
            data=sample_features,
            metadata={'coverage_ratio': 0.8}
        )
        
        assert result.coverage == 0.8
    
    def test_agent_result_errors(self):
        """Test error handling in AgentResult."""
        result = AgentResult(
            name='test',
            data=pd.DataFrame(),
            metadata={'errors': ['Error 1', 'Error 2']}
        )
        
        assert result.has_errors
        assert len(result.errors) == 2
    
    def test_agent_result_add_error(self, sample_features):
        """Test adding errors to result."""
        result = AgentResult(name='test', data=sample_features)
        
        result.add_error('New error')
        
        assert 'New error' in result.errors


class TestUniverseAgent:
    """Tests for UniverseAgent."""
    
    @pytest.mark.asyncio
    async def test_universe_agent_with_provided_tickers(
        self,
        test_context,
        sample_tickers
    ):
        """Test that provided tickers are returned."""
        agent = UniverseAgent()
        
        result = await agent.run(sample_tickers, test_context)
        
        assert result.name == 'universe'
        assert len(result.data) == len(sample_tickers)
        assert all(t in result.data.index for t in sample_tickers)
    
    @pytest.mark.asyncio
    async def test_universe_agent_respects_size_limit(
        self,
        test_context,
        sample_tickers
    ):
        """Test universe size limiting."""
        test_context.config.universe_size = 5
        agent = UniverseAgent()
        
        result = await agent.run(sample_tickers, test_context)
        
        assert len(result.data) == 5


class TestTechnicalIndicatorAgent:
    """Tests for TechnicalIndicatorAgent."""
    
    @pytest.mark.asyncio
    async def test_technical_agent_computes_indicators(
        self,
        test_context,
        sample_tickers,
        sample_price_data
    ):
        """Test that technical agent computes expected indicators."""
        agent = TechnicalIndicatorAgent()
        
        result = await agent.run(
            sample_tickers,
            test_context,
            price_data=sample_price_data
        )
        
        assert result.name == 'technical_indicators'
        assert not result.data.empty
        
        # Check expected columns exist
        assert 'returns_1m' in result.data.columns
        assert 'volatility_21d' in result.data.columns
    
    @pytest.mark.asyncio
    async def test_technical_agent_handles_empty_price_data(
        self,
        test_context,
        sample_tickers
    ):
        """Test graceful handling of empty price data."""
        agent = TechnicalIndicatorAgent()
        
        result = await agent.run(
            sample_tickers,
            test_context,
            price_data=pd.DataFrame()
        )
        
        assert result.data.empty
        assert 'No price data' in str(result.errors)
    
    def test_compute_returns(self, sample_close_prices):
        """Test returns calculation."""
        agent = TechnicalIndicatorAgent()
        
        close = sample_close_prices['AAPL']
        returns = agent._compute_returns(close, 21)
        
        assert returns is not None
        assert isinstance(returns, float)
    
    def test_compute_volatility(self, sample_close_prices):
        """Test volatility calculation."""
        agent = TechnicalIndicatorAgent()
        
        close = sample_close_prices['AAPL']
        volatility = agent._compute_volatility(close, 21)
        
        assert volatility is not None
        assert volatility >= 0
    
    def test_compute_momentum(self, sample_close_prices):
        """Test momentum calculation."""
        agent = TechnicalIndicatorAgent()
        
        close = sample_close_prices['AAPL']
        momentum = agent._compute_momentum(close, 50)
        
        assert momentum is not None
        assert isinstance(momentum, float)
    
    def test_compute_rsi(self, sample_close_prices):
        """Test RSI calculation."""
        agent = TechnicalIndicatorAgent()
        
        close = sample_close_prices['AAPL']
        rsi = agent._compute_rsi(close, 14)
        
        assert rsi is not None
        assert 0 <= rsi <= 100


class TestSentimentAgent:
    """Tests for SentimentAgent."""
    
    @pytest.mark.asyncio
    async def test_sentiment_agent_generates_synthetic_data(
        self,
        test_context,
        sample_tickers
    ):
        """Test that sentiment agent generates synthetic data."""
        test_context.config.agents['sentiment'].enabled = True
        agent = SentimentAgent()
        
        result = await agent.run(sample_tickers, test_context)
        
        assert result.name == 'sentiment'
        assert len(result.data) == len(sample_tickers)
        assert 'sentiment_score' in result.data.columns
        assert 'analyst_rating' in result.data.columns
    
    @pytest.mark.asyncio
    async def test_sentiment_agent_is_deterministic_with_seed(
        self,
        test_context,
        sample_tickers
    ):
        """Test that sentiment is deterministic with same seed."""
        test_context.config.agents['sentiment'].enabled = True
        test_context.random_seed = 42
        
        agent = SentimentAgent()
        
        result1 = await agent.run(sample_tickers, test_context)
        result2 = await agent.run(sample_tickers, test_context)
        
        # Same seed should produce same results
        pd.testing.assert_frame_equal(result1.data, result2.data)
    
    @pytest.mark.asyncio
    async def test_sentiment_agent_returns_empty_when_disabled(
        self,
        test_context,
        sample_tickers
    ):
        """Test that disabled sentiment agent returns empty."""
        test_context.config.agents['sentiment'].enabled = False
        agent = SentimentAgent()
        
        result = await agent.run(sample_tickers, test_context)
        
        assert result.data.empty
    
    def test_compute_sentiment_score(self, sample_tickers):
        """Test composite sentiment score calculation."""
        agent = SentimentAgent()
        
        # Create sample sentiment data
        data = pd.DataFrame({
            'sentiment_score': np.random.uniform(-1, 1, len(sample_tickers)),
            'analyst_rating': np.random.uniform(1, 5, len(sample_tickers)),
            'news_volume': np.random.randint(1, 100, len(sample_tickers)),
            'social_mentions': np.random.randint(1, 1000, len(sample_tickers))
        }, index=sample_tickers)
        
        scores = agent.compute_sentiment_score(data)
        
        assert len(scores) == len(sample_tickers)
        assert all(0 <= s <= 1 for s in scores)


class TestFundamentalsAgent:
    """Tests for FundamentalsAgent."""
    
    def test_compute_fundamentals_score(self, sample_features):
        """Test composite fundamentals score calculation."""
        agent = FundamentalsAgent()
        
        # Rename columns to match expected format
        data = sample_features.rename(columns={
            'fund_market_cap': 'market_cap',
            'fund_pe_ratio': 'pe_ratio'
        })
        
        scores = agent.compute_fundamentals_score(data)
        
        assert len(scores) == len(sample_features)


class TestPriceHistoryAgent:
    """Tests for PriceHistoryAgent."""
    
    def test_get_close_prices(self, sample_price_data, sample_tickers):
        """Test extracting close prices as pivot table."""
        agent = PriceHistoryAgent()
        
        close_prices = agent.get_close_prices(sample_price_data)
        
        assert not close_prices.empty
        assert all(t in close_prices.columns for t in sample_tickers)
        # Index should be dates
        assert close_prices.index.name == 'date'
    
    def test_get_prices_for_ticker(self, sample_price_data):
        """Test extracting data for single ticker."""
        agent = PriceHistoryAgent()
        
        aapl_data = agent.get_prices_for_ticker(sample_price_data, 'AAPL')
        
        assert not aapl_data.empty
        assert 'close' in aapl_data.columns
        assert 'volume' in aapl_data.columns
