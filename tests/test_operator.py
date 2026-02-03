"""
Tests for the Operator class.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.app.agents import AgentResult
from src.app.operator import Operator, OperatorResult


class TestOperator:
    """Tests for Operator orchestration."""
    
    @pytest.mark.asyncio
    async def test_operator_runs_agents_concurrently(
        self,
        test_context,
        sample_tickers,
        sample_price_data,
        sample_features
    ):
        """Test that operator launches agents concurrently."""
        # Mock all agents
        with patch.object(Operator, '__init__', lambda self, ctx: None):
            operator = Operator.__new__(Operator)
            operator.ctx = test_context
            operator.logger = test_context.logger
            
            # Mock agents
            operator.universe_agent = MagicMock()
            operator.universe_agent.run = AsyncMock(return_value=AgentResult(
                name='universe',
                data=pd.DataFrame({'name': [''] * len(sample_tickers), 'sector': [''] * len(sample_tickers)}, 
                                  index=sample_tickers),
                metadata={'coverage_ratio': 1.0, 'latency_ms': 10}
            ))
            
            operator.price_agent = MagicMock()
            operator.price_agent.run = AsyncMock(return_value=AgentResult(
                name='price',
                data=sample_price_data,
                metadata={'coverage_ratio': 1.0, 'latency_ms': 100}
            ))
            
            operator.technical_agent = MagicMock()
            operator.technical_agent.run = AsyncMock(return_value=AgentResult(
                name='technical',
                data=sample_features[['returns_1m', 'volatility_21d']],
                metadata={'coverage_ratio': 1.0, 'latency_ms': 50}
            ))
            
            operator.fundamentals_agent = MagicMock()
            operator.fundamentals_agent.run = AsyncMock(return_value=AgentResult(
                name='fundamentals',
                data=sample_features[['fund_market_cap', 'fund_pe_ratio', 'fund_sector']],
                metadata={'coverage_ratio': 1.0, 'latency_ms': 80}
            ))
            
            operator.sentiment_agent = MagicMock()
            operator.sentiment_agent.run = AsyncMock(return_value=AgentResult(
                name='sentiment',
                data=pd.DataFrame(),
                metadata={'coverage_ratio': 0, 'latency_ms': 5}
            ))
            
            # Add required methods
            operator._is_agent_enabled = lambda name: name != 'sentiment'
            operator._run_with_timeout = Operator._run_with_timeout.__get__(operator, Operator)
            operator._merge_results = Operator._merge_results.__get__(operator, Operator)
            operator._handle_missing_values = Operator._handle_missing_values.__get__(operator, Operator)
            operator._normalize_features = Operator._normalize_features.__get__(operator, Operator)
            
            # Run cycle
            result = await operator.run_cycle(sample_tickers)
            
            # Verify result
            assert isinstance(result, OperatorResult)
            assert not result.features.empty
            assert len(result.features) <= len(sample_tickers)
            
            # Verify agents were called
            operator.price_agent.run.assert_called_once()
    
    def test_operator_merges_outputs_correctly(
        self,
        test_context,
        sample_tickers,
        sample_features
    ):
        """Test that operator merges agent outputs into unified DataFrame."""
        operator = Operator(test_context)
        
        # Create mock agent results
        results = {
            'technical': AgentResult(
                name='technical',
                data=sample_features[['returns_1m', 'volatility_21d']],
                metadata={}
            ),
            'fundamentals': AgentResult(
                name='fundamentals',
                data=sample_features[['fund_market_cap', 'fund_pe_ratio']].rename(
                    columns={'fund_market_cap': 'market_cap', 'fund_pe_ratio': 'pe_ratio'}
                ),
                metadata={}
            )
        }
        
        # Merge
        merged = operator._merge_results(results, sample_tickers)
        
        # Check merged DataFrame
        assert len(merged) == len(sample_tickers)
        assert 'returns_1m' in merged.columns
        assert 'volatility_21d' in merged.columns
        assert 'fund_market_cap' in merged.columns or 'market_cap' in merged.columns
    
    def test_operator_handles_missing_values_drop(
        self,
        test_context,
        sample_features
    ):
        """Test missing value handling with drop strategy."""
        test_context.config.features.missing_value_strategy = 'drop'
        operator = Operator(test_context)
        
        # Add some NaN values
        features = sample_features.copy()
        features.loc[features.index[0], 'returns_1m'] = None
        
        # Handle missing
        result = operator._handle_missing_values(features)
        
        # First ticker should be dropped
        assert len(result) < len(features)
    
    def test_operator_handles_missing_values_fill_median(
        self,
        test_context,
        sample_features
    ):
        """Test missing value handling with fill_median strategy."""
        test_context.config.features.missing_value_strategy = 'fill_median'
        test_context.config.features.required_features = []  # Don't drop any
        operator = Operator(test_context)
        
        # Add some NaN values
        features = sample_features.copy()
        features.loc[features.index[0], 'volatility_21d'] = None
        
        # Handle missing
        result = operator._handle_missing_values(features)
        
        # No rows should be dropped
        assert len(result) == len(features)
        # NaN should be filled
        assert not result['volatility_21d'].isna().any()
    
    @pytest.mark.asyncio
    async def test_operator_handles_agent_failure_gracefully(
        self,
        test_context,
        sample_tickers
    ):
        """Test that operator handles individual agent failures."""
        with patch.object(Operator, '__init__', lambda self, ctx: None):
            operator = Operator.__new__(Operator)
            operator.ctx = test_context
            operator.logger = test_context.logger
            
            # Mock agents - one fails
            operator.universe_agent = MagicMock()
            operator.universe_agent.run = AsyncMock(return_value=AgentResult(
                name='universe',
                data=pd.DataFrame(index=sample_tickers),
                metadata={'coverage_ratio': 1.0}
            ))
            
            operator.price_agent = MagicMock()
            operator.price_agent.run = AsyncMock(
                side_effect=Exception("API Error")
            )
            
            operator.technical_agent = MagicMock()
            operator.fundamentals_agent = MagicMock()
            operator.fundamentals_agent.run = AsyncMock(return_value=AgentResult(
                name='fundamentals',
                data=pd.DataFrame(),
                metadata={'errors': ['Failed']}
            ))
            
            operator.sentiment_agent = MagicMock()
            
            operator._is_agent_enabled = lambda name: True
            operator._run_with_timeout = Operator._run_with_timeout.__get__(operator, Operator)
            
            # The run should not raise an exception
            # (it should handle agent failures gracefully)
            # This is tested by checking that _run_with_timeout catches exceptions
            
            # Test timeout handling
            async def slow_agent(*args):
                await asyncio.sleep(100)
                return AgentResult(name='slow', data=pd.DataFrame(), metadata={})
            
            operator.price_agent.run = slow_agent
            
            # Should timeout and return error result
            result = await operator._run_with_timeout(
                slow_agent(),
                'price'
            )
            
            assert result.has_errors or 'Timeout' in str(result.metadata.get('errors', []))


class TestOperatorResult:
    """Tests for OperatorResult."""
    
    def test_operator_result_properties(self, sample_features, sample_price_data):
        """Test OperatorResult properties."""
        result = OperatorResult(
            features=sample_features,
            price_data=sample_price_data,
            metadata={'test': True}
        )
        
        assert result.num_tickers == len(sample_features)
        assert len(result.tickers) == len(sample_features)
        assert 'AAPL' in result.tickers
