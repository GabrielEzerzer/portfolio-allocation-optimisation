"""
Main entry point for ACO Portfolio Optimizer.
"""

import argparse
import asyncio
import json
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

import aiohttp
import numpy as np
from aiolimiter import AsyncLimiter

from .backtesting import BacktestResult, WalkForwardBacktester
from .config import Config, load_config
from .context import RunContext
from .logging_setup import setup_logging
from .operator import Operator, OperatorResult
from .optimization import ACOOptimizer, Portfolio
from .utils import CacheManager, format_date, parse_date


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ACO Portfolio Optimizer - Ant Colony Optimization for S&P 500 portfolios',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live mode with specific tickers
  python -m src.app.main --mode live --tickers AAPL,MSFT,GOOGL

  # Cached mode (no API calls)
  python -m src.app.main --mode cached --tickers AAPL,MSFT,GOOGL

  # Backtest mode
  python -m src.app.main --mode backtest --start 2023-01-01 --end 2024-01-01

  # With custom config and seed
  python -m src.app.main --mode live --config config/settings.local.yaml --seed 42
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['live', 'cached', 'backtest'],
        required=True,
        help='Execution mode: live (API calls), cached (use cache), backtest (walk-forward)'
    )
    
    parser.add_argument(
        '--config',
        default='config/settings.example.yaml',
        help='Path to configuration file (default: config/settings.example.yaml)'
    )
    
    parser.add_argument(
        '--tickers',
        type=str,
        help='Comma-separated list of tickers (overrides universe)'
    )
    
    parser.add_argument(
        '--universe-size',
        type=int,
        help='Limit universe to N tickers (for debugging)'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD) for backtest or data range'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD) for backtest or data range'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for deterministic runs'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for results (overrides config)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


async def run_live_cycle(
    ctx: RunContext,
    universe: list[str] | None
) -> tuple[OperatorResult, Portfolio]:
    """Run a single live optimization cycle."""
    ctx.logger.info("Starting live optimization cycle...")
    
    # Create operator and run
    operator = Operator(ctx)
    result = await operator.run_cycle(universe)
    
    if result.features.empty:
        raise ValueError("No features available for optimization")
    
    ctx.logger.info(f"Features computed for {result.num_tickers} tickers")
    
    # Get returns for fitness calculation
    from .agents import PriceHistoryAgent
    price_agent = PriceHistoryAgent()
    close_prices = price_agent.get_close_prices(result.price_data)
    returns = close_prices.pct_change().dropna()
    
    # Run ACO optimization
    ctx.logger.info("Running ACO optimization...")
    optimizer = ACOOptimizer(
        ctx.config.aco,
        ctx.config.constraints,
        rng=np.random.default_rng(ctx.random_seed)
    )
    
    # Get sectors if available
    sectors = {}
    if 'fund_sector' in result.features.columns:
        sectors = result.features['fund_sector'].to_dict()
    
    portfolio = optimizer.optimize(result.features, returns, sectors)
    
    ctx.logger.info(f"Optimization complete: {portfolio}")
    
    return result, portfolio


async def run_cached_cycle(
    ctx: RunContext,
    universe: list[str] | None
) -> tuple[OperatorResult, Portfolio]:
    """Run cycle using cached data only."""
    ctx.logger.info("Starting cached optimization cycle...")
    
    # Same as live but mode is 'cached'
    return await run_live_cycle(ctx, universe)


async def run_backtest(
    ctx: RunContext,
    universe: list[str] | None,
    start_date: date,
    end_date: date
) -> BacktestResult:
    """Run walk-forward backtest."""
    ctx.logger.info(f"Starting backtest from {start_date} to {end_date}...")
    
    # Get universe if not provided
    if not universe:
        from .agents import UniverseAgent
        agent = UniverseAgent()
        result = await agent.run([], ctx)
        universe = list(result.data.index)
    
    # Create backtester and run
    backtester = WalkForwardBacktester(ctx, ctx.config.backtest)
    result = await backtester.run(universe, start_date, end_date)
    
    return result


def print_portfolio(portfolio: Portfolio) -> None:
    """Print portfolio summary to console."""
    print("\n" + "=" * 60)
    print("PORTFOLIO OPTIMIZATION RESULT")
    print("=" * 60)
    
    print(f"\nFitness Score: {portfolio.fitness:.4f}")
    print(f"Number of Holdings: {portfolio.num_holdings}")
    print(f"Total Weight: {portfolio.total_weight:.2%}")
    print(f"Max Weight: {portfolio.max_weight:.2%}")
    
    print("\nTop 10 Holdings:")
    print("-" * 40)
    for ticker, weight in portfolio.top_holdings:
        print(f"  {ticker:8s} {weight:8.2%}")
    
    if 'annualized_return' in portfolio.diagnostics:
        print(f"\nExpected Annual Return: {portfolio.diagnostics['annualized_return']:.2%}")
    if 'annualized_volatility' in portfolio.diagnostics:
        print(f"Expected Volatility: {portfolio.diagnostics['annualized_volatility']:.2%}")
    if 'sharpe_ratio' in portfolio.diagnostics:
        print(f"Sharpe Ratio: {portfolio.diagnostics['sharpe_ratio']:.2f}")
    
    if portfolio.diagnostics.get('violations'):
        print("\nConstraint Violations:")
        for v in portfolio.diagnostics['violations']:
            print(f"  - {v}")
    
    print("\n" + "=" * 60)


def save_portfolio(portfolio: Portfolio, output_dir: str, filename: str = 'portfolio.json') -> Path:
    """Save portfolio to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    with open(filepath, 'w') as f:
        json.dump(portfolio.to_dict(), f, indent=2, default=str)
    
    return filepath


async def main_async() -> int:
    """Async main function."""
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Apply CLI overrides
    config.mode = args.mode
    
    if args.tickers:
        config.tickers = [t.strip() for t in args.tickers.split(',')]
    
    if args.universe_size:
        config.universe_size = args.universe_size
    
    if args.seed is not None:
        config.random_seed = args.seed
        config.aco.random_seed = args.seed
    
    if args.start:
        config.start_date = args.start
    
    if args.end:
        config.end_date = args.end
    
    if args.verbose:
        config.logging.level = 'DEBUG'
    
    # Setup logging
    logger = setup_logging(config.logging)
    logger.info(f"ACO Portfolio Optimizer starting in {args.mode} mode")
    
    # Create cache manager
    cache_manager = CacheManager(
        cache_dir=config.cache.path,
        default_ttl_days=config.cache.refresh_days,
        format=config.cache.format
    ) if config.cache.enabled else None
    
    # Create rate limiter
    provider_config = config.providers.get(config.primary_provider)
    rate_limit = provider_config.rate_limit_per_minute if provider_config else 5
    rate_limiter = AsyncLimiter(rate_limit, 60.0)
    
    # Create context
    ctx = RunContext(
        config=config,
        logger=logger,
        mode=args.mode,
        rate_limiter=rate_limiter,
        cache_manager=cache_manager,
        random_seed=config.random_seed or args.seed,
        run_id=str(uuid.uuid4())[:8]
    )
    
    # Parse dates if provided
    start_date = parse_date(args.start) if args.start else None
    end_date = parse_date(args.end) if args.end else date.today()
    
    ctx.start_date = start_date
    ctx.end_date = end_date
    
    # Get universe from CLI or config
    universe = config.tickers
    
    output_dir = args.output or config.output.path
    
    try:
        # Create aiohttp session for API calls
        async with aiohttp.ClientSession() as session:
            ctx.session = session if args.mode in ['live', 'backtest'] else None
            
            if args.mode == 'live':
                result, portfolio = await run_live_cycle(ctx, universe)
                print_portfolio(portfolio)
                
                if config.output.save_weights:
                    filepath = save_portfolio(portfolio, output_dir)
                    logger.info(f"Portfolio saved to {filepath}")
                
            elif args.mode == 'cached':
                result, portfolio = await run_cached_cycle(ctx, universe)
                print_portfolio(portfolio)
                
                if config.output.save_weights:
                    filepath = save_portfolio(portfolio, output_dir)
                    logger.info(f"Portfolio saved to {filepath}")
                
            elif args.mode == 'backtest':
                if not start_date:
                    # Default to 1 year ago
                    start_date = end_date - timedelta(days=365)
                
                backtest_result = await run_backtest(ctx, universe, start_date, end_date)
                backtest_result.print_summary()
                
                if config.output.save_metrics:
                    filepath = backtest_result.save(output_dir)
                    logger.info(f"Backtest results saved to {filepath}")
        
        logger.info("ACO Portfolio Optimizer completed successfully")
        return 0
        
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
        return 1


def main() -> int:
    """Main entry point."""
    return asyncio.run(main_async())


if __name__ == '__main__':
    sys.exit(main())
