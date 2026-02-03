"""
Metrics calculation for backtesting.
"""

import numpy as np
import pandas as pd


def calculate_cumulative_return(returns: pd.Series) -> float:
    """
    Calculate cumulative return from a series of returns.
    
    Args:
        returns: Series of period returns
    
    Returns:
        Total cumulative return as a decimal (e.g., 0.10 for 10%)
    """
    if returns.empty:
        return 0.0
    
    cumulative = (1 + returns).prod() - 1
    return float(cumulative)


def calculate_annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return from a series of returns.
    
    Args:
        returns: Series of period returns
        periods_per_year: Number of periods per year (252 for daily)
    
    Returns:
        Annualized return as a decimal
    """
    if returns.empty:
        return 0.0
    
    cumulative = calculate_cumulative_return(returns)
    n_periods = len(returns)
    
    if n_periods == 0:
        return 0.0
    
    # Annualize: (1 + total)^(periods_per_year / n) - 1
    annualized = (1 + cumulative) ** (periods_per_year / n_periods) - 1
    return float(annualized)


def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility from a series of returns.
    
    Args:
        returns: Series of period returns
        periods_per_year: Number of periods per year
    
    Returns:
        Annualized volatility (standard deviation)
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    
    daily_vol = returns.std()
    annualized = daily_vol * np.sqrt(periods_per_year)
    return float(annualized)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Sharpe ratio
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    
    annualized_return = calculate_annualized_return(returns, periods_per_year)
    annualized_vol = calculate_volatility(returns, periods_per_year)
    
    if annualized_vol == 0:
        return 0.0
    
    sharpe = (annualized_return - risk_free_rate) / annualized_vol
    return float(sharpe)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from a series of returns.
    
    Args:
        returns: Series of period returns
    
    Returns:
        Maximum drawdown as a positive decimal (e.g., 0.20 for 20% drawdown)
    """
    if returns.empty:
        return 0.0
    
    # Calculate cumulative wealth
    cumulative = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cumulative.cummax()
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Return max drawdown (as positive number)
    return float(-drawdown.min())


def calculate_turnover(
    weights_prev: dict[str, float],
    weights_curr: dict[str, float]
) -> float:
    """
    Calculate portfolio turnover between two allocations.
    
    Turnover is the sum of absolute weight changes / 2.
    A complete portfolio replacement would give turnover = 1.
    
    Args:
        weights_prev: Previous period weights
        weights_curr: Current period weights
    
    Returns:
        Turnover as a decimal (0-1 typically)
    """
    all_tickers = set(weights_prev.keys()) | set(weights_curr.keys())
    
    total_change = sum(
        abs(weights_curr.get(t, 0) - weights_prev.get(t, 0))
        for t in all_tickers
    )
    
    # Divide by 2 because a complete swap counts both buy and sell
    return total_change / 2


def calculate_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate information ratio (active return / tracking error).
    
    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns
        periods_per_year: Number of periods per year
    
    Returns:
        Information ratio
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        return 0.0
    
    # Align series
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if len(aligned) < 2:
        return 0.0
    
    active_returns = aligned['portfolio'] - aligned['benchmark']
    
    annualized_active = calculate_annualized_return(active_returns, periods_per_year)
    tracking_error = calculate_volatility(active_returns, periods_per_year)
    
    if tracking_error == 0:
        return 0.0
    
    return annualized_active / tracking_error


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (return / downside deviation).
    
    Args:
        returns: Series of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Sortino ratio
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    
    # Calculate downside deviation (only negative returns)
    negative_returns = returns[returns < 0]
    
    if len(negative_returns) == 0:
        return float('inf')  # No downside
    
    downside_std = np.sqrt(np.mean(negative_returns ** 2))
    annualized_downside = downside_std * np.sqrt(periods_per_year)
    
    if annualized_downside == 0:
        return 0.0
    
    annualized_return = calculate_annualized_return(returns, periods_per_year)
    
    return (annualized_return - risk_free_rate) / annualized_downside


def calculate_all_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0
) -> dict:
    """
    Calculate all standard metrics for a return series.
    
    Args:
        returns: Series of portfolio returns
        benchmark_returns: Optional benchmark returns for comparison
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Dict of metric name -> value
    """
    metrics = {
        'cumulative_return': calculate_cumulative_return(returns),
        'annualized_return': calculate_annualized_return(returns),
        'annualized_volatility': calculate_volatility(returns),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate),
        'max_drawdown': calculate_max_drawdown(returns),
        'num_periods': len(returns),
    }
    
    if benchmark_returns is not None and not benchmark_returns.empty:
        metrics['information_ratio'] = calculate_information_ratio(
            returns, benchmark_returns
        )
        metrics['benchmark_cumulative_return'] = calculate_cumulative_return(
            benchmark_returns
        )
        metrics['excess_return'] = (
            metrics['cumulative_return'] - metrics['benchmark_cumulative_return']
        )
    
    return metrics
