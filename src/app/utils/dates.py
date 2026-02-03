"""
Date utilities for trading day calculations.
"""

from datetime import date, timedelta
from typing import Iterator


# Standard US market holidays (simplified - a real implementation would use a library)
# This is a simplified version for the MVP
def is_weekend(d: date) -> bool:
    """Check if date is a weekend."""
    return d.weekday() >= 5  # Saturday = 5, Sunday = 6


def get_trading_days(start: date, end: date) -> list[date]:
    """
    Get list of trading days between start and end (inclusive).
    
    Note: This is a simplified implementation that only excludes weekends.
    A production system would use a proper market calendar.
    
    Args:
        start: Start date
        end: End date
    
    Returns:
        List of trading days
    """
    days = []
    current = start
    while current <= end:
        if not is_weekend(current):
            days.append(current)
        current += timedelta(days=1)
    return days


def get_previous_trading_day(d: date) -> date:
    """Get the previous trading day before the given date."""
    current = d - timedelta(days=1)
    while is_weekend(current):
        current -= timedelta(days=1)
    return current


def get_next_trading_day(d: date) -> date:
    """Get the next trading day after the given date."""
    current = d + timedelta(days=1)
    while is_weekend(current):
        current += timedelta(days=1)
    return current


def get_trading_days_back(from_date: date, num_days: int) -> date:
    """
    Get the date that is num_days trading days before from_date.
    
    Args:
        from_date: Starting date
        num_days: Number of trading days to go back
    
    Returns:
        Date that is num_days trading days before from_date
    """
    count = 0
    current = from_date
    while count < num_days:
        current -= timedelta(days=1)
        if not is_weekend(current):
            count += 1
    return current


def iter_monthly_dates(start: date, end: date) -> Iterator[date]:
    """
    Iterate through month-end trading dates.
    
    Args:
        start: Start date
        end: End date
    
    Yields:
        Last trading day of each month
    """
    current = start
    while current <= end:
        # Move to end of month
        if current.month == 12:
            month_end = date(current.year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = date(current.year, current.month + 1, 1) - timedelta(days=1)
        
        # Ensure it's a trading day
        while is_weekend(month_end):
            month_end -= timedelta(days=1)
        
        if start <= month_end <= end:
            yield month_end
        
        # Move to next month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def iter_weekly_dates(start: date, end: date, weekday: int = 4) -> Iterator[date]:
    """
    Iterate through weekly dates (default: Friday).
    
    Args:
        start: Start date
        end: End date
        weekday: Day of week (0=Monday, 4=Friday)
    
    Yields:
        Weekly dates
    """
    current = start
    # Move to first target weekday
    while current.weekday() != weekday:
        current += timedelta(days=1)
    
    while current <= end:
        if not is_weekend(current):
            yield current
        current += timedelta(days=7)


def parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format."""
    return date.fromisoformat(date_str)


def format_date(d: date) -> str:
    """Format a date as YYYY-MM-DD."""
    return d.isoformat()
