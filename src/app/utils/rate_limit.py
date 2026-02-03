"""
Rate limiting utilities for API calls.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from aiolimiter import AsyncLimiter


class RateLimiter:
    """
    Rate limiter wrapper for API calls.
    Supports multiple rate limit tiers (per-second and per-minute).
    """
    
    def __init__(
        self,
        requests_per_minute: int = 5,
        requests_per_second: int | None = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_second: Optional per-second limit
        """
        self._per_minute = AsyncLimiter(requests_per_minute, 60.0)
        self._per_second = (
            AsyncLimiter(requests_per_second, 1.0)
            if requests_per_second
            else None
        )
    
    async def acquire(self) -> None:
        """Acquire a rate limit slot (blocks if needed)."""
        if self._per_second:
            await self._per_second.acquire()
        await self._per_minute.acquire()
    
    @asynccontextmanager
    async def limit(self) -> AsyncIterator[None]:
        """Context manager for rate-limited operations."""
        await self.acquire()
        yield


def create_rate_limiter(requests_per_minute: int) -> AsyncLimiter:
    """Create a simple rate limiter."""
    return AsyncLimiter(requests_per_minute, 60.0)
