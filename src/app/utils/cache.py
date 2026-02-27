"""
Caching utilities for storing and retrieving data.
"""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


class CacheManager:
    """
    File-based cache manager for API responses and computed data.
    
    Supports:
    - TTL-based expiration
    - Parquet format for DataFrames
    - JSON format for other data
    - Atomic writes
    """
    
    def __init__(
        self,
        cache_dir: str = "data/cache",
        default_ttl_days: int = 1,
        format: str = "parquet"
    ):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Base directory for cache files
            default_ttl_days: Default time-to-live in days
            format: Default format (parquet or json)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = timedelta(days=default_ttl_days)
        self.format = format
    
    def _get_cache_key(self, namespace: str, key: str) -> str:
        """Generate a cache key hash."""
        combined = f"{namespace}:{key}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, namespace: str, key: str, ext: str) -> Path:
        """Get the file path for a cache entry."""
        cache_key = self._get_cache_key(namespace, key)
        namespace_dir = self.cache_dir / namespace
        namespace_dir.mkdir(parents=True, exist_ok=True)
        return namespace_dir / f"{cache_key}.{ext}"
    
    def _get_meta_path(self, cache_path: Path) -> Path:
        """Get the metadata file path for a cache entry."""
        return cache_path.with_suffix('.meta.json')
    
    def _is_expired(self, meta_path: Path, ttl: timedelta | None = None) -> bool:
        """Check if a cache entry is expired."""
        if not meta_path.exists():
            return True
        
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            cached_time = datetime.fromisoformat(meta['timestamp'])
            effective_ttl = ttl if ttl is not None else self.default_ttl
            return datetime.now() - cached_time > effective_ttl
        except (json.JSONDecodeError, KeyError, ValueError):
            return True
    
    def get_dataframe(
        self,
        namespace: str,
        key: str,
        ttl: timedelta | None = None
    ) -> pd.DataFrame | None:
        """
        Retrieve a cached DataFrame.
        
        Args:
            namespace: Cache namespace (e.g., 'prices')
            key: Cache key (e.g., 'AAPL_2023-01-01_2024-01-01')
            ttl: Optional TTL override
        
        Returns:
            Cached DataFrame or None if not found/expired
        """
        cache_path = self._get_cache_path(namespace, key, 'parquet')
        meta_path = self._get_meta_path(cache_path)
        
        if not cache_path.exists() or self._is_expired(meta_path, ttl):
            return None
        
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            return None
    
    def set_dataframe(
        self,
        namespace: str,
        key: str,
        data: pd.DataFrame,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Store a DataFrame in cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            data: DataFrame to cache
            metadata: Optional additional metadata
        """
        cache_path = self._get_cache_path(namespace, key, 'parquet')
        meta_path = self._get_meta_path(cache_path)
        
        # Write DataFrame atomically
        temp_path = cache_path.with_suffix('.tmp')
        data.to_parquet(temp_path)
        temp_path.rename(cache_path)
        
        # Write metadata
        meta = {
            'timestamp': datetime.now().isoformat(),
            'key': key,
            'shape': list(data.shape),
            **(metadata or {})
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
    
    def get_json(
        self,
        namespace: str,
        key: str,
        ttl: timedelta | None = None
    ) -> dict | None:
        """Retrieve cached JSON data."""
        cache_path = self._get_cache_path(namespace, key, 'json')
        meta_path = self._get_meta_path(cache_path)
        
        if not cache_path.exists() or self._is_expired(meta_path, ttl):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    def set_json(
        self,
        namespace: str,
        key: str,
        data: dict,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Store JSON data in cache."""
        cache_path = self._get_cache_path(namespace, key, 'json')
        meta_path = self._get_meta_path(cache_path)
        
        # Write data atomically
        temp_path = cache_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        temp_path.rename(cache_path)
        
        # Write metadata
        meta = {
            'timestamp': datetime.now().isoformat(),
            'key': key,
            **(metadata or {})
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
    
    def clear(self, namespace: str | None = None) -> int:
        """
        Clear cache entries.
        
        Args:
            namespace: If provided, only clear entries in this namespace
        
        Returns:
            Number of entries cleared
        """
        count = 0
        if namespace:
            namespace_dir = self.cache_dir / namespace
            if namespace_dir.exists():
                for f in namespace_dir.iterdir():
                    if f.is_file():
                        f.unlink()
                        count += 1
        else:
            for namespace_dir in self.cache_dir.iterdir():
                if namespace_dir.is_dir():
                    for f in namespace_dir.iterdir():
                        if f.is_file():
                            f.unlink()
                            count += 1
        return count
