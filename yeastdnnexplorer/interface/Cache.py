import logging
from typing import Any

from cachetools import TTLCache  # type: ignore


class Cache:
    """A caching class that uses cachetools for TTL caching with an LRU eviction
    policy."""

    def __init__(self, maxsize: int = 100, ttl: int = 300):
        self.ttl_cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.logger = logging.getLogger(__name__)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the cache."""
        return self.ttl_cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache."""
        self.ttl_cache[key] = value

    def list(self) -> list[str]:
        """List all keys in the cache."""
        return list(self.ttl_cache.keys())

    def delete(self, key: str) -> None:
        """Delete a key from the cache."""
        self.ttl_cache.pop(key, None)
