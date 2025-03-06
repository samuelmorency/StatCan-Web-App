"""Cache service for optimized data storage and retrieval."""

import os
import tempfile
import time
import logging
import hashlib
import functools
from diskcache import Cache
from config.settings import APP_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

class AzureCache:
    """Cache implementation optimized for Azure deployment environments."""

    def __init__(self):
        """Initialize the cache with configuration settings."""
        self._CACHE_DIR = os.path.join(tempfile.gettempdir(), APP_CONFIG["cache_dir"])
        self._DEFAULT_SIZE = APP_CONFIG["cache_size"]  # Default cache size from config
        self._FALLBACK_SIZE = APP_CONFIG["cache_size"] * 0.25  # 25% of default size as fallback
        self._TTL = APP_CONFIG["cache_ttl"]  # Default TTL from config
        self._cache = None
        self._memory_cache = {}
        self._last_access = {}
        self._MAX_MEMORY_ITEMS = 1000
        self.initialize_cache()

    def _prune_memory_cache(self):
        """Remove oldest items when memory cache exceeds maximum size."""
        if len(self._memory_cache) > self._MAX_MEMORY_ITEMS:
            sorted_items = sorted(self._last_access.items(), key=lambda x: x[1])
            to_remove = len(self._memory_cache) - self._MAX_MEMORY_ITEMS
            for key, _ in sorted_items[:to_remove]:
                self._memory_cache.pop(key, None)
                self._last_access.pop(key, None)

    def get_cache(self):
        """Return the underlying disk cache instance."""
        return self._cache

    def get_cache_value(self, key):
        """
        Retrieve a value from cache, using memory cache first for speed.

        Args:
            key: The cache key to retrieve

        Returns:
            The cached value or None if not found
        """
        current_time = time.time()
        
        if key in self._memory_cache:
            self._last_access[key] = current_time
            return self._memory_cache[key]

        if self._cache:
            try:
                value = self._cache.get(key)
                if value is not None:
                    self._memory_cache[key] = value
                    self._last_access[key] = current_time
                    self._prune_memory_cache()
                return value
            except Exception as e:
                logger.warning(f"Disk cache retrieval failed: {e}")
        return None

    def set_cache_value(self, key, value, ttl=None):
        """
        Store a value in both memory and disk cache.

        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds, uses default if None
        """
        ttl = ttl or self._TTL
        current_time = time.time()

        self._memory_cache[key] = value
        self._last_access[key] = current_time
        self._prune_memory_cache()

        if self._cache:
            try:
                self._cache.set(key, value, expire=ttl)
            except Exception as e:
                logger.warning(f"Disk cache set failed: {e}")

    def initialize_cache(self):
        """Set up the disk cache with error handling and fallback options."""
        try:
            os.makedirs(self._CACHE_DIR, exist_ok=True)
            self._cache = Cache(
                directory=self._CACHE_DIR,
                size_limit=self._DEFAULT_SIZE,
                eviction_policy='least-recently-used',
                cull_limit=10,
                statistics=True
            )
            logger.info(f"Cache initialized at {self._CACHE_DIR} with size {self._DEFAULT_SIZE/1e6}MB")
        except Exception as e:
            logger.warning(f"Primary cache initialization failed: {e}")
            try:
                self._cache = Cache(
                    directory=self._CACHE_DIR,
                    size_limit=self._FALLBACK_SIZE,
                    eviction_policy='least-recently-used'
                )
                logger.info(f"Fallback cache initialized with size {self._FALLBACK_SIZE/1e6}MB")
            except Exception as e:
                logger.error(f"Cache initialization completely failed: {e}")
                self._cache = None

    def clear_cache(self):
        """Clear both memory and disk cache."""
        self._memory_cache.clear()
        self._last_access.clear()
        if self._cache:
            try:
                self._cache.clear()
                logger.info("Cache cleared successfully")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")

    def monitor_usage(self):
        """Monitor cache size and usage."""
        if not self._cache:
            return
            
        try:
            total_size = sum(os.path.getsize(os.path.join(self._cache.directory, f))
                            for f in os.listdir(self._cache.directory)
                            if os.path.isfile(os.path.join(self._cache.directory, f)))
            
            usage_mb = total_size / 1e6
            logger.info(f"Current cache usage: {usage_mb:.2f}MB")
            
            if usage_mb > (self._cache.size_limit / 1e6) * 0.9:  # 90% threshold
                logger.warning("Cache usage approaching limit")
                self._cache.expire()
        except Exception as e:
            logger.error(f"Error monitoring cache: {e}")


def create_cache_decorator(azure_cache, ttl=None):
    """
    Create a decorator for caching function results with configurable TTL.
    
    Args:
        azure_cache: AzureCache instance to use for caching
        ttl: Time-to-live in seconds (uses AzureCache default if None)
        
    Returns:
        decorator: Function decorator for caching results
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Create consistent cache key using hash
                key_str = f"{func.__module__}.{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                key = hashlib.md5(key_str.encode()).hexdigest()

                # Try getting from cache
                result = azure_cache.get_cache_value(key)
                if result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result

                # Calculate and cache result
                result = func(*args, **kwargs)
                azure_cache.set_cache_value(key, result, ttl=ttl)
                logger.debug(f"Cache miss for {func.__name__}, stored new result")
                return result
            except Exception as e:
                logger.error(f"Cache decorator failed for {func.__name__}: {e}")
                return func(*args, **kwargs)  # Fallback to original function
        return wrapper
    return decorator


# Initialize global cache instance
azure_cache = AzureCache()
cache = azure_cache.get_cache()

# Create decorator with default app config TTL
azure_cache_decorator = functools.partial(create_cache_decorator, azure_cache)