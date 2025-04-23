# cache_utils.py – Caching system implementation
import os, time, tempfile, logging, atexit, platform
from diskcache import Cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureCache:
    """In-memory + disk cache with LRU eviction."""
    def __init__(self):
        # Determine cache directory (use Azure local storage if available)
        self._CACHE_DIR = self._get_cache_path()
        self._DEFAULT_SIZE = 2048e6  # 2 GB
        self._FALLBACK_SIZE = 512e6  # 512 MB
        self._TTL = 3600  # default TTL 1h
        self._cache = None
        self._memory_cache = {}
        self._last_access = {}
        self._is_azure = self._detect_azure_environment()
        self._MAX_MEMORY_ITEMS = 2000 if self._is_azure else 1000
        self._init_disk_cache()
    def _detect_azure_environment(self):
        indicators = ['WEBSITE_SITE_NAME','WEBSITE_INSTANCE_ID','WEBSITE_RESOURCE_GROUP']
        return any(var in os.environ for var in indicators)
    def _get_cache_path(self):
        """Determines the appropriate path for disk caching."""
        # Check if running in Azure App Service (Windows or Linux)
        if 'WEBSITE_INSTANCE_ID' in os.environ:
            # Use persistent storage under D:\home (Windows) or /home (Linux)
            # Note: App Service maps D:\home to /home on Linux containers
            home_dir = os.environ.get('HOME', 'D:\\home') # Default to D:\home for Windows
            cache_dir = os.path.join(home_dir, 'site', 'wwwroot', 'dash_cache')
            logger.info(f"Azure App Service detected. Using persistent cache path: {cache_dir}")
        elif platform.system() == 'Windows':
            # Local Windows development: Use temp directory
            cache_dir = os.path.join(tempfile.gettempdir(), 'dash_cache')
            logger.info(f"Local Windows detected. Using temp cache path: {cache_dir}")
        else:
            # Other environments (e.g., local Linux/Mac): Use temp directory
            cache_dir = os.path.join(tempfile.gettempdir(), 'dash_cache')
            logger.info(f"Other environment detected. Using temp cache path: {cache_dir}")

        # Ensure the directory exists
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create cache directory {cache_dir}: {e}")
            # Fallback to a basic temp dir if creation fails
            cache_dir = os.path.join(tempfile.gettempdir(), 'dash_cache_fallback')
            try:
                os.makedirs(cache_dir, exist_ok=True)
                logger.warning(f"Using fallback cache directory: {cache_dir}")
            except OSError as e_fallback:
                logger.error(f"Failed to create fallback cache directory {cache_dir}: {e_fallback}")
                return None # Indicate failure if even fallback fails
        return cache_dir
    def _init_disk_cache(self):
        try:
            os.makedirs(self._CACHE_DIR, exist_ok=True)
            # Limit size to 50% of free space or default
            free_space = (os.statvfs(self._CACHE_DIR).f_frsize * os.statvfs(self._CACHE_DIR).f_bavail 
                          if hasattr(os, 'statvfs') else 0)
            size_limit = min(self._DEFAULT_SIZE, free_space * 0.5) if free_space else self._DEFAULT_SIZE
            self._cache = Cache(directory=self._CACHE_DIR, size_limit=size_limit,
                                 eviction_policy='least-recently-used', cull_limit=10, statistics=True)
            logger.info(f"Cache initialized at {self._CACHE_DIR} (size ~{size_limit/1e6:.0f} MB)")
        except Exception as e:
            logger.warning(f"Primary cache init failed: {e}")
            try:
                self._cache = Cache(directory=self._CACHE_DIR, size_limit=self._FALLBACK_SIZE,
                                     eviction_policy='least-recently-used')
                logger.info(f"Initialized fallback cache (size {self._FALLBACK_SIZE/1e6:.0f} MB)")
            except Exception as e2:
                logger.error(f"Cache init failed completely: {e2}")
                self._cache = None
    def _prune_memory_cache(self):
        # Enforce LRU policy on memory cache
        if len(self._memory_cache) > self._MAX_MEMORY_ITEMS:
            # Remove least recently accessed entries
            oldest_keys = sorted(self._last_access.items(), key=lambda kv: kv[1])[:len(self._memory_cache) - self._MAX_MEMORY_ITEMS]
            for key, _ in oldest_keys:
                self._memory_cache.pop(key, None)
                self._last_access.pop(key, None)
            logger.info(f"CACHE EVICTION: Pruned {len(oldest_keys)} items from memory cache")
    def get_cache_value(self, key):
        now = time.time()
        if key in self._memory_cache:  # check in-memory first
            self._last_access[key] = now
            logger.info(f"CACHE HIT (Memory): {key}")
            return self._memory_cache[key]
        if self._cache:  # check disk cache
            try:
                value = self._cache.get(key)
            except Exception as e:
                logger.warning(f"Disk cache get failed for {key}: {e}")
                value = None
            if value is not None:
                self._memory_cache[key] = value  # promote to memory
                self._last_access[key] = now
                self._prune_memory_cache()
                logger.info(f"CACHE HIT (Disk): {key}")
                return value
        logger.info(f"CACHE MISS: {key}")
        return None
    def set_cache_value(self, key, value, ttl=None):
        ttl = ttl or self._TTL
        self._memory_cache[key] = value
        self._last_access[key] = time.time()
        self._prune_memory_cache()
        if self._cache:
            try:
                self._cache.set(key, value, expire=ttl)
            except Exception as e:
                logger.warning(f"Disk cache set failed for {key}: {e}")
        logger.info(f"CACHE INSERT: Stored key {key} (TTL={ttl}s)")
    def clear_cache(self):
        self._memory_cache.clear()
        self._last_access.clear()
        if self._cache:
            try:
                self._cache.clear()
                logger.info("Cache cleared successfully")
            except Exception as e:
                logger.error(f"Error clearing disk cache: {e}")
    def cleanup_expired(self):
        if not self._cache: return
        expired_count = 0
        try:
            # Remove expired keys from disk cache
            for key in list(self._cache):
                if key in getattr(self._cache, '_expire', {}):
                    if time.time() > self._cache._expire[key]:
                        self._cache.delete(key)
                        expired_count += 1
        except Exception as e:
            logger.warning(f"Error during cache cleanup: {e}")
        if expired_count:
            logger.info(f"Cleaned up {expired_count} expired disk cache items")
    def get_cache(self):
        return self._cache
    def get_stats(self):
        stats = {
            "memory_items": len(self._memory_cache),
            "memory_limit": getattr(self, '_MAX_MEMORY_ITEMS', None),
            "disk_items": None,
            "disk_size_mb": None,
            "disk_hits": None,
            "disk_misses": None,
            "hit_rate": None
        }
        if self._cache and hasattr(self._cache, 'stats'):
            disk_stats = self._cache.stats()
            stats.update({
                "disk_items": disk_stats.get('items'),
                "disk_size_mb": disk_stats.get('size', 0)/1e6,
                "disk_hits": disk_stats.get('hits'),
                "disk_misses": disk_stats.get('misses'),
                "hit_rate": disk_stats.get('hit_rate')
            })
        return stats

# Global cache instance
azure_cache = None
cache = None

def initialize_cache():
    """Initialize the global cache (called in app.py at startup)"""
    global azure_cache, cache
    if azure_cache is None:
        logger.info("Initializing cache system...")
        azure_cache = AzureCache()
        cache = azure_cache.get_cache()
        atexit.register(azure_cache.clear_cache)  # ensure cache clears on shutdown
        # Warm up memory cache with recent disk entries (optional, can be extended)
        try:
            if cache:
                keys = list(cache)[:azure_cache._MAX_MEMORY_ITEMS//2]
                for k in keys:
                    val = cache.get(k)
                    if val is not None:
                        azure_cache._memory_cache[k] = val
                        azure_cache._last_access[k] = time.time()
                logger.info(f"Warmed memory cache with {len(azure_cache._memory_cache)} items")
        except Exception as e:
            logger.warning(f"Memory cache warm-up failed: {e}")
        # Start maintenance thread for periodic cleanup
        import threading
        def maintenance():
            while True:
                time.sleep(3600)
                logger.info("Running scheduled cache maintenance")
                azure_cache.cleanup_expired()
                stats = azure_cache.get_stats()
                logger.info(f"Memory cache: {stats['memory_items']}/{stats['memory_limit']} items")
                if stats['disk_items'] is not None:
                    logger.info(f"Disk cache: {stats['disk_items']} items, {stats['disk_size_mb']:.1f} MB; "
                                f"Hits: {stats['disk_hits']}, Misses: {stats['disk_misses']}, "
                                f"Hit rate: {stats['hit_rate']:.1f}%")
        threading.Thread(target=maintenance, daemon=True).start()
        logger.info("Cache system initialized.")
    return azure_cache, cache

def azure_cache_decorator(ttl=3600):
    """Decorator to cache a function's return value in AzureCache."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if azure_cache is None:
                initialize_cache()
            # Create a unique cache key for this call
            key_str = f"{func.__module__}.{func.__name__}:{args}:{sorted(kwargs.items())}"
            import hashlib
            cache_key = hashlib.md5(key_str.encode()).hexdigest()
            try:
                result = azure_cache.get_cache_value(cache_key)
                if result is not None:
                    logger.info(f"Cache hit for {func.__name__}")
                    return result
                result = func(*args, **kwargs)
                azure_cache.set_cache_value(cache_key, result, ttl=ttl)
                logger.info(f"Cache miss for {func.__name__} – computed and cached result")
                return result
            except Exception as e:
                logger.error(f"Cache wrapper error for {func.__name__}: {e}")
                return func(*args, **kwargs)
        return wrapper
    return decorator

def monitor_cache_usage():
    """Log current cache stats (used in update_visualizations for debugging)."""
    if azure_cache is None: 
        return
    stats = azure_cache.get_stats()
    logger.info(f"Memory cache: {stats['memory_items']}/{stats['memory_limit']} items")
    if stats['disk_items'] is not None:
        logger.info(f"Disk cache: {stats['disk_items']} items, {stats['disk_size_mb']:.1f} MB, "
                    f"Hit rate: {stats['hit_rate']:.1f}% (Hits: {stats['disk_hits']}, Misses: {stats['disk_misses']})")
