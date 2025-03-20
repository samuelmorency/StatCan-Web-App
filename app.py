import tempfile
import os
from dash import Dash, html, dcc, callback, Output, Input, State, callback_context, Patch, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import geopandas as gpd
import plotly.express as px
import functools
import dash_leaflet as dl
import colorlover as cl
from dash_extensions.javascript import assign
import dash
from dash.exceptions import PreventUpdate
import time
import numpy as np
from functools import lru_cache
from diskcache import Cache
import orjson
import logging
import atexit
import brand_colours as bc
from app_layout import create_layout
import io
import csv
import hashlib
from dash_ag_grid import AgGrid
from time import perf_counter
from functools import wraps
import threading
from collections import defaultdict
import plotly.graph_objects as go
from pathlib import Path
from dash.dependencies import MATCH, ALL
import json
import pickle
import sys

NEW_DATA = True
NEW_SF = True
SIMPLIFIED_SF = True
COLOUR_SCALE = bc.BRIGHT_RED_SCALE

_cache_initialized = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureCache:
    """
    A sophisticated dual-layer caching system optimized for Azure App Service environments.
    
    This class implements a two-tier caching mechanism with an in-memory cache for
    fast access and a disk-based cache for persistence. The cache is optimized to use
    Azure App Service's local storage paths for better performance and persistence.
    
    Attributes:
        _CACHE_DIR (str): Directory path for disk-based cache storage, optimized for Azure.
        _DEFAULT_SIZE (float): Default size limit for disk cache (2GB).
        _FALLBACK_SIZE (float): Fallback size limit if primary fails (512MB).
        _TTL (int): Default time-to-live for cache entries in seconds (3600).
        _cache (diskcache.Cache): The disk-based cache instance.
        _memory_cache (dict): In-memory cache dictionary.
        _last_access (dict): Timestamp tracking for LRU eviction.
        _MAX_MEMORY_ITEMS (int): Maximum number of items in memory cache (2000 for Azure).
        _is_azure (bool): Flag indicating if running on Azure App Service.
    """
    def __init__(self):
        # Use optimized cache path based on environment
        self._CACHE_DIR = self._get_optimized_cache_path()
        self._DEFAULT_SIZE = 2048e6  # 2GB - D:\local has more space
        self._FALLBACK_SIZE = 512e6  # 512MB fallback
        self._TTL = 3600
        self._cache = None
        self._memory_cache = {}
        self._last_access = {}
        
        # Configure for Azure App Service if detected
        self._is_azure = self._detect_azure_environment()
        if self._is_azure:
            logger.info("Running on Azure App Service - optimizing cache configuration")
            self._MAX_MEMORY_ITEMS = 2000  # Higher limit on App Service
        else:
            logger.info("Running in local development environment")
            self._MAX_MEMORY_ITEMS = 1000
            
        self._initialize_cache()
        
    def _detect_azure_environment(self):
        """Detect if the application is running on Azure App Service"""
        # Check for environment variables that indicate Azure App Service
        azure_indicators = [
            'WEBSITE_SITE_NAME',
            'WEBSITE_INSTANCE_ID',
            'WEBSITE_RESOURCE_GROUP'
        ]
        
        return any(env in os.environ for env in azure_indicators)
        
    def _get_optimized_cache_path(self):
        """Determine the optimal cache path based on environment"""
        # First try Azure App Service's D:\local directory
        azure_local_path = r'D:\local\Temp\dash_cache'
        
        if os.path.exists(r'D:\local'):
            logger.info("Using Azure App Service local storage for cache")
            return azure_local_path
            
        # Fallback to home directory if available (also persistent on App Service)
        home_path = os.environ.get('HOME')
        if home_path:
            home_cache = os.path.join(home_path, 'dash_cache')
            logger.info(f"Using home directory for cache: {home_cache}")
            return home_cache
            
        # Last resort - use system temp directory
        temp_path = os.path.join(tempfile.gettempdir(), 'dash_cache')
        logger.info(f"Using system temp directory for cache: {temp_path}")
        return temp_path

    def _prune_memory_cache(self):
        """Prune memory cache using LRU policy if it exceeds size limit"""
        if len(self._memory_cache) > self._MAX_MEMORY_ITEMS:
            sorted_items = sorted(self._last_access.items(), key=lambda x: x[1])
            to_remove = len(self._memory_cache) - self._MAX_MEMORY_ITEMS
            for key, _ in sorted_items[:to_remove]:
                self._memory_cache.pop(key, None)
                self._last_access.pop(key, None)
            logger.debug(f"Pruned {to_remove} items from memory cache")

    def get_cache(self):
        """Returns the disk cache instance for backwards compatibility"""
        return self._cache

    def get_cache_value(self, key):
        """
        Retrieves a value from the cache system with optimized access pattern.
        
        First checks memory cache for fastest access, then falls back to disk cache.
        Updates memory cache if value is found in disk cache for faster future access.
        
        Args:
            key (str): Cache key to retrieve.
            
        Returns:
            Any: Cached value if found, None otherwise.
        """
        current_time = time.time()
        
        # Try memory cache first (fastest)
        if key in self._memory_cache:
            self._last_access[key] = current_time
            return self._memory_cache[key]

        # Try disk cache if memory cache misses
        if self._cache:
            try:
                value = self._cache.get(key)
                if value is not None:
                    # Update memory cache with disk cache hit
                    self._memory_cache[key] = value
                    self._last_access[key] = current_time
                    self._prune_memory_cache()
                    return value
            except Exception as e:
                logger.warning(f"Disk cache retrieval failed: {e}")
                
        return None

    def set_cache_value(self, key, value, ttl=None):
        """
        Stores a value in both memory and disk cache layers.
        
        Updates both memory cache for fast access and disk cache for persistence.
        Applies automatic pruning to memory cache if it exceeds size limits.
        
        Args:
            key (str): Cache key to store.
            value (Any): Value to cache.
            ttl (int, optional): Time-to-live in seconds. Defaults to self._TTL.
        """
        ttl = ttl or self._TTL
        current_time = time.time()

        # Update memory cache
        self._memory_cache[key] = value
        self._last_access[key] = current_time
        self._prune_memory_cache()

        # Update disk cache
        if self._cache:
            try:
                self._cache.set(key, value, expire=ttl)
            except Exception as e:
                logger.warning(f"Disk cache set failed: {e}")

    def delete_cache_value(self, key):
        """
        Deletes a specific value from all cache layers.
        
        Args:
            key (str): Cache key to delete.
        """
        # Remove from memory cache
        if key in self._memory_cache:
            self._memory_cache.pop(key, None)
            self._last_access.pop(key, None)
                
        # Remove from disk cache
        if self._cache:
            try:
                self._cache.delete(key)
            except Exception as e:
                logger.warning(f"Disk cache delete failed: {e}")

    def clear_cache(self):
        """Clears all cache layers completely"""
        # Clear memory cache
        self._memory_cache.clear()
        self._last_access.clear()
        
        # Clear disk cache
        if self._cache:
            try:
                self._cache.clear()
                logger.info("Cache cleared successfully")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")

    def _initialize_cache(self):
        """
        Initialize the disk-based cache with improved error handling and diagnostics.
        
        Determines available disk space and adjusts cache size accordingly to prevent
        disk space issues. Falls back to smaller size if initialization fails.
        """
        try:
            # Ensure directory exists
            os.makedirs(self._CACHE_DIR, exist_ok=True)
            
            # Check available disk space
            if os.name == 'nt':  # Windows
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(self._CACHE_DIR), None, None, 
                    ctypes.pointer(free_bytes)
                )
                free_space = free_bytes.value
            else:
                import shutil
                free_space = shutil.disk_usage(self._CACHE_DIR).free
                
            # Adjust cache size based on available space
            adjusted_size = min(self._DEFAULT_SIZE, free_space * 0.5)  # Use max 50% of free space
            
            logger.info(f"Initializing cache at {self._CACHE_DIR}")
            logger.info(f"Available disk space: {free_space/1e9:.2f}GB")
            logger.info(f"Setting cache size to: {adjusted_size/1e6:.2f}MB")
            
            self._cache = Cache(
                directory=self._CACHE_DIR,
                size_limit=adjusted_size,
                eviction_policy='least-recently-used',
                cull_limit=10,
                statistics=True
            )
            logger.info(f"Cache initialized successfully")
            
        except Exception as e:
            logger.warning(f"Primary cache initialization failed: {e}")
            try:
                # Try with smaller size
                self._cache = Cache(
                    directory=self._CACHE_DIR,
                    size_limit=self._FALLBACK_SIZE,
                    eviction_policy='least-recently-used'
                )
                logger.info(f"Fallback cache initialized with size {self._FALLBACK_SIZE/1e6}MB")
            except Exception as e:
                logger.error(f"Cache initialization completely failed: {e}")
                self._cache = None

    def warm_memory_cache(self):
        """
        Warm memory cache with frequently used items from disk cache.
        
        Pre-loads items from disk cache into memory cache for faster initial access.
        Prioritizes most recently accessed items (up to half the memory cache capacity).
        """
        if not self._cache:
            return
            
        try:
            logger.info("Warming memory cache from disk cache...")
            # Get most recently used keys from disk
            cache_items = list(self._cache.iterkeys())
            
            # Warm up to MAX_MEMORY_ITEMS/2
            warm_limit = min(len(cache_items), self._MAX_MEMORY_ITEMS // 2)
            
            warmed = 0
            start_time = time.time()
            
            for key in cache_items[:warm_limit]:
                value = self._cache.get(key)
                if value is not None:
                    self._memory_cache[key] = value
                    self._last_access[key] = time.time()
                    warmed += 1
                    
            duration = time.time() - start_time
            logger.info(f"Warmed memory cache with {warmed} items in {duration:.2f}s")
        except Exception as e:
            logger.warning(f"Error warming memory cache: {e}")

    def cleanup_expired_items(self):
        """
        Remove expired items from disk cache to free up space.
        
        Scans through disk cache and removes items that have exceeded their TTL.
        This helps keep the cache size under control and improves performance.
        """
        if not self._cache:
            return
            
        try:
            # Get current time
            current_time = time.time()
            expired_count = 0
            
            # Scan for expired items
            for key in list(self._cache):
                # Check if item has expired based on its expire time
                # diskcache doesn't expose this directly, so we need to use internal API
                if hasattr(self._cache, '_expire') and key in self._cache._expire:
                    expire_time = self._cache._expire[key]
                    if expire_time < current_time:
                        self._cache.delete(key)
                        expired_count += 1
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired items from disk cache")
                
        except Exception as e:
            logger.warning(f"Error cleaning up expired cache items: {e}")

    def get_cache_stats(self):
        """
        Get comprehensive statistics about the cache system.
        
        Returns detailed statistics about both memory and disk cache usage
        for monitoring and debugging purposes.
        
        Returns:
            dict: A dictionary containing cache statistics.
        """
        stats = {
            "memory_cache": {
                "items": len(self._memory_cache),
                "limit": self._MAX_MEMORY_ITEMS,
                "usage_percent": (len(self._memory_cache) / self._MAX_MEMORY_ITEMS) * 100 if self._MAX_MEMORY_ITEMS > 0 else 0
            },
            "disk_cache": {
                "available": self._cache is not None,
                "path": self._CACHE_DIR,
                "size_limit_mb": self._DEFAULT_SIZE / 1e6 if self._cache else 0
            },
            "environment": "azure" if self._is_azure else "local"
        }
        
        # Add disk cache stats if available
        if self._cache and hasattr(self._cache, 'stats'):
            try:
                disk_stats = self._cache.stats()
                stats["disk_cache"].update({
                    "hits": disk_stats.get("hits", 0),
                    "misses": disk_stats.get("misses", 0),
                    "hit_rate": disk_stats.get("hit rate", 0) * 100,
                    "size_mb": disk_stats.get("size", 0) / 1e6,
                    "items": disk_stats.get("count", 0)
                })
            except Exception as e:
                logger.warning(f"Error getting disk cache stats: {e}")
                
        return stats

    def get_memory_usage(self):
        """
        Calculate estimated memory usage of the memory cache.
        
        Returns:
            dict: Memory usage statistics.
        """
        try:
            total_size = 0
            item_sizes = {}
            
            # Sample up to 100 items to estimate average size
            sample_keys = list(self._memory_cache.keys())[:100]
            for key in sample_keys:
                item_size = sys.getsizeof(pickle.dumps(self._memory_cache[key]))
                total_size += item_size
                item_sizes[key] = item_size
            
            avg_size = total_size / len(sample_keys) if sample_keys else 0
            estimated_total = avg_size * len(self._memory_cache)
            
            return {
                "sample_count": len(sample_keys),
                "avg_item_size_kb": avg_size / 1024,
                "estimated_total_mb": estimated_total / (1024 * 1024),
                "largest_items": sorted(item_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        except Exception as e:
            logger.warning(f"Error calculating memory usage: {e}")
            return {"error": str(e)}

def setup_cache_maintenance(azure_cache, interval=3600):
    """
    Setup background thread for periodic cache maintenance.
    
    Args:
        azure_cache: AzureCache instance
        interval: Maintenance interval in seconds (default: 3600)
    """
    def maintenance_worker():
        while True:
            try:
                time.sleep(interval)  # Run every hour by default
                logger.info("Running scheduled cache maintenance")
                azure_cache.cleanup_expired_items()
                
                # Log cache stats
                stats = azure_cache.get_cache_stats()
                mem_stats = stats["memory_cache"]
                logger.info(f"Memory cache: {mem_stats['items']}/{mem_stats['limit']} items ({mem_stats['usage_percent']:.1f}%)")
                
                disk_stats = stats["disk_cache"]
                if disk_stats["available"] and "hit_rate" in disk_stats:
                    logger.info(f"Disk cache: {disk_stats['items']} items, {disk_stats['size_mb']:.1f}MB used")
                    logger.info(f"Hit rate: {disk_stats['hit_rate']:.1f}%, Hits: {disk_stats['hits']}, Misses: {disk_stats['misses']}")
            except Exception as e:
                logger.error(f"Error in cache maintenance: {e}")
    
    # Start maintenance thread
    maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
    maintenance_thread.start()
    logger.info("Cache maintenance scheduler started")

def test_cache_optimization(azure_cache):
    """
    Test function to verify cache optimization is working correctly.
    
    Args:
        azure_cache: AzureCache instance
        
    Returns:
        dict: Test results
    """
    try:
        # Test key with reasonably large data
        test_key = f"optimization_test_{time.time()}"
        test_data = {"large_array": [i for i in range(10000)]}
        
        # Test set operation
        set_start = time.time()
        azure_cache.set_cache_value(test_key, test_data)
        set_time = time.time() - set_start
        
        # Test get operation from memory
        mem_get_start = time.time()
        mem_result = azure_cache.get_cache_value(test_key)
        mem_get_time = time.time() - mem_get_start
        
        # Clear memory cache to test disk cache
        azure_cache._memory_cache.pop(test_key, None)
        azure_cache._last_access.pop(test_key, None)
        
        # Test get operation from disk
        disk_get_start = time.time()
        disk_result = azure_cache.get_cache_value(test_key)
        disk_get_time = time.time() - disk_get_start
        
        # Log results
        logger.info("Cache optimization test results:")
        logger.info(f"- Set time: {set_time:.6f}s")
        logger.info(f"- Memory get time: {mem_get_time:.6f}s")
        logger.info(f"- Disk get time: {disk_get_time:.6f}s")
        logger.info(f"- Memory:Disk speed ratio: {disk_get_time/mem_get_time:.1f}x")
        
        return {
            "success": True,
            "set_time": set_time,
            "memory_get_time": mem_get_time,
            "disk_get_time": disk_get_time,
            "speedup": disk_get_time/mem_get_time
        }
    except Exception as e:
        logger.error(f"Cache optimization test failed: {e}")
        return {"success": False, "error": str(e)}

# These variables will be initialized at runtime
azure_cache = None
cache = None

def initialize_cache():
    """Initialize the cache system and related resources"""
    global azure_cache, cache, _cache_initialized
    
    # Only create cache if it hasn't been initialized yet
    if not _cache_initialized:
        logger.info("Initializing cache system")
        
        # Initialize Azure-specific cache
        azure_cache = AzureCache()
        cache = azure_cache.get_cache()
        
        # Register cleanup
        atexit.register(azure_cache.clear_cache)
        
        # Warm cache on startup
        azure_cache.warm_memory_cache()
        
        # Set up maintenance
        setup_cache_maintenance(azure_cache)
        
        _cache_initialized = True
        logger.info("Cache system initialized")
    else:
        logger.debug("Cache already initialized, skipping")
    
    return azure_cache, cache


def azure_cache_decorator(ttl=3600):
    """Decorator for caching function results with configurable time-to-live."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Ensure cache is initialized
            global azure_cache, cache
            if azure_cache is None:
                azure_cache, cache = initialize_cache()
                
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

# Initialize the app
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        # Add Google Fonts link for Open Sans SemiBold
        'https://fonts.googleapis.com/css2?family=Open+Sans:wght@600&display=swap',
        #dbc.icons.BOOTSTRAP
    ]
)

# Set global font styles for the app 
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                font-family: 'Open Sans', sans-serif;
                font-weight: 600;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
server = app.server

class MapState:
    """
    Maintains and manages the state of the interactive map visualization.
    
    This class tracks the current state of the map including viewport bounds, zoom level,
    selections, and timing information. It prevents excessive or unintended viewport
    updates by implementing timing-based locks and controlled viewport adjustments.
    
    Attributes:
        _bounds (list): Current map bounds as [[min_lat, min_lon], [max_lat, max_lon]].
        _zoom (int): Current zoom level of the map.
        _selected_feature (str): DGUID of the currently selected geographic feature.
        _hover_feature (str): DGUID of the currently hovered geographic feature.
        _color_scale (dict): Current color scale mapping values to colors.
        _last_update_time (float): Timestamp of the last viewport update.
        _viewport_locked (bool): Flag indicating if viewport updates are temporarily locked.
    
    Methods:
        current_bounds: Property returning the current map bounds.
        current_zoom: Property returning the current zoom level.
        is_viewport_locked: Property returning the viewport lock status.
        should_update_viewport(new_bounds, force=False): Determines if viewport should be updated.
        update_state(bounds=None, zoom=None, selected=None, hover=None, color_scale=None): Updates map state.
    
    The viewport locking mechanism works as follows:
    1. After a viewport update, the viewport is temporarily locked
    2. Updates are blocked while locked unless forced
    3. The lock automatically releases after 2 seconds
    4. Forced updates bypass the lock entirely
    
    This prevents rapid successive viewport changes that can disorient users.
    """
    def __init__(self):
        """
        Initializes a new MapState instance with all state variables set to None or
        their default values. The initial state does not have any current bounds,
        zoom level, selected feature, or hover feature. The viewport lock is initially
        false and the last update time is zero.
        """
        self._bounds = None
        self._zoom = None
        self._selected_feature = None
        self._hover_feature = None
        self._color_scale = None
        self._last_update_time = 0
        self._viewport_locked = False
        
    @property
    def current_bounds(self):
        return self._bounds
        
    @property
    def current_zoom(self):
        return self._zoom
        
    @property
    def is_viewport_locked(self):
        return self._viewport_locked
    
    def should_update_viewport(self, new_bounds, force=False):
        """
        Determines whether the map viewport should be updated based on the current
        state, timing since last update, and whether a forced update is requested.
        If the viewport is locked and a forced update is not requested, the update
        will be prevented. If sufficient time has elapsed since the last update, the
        viewport lock is reset to allow changes.

        Args:
            new_bounds (list): A bounding box in the format [[min_lat, min_lon],
                               [max_lat, max_lon]] representing the new candidate
                               viewport bounds.
            force (bool): If True, overrides locking and forces an update regardless
                          of the elapsed time. If False, normal timing and lock
                          considerations apply.

        Returns:
            bool: True if the viewport should be updated, False otherwise.
        """
        if not self._bounds or force:
            return True
            
        # Reset viewport lock periodically
        current_time = time.time()
        if current_time - self._last_update_time > 2.0:  # Reset lock after 2 seconds
            self._viewport_locked = False
            
        if self._viewport_locked and not force:
            return False
            
        return True
        
    def update_state(self, bounds=None, zoom=None, selected=None, hover=None, color_scale=None):
        """
        Updates the internal state of the map. Any parameter that is not None will
        update the corresponding attribute. Updating the bounds also updates the
        last update time to the current timestamp.

        Args:
            bounds (list or None): New viewport bounds as [[min_lat, min_lon],
                                  [max_lat, max_lon]] or None to leave unchanged.
            zoom (int or None): New zoom level or None to leave unchanged.
            selected (str or None): New selected feature identifier or None to leave unchanged.
            hover (str or None): New hovered feature identifier or None to leave unchanged.
            color_scale (list or None): New color scale list or None to leave unchanged.
        """
        if bounds is not None:
            self._bounds = bounds
            self._last_update_time = time.time()
        if zoom is not None:
            self._zoom = zoom
        if selected is not None:
            self._selected_feature = selected
        if hover is not None:
            self._hover_feature = hover
        if color_scale is not None:
            self._color_scale = color_scale

# Initialize map state
map_state = MapState()

@azure_cache_decorator(ttl=600)
def calculate_optimal_viewport(bounds, padding_factor=0.1):
    """
    Computes an optimal map viewport given geographic bounds with proper padding.
    
    This function takes a set of geographic bounds and calculates appropriate 
    viewport settings for the map, including padded bounds, center point, and zoom level.
    It handles edge cases such as invalid or zero-sized bounds and ensures the viewport
    is correctly sized for the data being displayed.
    
    Parameters:
        bounds (tuple): A tuple (min_lat, min_lon, max_lat, max_lon) specifying
                        the geographic area to fit in the viewport.
        padding_factor (float): A factor to apply as padding around the given bounds
                                to avoid overly tight fitting (default: 0.1 = 10%).
    
    Returns:
        dict: A dictionary containing:
              - 'bounds': A 2D list of adjusted bounding coordinates [[min_lat, min_lon], [max_lat, max_lon]].
              - 'center': A 2-element list representing the center [lat, lon].
              - 'zoom': An integer representing the computed zoom level.
    
    Viewport Calculation:
        1. Validates input bounds for correctness
        2. Handles small or zero-sized bounds by adding minimum spacing
        3. Applies padding based on the padding_factor
        4. Calculates center point from padded bounds
        5. Computes appropriate zoom level using calculate_zoom_level()
    
    Default Bounds:
        If bounds are invalid or calculation fails, returns default Canadian bounds:
        - Bounds: [[41, -141], [83, -52]]
        - Center: [56, -96]
        - Zoom: 4
    
    This function is cached for 10 minutes to avoid recalculating for identical bounds.
    
    Usage:
        When a user selects a feature or changes filters, this function computes
        the optimal viewport to ensure all relevant data is visible and properly framed.
    """
    try:
        min_lat, min_lon, max_lat, max_lon = bounds
        
        # Check for invalid or zero-size bounds
        if (min_lat == max_lat and min_lon == max_lon) or \
           any(not isinstance(x, (int, float)) for x in bounds):
            # Return default Canadian bounds
            return {
                'bounds': [[41, -141], [83, -52]],
                'center': [56, -96],
                'zoom': 4
            }
            
        # Handle small differences
        if abs(max_lat - min_lat) < 0.1:
            center_lat = (max_lat + min_lat) / 2
            min_lat = center_lat - 0.05
            max_lat = center_lat + 0.05
        if abs(max_lon - min_lon) < 0.1:
            center_lon = (max_lon + min_lat) / 2
            min_lon = center_lon - 0.05
            max_lon = center_lon + 0.05
            
        lat_padding = (max_lat - min_lat) * padding_factor
        lon_padding = (max_lon - min_lon) * padding_factor
        
        padded_bounds = [
            [min_lat - lat_padding, min_lon - lon_padding],
            [max_lat + lat_padding, max_lon + lon_padding]
        ]
        
        return {
            'bounds': padded_bounds,
            'center': [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2],
            'zoom': calculate_zoom_level(min_lat, min_lon, max_lat, max_lon)
        }
    except Exception:
        # Return default Canadian bounds on any error
        return {
            'bounds': [[41, -141], [83, -52]],
            'center': [56, -96],
            'zoom': 4
        }

@azure_cache_decorator(ttl=600)
def calculate_zoom_level(min_lat, min_lon, max_lat, max_lon):
    """
    Determines an appropriate map zoom level based on the geographic area size.
    
    This function computes a suitable zoom level based on the size of a bounding box,
    with smaller areas getting higher zoom levels (more detail) and larger areas
    getting lower zoom levels (more overview). This ensures that geographic features
    are displayed at an appropriate scale regardless of their physical size.
    
    Parameters:
        min_lat (float): Minimum latitude of the bounding box.
        min_lon (float): Minimum longitude of the bounding box.
        max_lat (float): Maximum latitude of the bounding box.
        max_lon (float): Maximum longitude of the bounding box.
    
    Returns:
        int: A zoom level between 4 and 12, where:
             - 4 is zoomed out (good for country-level view)
             - 12 is zoomed in (good for city-level view)
    
    Zoom Determination Logic:
        1. Calculates the larger of latitude and longitude span
        2. Applies a non-linear scale to map span size to zoom level:
           - ≤0.1°: zoom 12 (very local view)
           - ≤0.25°: zoom 11
           - ≤0.5°: zoom 10
           - ≤1°: zoom 9
           - ≤2°: zoom 8
           - ≤4°: zoom 7
           - ≤8°: zoom 6
           - ≤16°: zoom 5
           - >16°: zoom 4 (continent view)
    
    Using both dimensions ensures appropriate zoom regardless of whether the
    region is tall and narrow or short and wide.
    
    This function is cached for 10 minutes to improve performance when calculating
    zoom levels for the same bounds multiple times.
    """
    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon
    max_diff = max(lat_diff, lon_diff)
    
    # More granular zoom levels
    if max_diff <= 0.1: return 12
    elif max_diff <= 0.25: return 11
    elif max_diff <= 0.5: return 10
    elif max_diff <= 1: return 9
    elif max_diff <= 2: return 8
    elif max_diff <= 4: return 7
    elif max_diff <= 8: return 6
    elif max_diff <= 16: return 5
    else: return 4

class CallbackContextManager:
    """
    Helper class that simplifies access to Dash callback context information.
    
    This class provides a simplified interface to extract and interpret information
    about which inputs triggered a callback and what values they contain. It abstracts
    away the complexity of parsing the callback_context object, making callbacks more
    maintainable and readable.
    
    Attributes:
        _ctx (dash.callback_context): The original Dash callback context.
        _triggered (dict or None): Information about the input that triggered the callback.
        _triggered_id (str or None): The ID of the component that triggered the callback.
    
    Properties:
        triggered_id: Returns the ID string of the triggering component.
        is_triggered: Returns True if the callback was triggered by an input change.
    
    Methods:
        get_input_value(input_id): Gets the value of a specific input that triggered the callback.
    
    Usage:
        ctx_manager = CallbackContextManager(callback_context)
        if ctx_manager.triggered_id == 'my-button':
            # Handle button click
        elif ctx_manager.triggered_id == 'my-dropdown':
            value = ctx_manager.get_input_value('my-dropdown')
            # Handle dropdown change
    """
    def __init__(self, context):
        """
        Initializes the CallbackContextManager with the given Dash callback context.

        Args:
            context (dash.callback_context): The current callback context provided by Dash.
        """
        self._ctx = context
        self._triggered = self._ctx.triggered[0] if self._ctx.triggered else None
        self._triggered_id = self._triggered['prop_id'].split('.')[0] if self._triggered else None

    @property
    def triggered_id(self):
        return self._triggered_id
        
    @property
    def is_triggered(self):
        return bool(self._ctx.triggered)
        
    def get_input_value(self, input_id):
        """Get the value of a specific input that triggered the callback"""
        if self._triggered and self._triggered['prop_id'].startswith(input_id):
            return self._triggered['value']
        return None

@azure_cache_decorator(ttl=3600)  # 1 hour cache
def load_spatial_data():
    """
    Loads geospatial data for provinces and combined CMA/CSD polygons from parquet files.
    This includes coordinates and identifiers necessary for geographic rendering and
    linking data to specific regions. The result is cached to avoid repeated disk reads.

    Returns:
        tuple: A tuple containing:
               - A GeoDataFrame of provinces with cleaned longitude/latitude.
               - A GeoDataFrame of combined CMA/CSD regions with cleaned coordinates.
    """
    
    if NEW_SF:
        if SIMPLIFIED_SF:
            combined_longlat_clean = gpd.read_parquet("data/combined_longlat_simplified.parquet")
        else:
            combined_longlat_clean = gpd.read_parquet("data/combined_longlat_clean.parquet")
    else:
        combined_longlat_clean = gpd.read_parquet("data/combined_longlat_clean - Backup.parquet")
        
    province_longlat_clean = gpd.read_parquet("data/province_longlat_clean.parquet")
    
    return province_longlat_clean, combined_longlat_clean

@azure_cache_decorator(ttl=3600)  # 1 hour cache
def load_and_process_educational_data():
    """
    Loads preprocessed educational data from a pickle file. The data includes variables
    such as STEM/BHASE classification, year, institution type, and associated graduate
    counts. The result is cached to minimize repeated I/O operations.

    Returns:
        pandas.DataFrame: A DataFrame containing cleaned and processed educational data,
                          ready for filtering and aggregation.
    """
    if NEW_DATA:
        data = pd.read_pickle("data/cleaned_data.pkl")
    else:
        data = pd.read_pickle("data/cleaned_data - Backup.pkl")
        data = data.rename(columns={'ISCED_level_of_education': 'ISCED Level of Education',
                           'Credential_Type': 'Credential Type',
                           'CIP_Name': 'CIP Name',
                           'Province_Territory': 'Province or Territory',
                           'year': 'Academic Year',
                           'value': 'Value',
                           'CMA_CA': 'CMA/CSD'})
        
    #data.to_excel('data.xlsx')
    
    categorical_cols = ["STEM/BHASE", "Academic Year", "Province or Territory", "ISCED Level of Education", "Credential Type", "Institution", "CMA/CSD", "DGUID"]
    for col in categorical_cols:
        data[col] = data[col].astype('category')
    data['Value'] = data['Value'].astype('float32')

    # Set a multi-index for direct indexing
    data = data.set_index(["STEM/BHASE", "Academic Year", "Province or Territory", "ISCED Level of Education", "Credential Type", "Institution", "CMA/CSD", "DGUID"]).sort_index()

    return data

def monitor_cache_usage():
    """
    Monitor current cache usage and log statistics.
    Can be called periodically from callbacks to track memory consumption.
    """
    if hasattr(azure_cache, '_cache') and azure_cache._cache is not None:
        try:
            stats = azure_cache._cache.stats()
            mem_usage = len(azure_cache._memory_cache)
            logger.info(f"Cache stats: {stats}")
            logger.info(f"Memory cache items: {mem_usage}/{azure_cache._MAX_MEMORY_ITEMS}")
        except Exception as e:
            logger.warning(f"Failed to monitor cache usage: {e}")

# Load initial data
province_longlat_clean, combined_longlat_clean = load_spatial_data()
data = load_and_process_educational_data()
#data.to_csv('data.csv', index=False)

# Add performance monitoring decorator
def monitor_performance(func):
    metrics = defaultdict(list)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        execution_time = perf_counter() - start_time
        metrics[func.__name__].append(execution_time)
        
        # Log average performance every 10 calls
        if len(metrics[func.__name__]) >= 10:
            avg_time = sum(metrics[func.__name__]) / len(metrics[func.__name__])
            logger.info(f"{func.__name__} average execution time: {avg_time:.4f}s")
            metrics[func.__name__] = []
            
        return result
    return wrapper

class FilterOptimizer:
    """
    Optimizes filtering operations on DataFrames with cached results and vectorized operations.
    
    This class provides efficient filtering of DataFrame data by implementing:
    1. Hash-based caching of filter results
    2. Vectorized operations for maximum performance
    3. Pre-computed indexes for frequent filtering dimensions
    
    Attributes:
        _data (pd.DataFrame): The source DataFrame with multi-level index.
        _cache (dict): Cache mapping filter hash strings to filtered DataFrames.
        _index_cache (dict): Cache of column-specific index values for faster filtering.
        _last_filter_hash (str): Hash of the last filter configuration applied.
    
    Methods:
        _create_filter_hash(filters): Creates a unique hash for filter configurations.
        _create_index(column): Creates and caches an index for a specific column.
        filter_data(filters): Performs optimized filtering with caching.
    
    When filter_data() is called:
    1. A hash is generated from the filter configuration
    2. If the hash exists in cache, the cached result is returned
    3. Otherwise, a vectorized filtering operation is performed
    4. The result is cached before returning
    
    The cache is limited to 10 most recent filter configurations to prevent memory growth.
    """
    def __init__(self, data):
        self._data = data
        self._cache = {}
        self._index_cache = {}
        self._last_filter_hash = None
        
    def _create_filter_hash(self, filters):
        """Create a unique hash for the current filter combination"""
        return hashlib.md5(str(sorted(filters.items())).encode()).hexdigest()
    
    def _create_index(self, column):
        """Create and cache index for faster filtering"""
        if column not in self._index_cache:
            self._index_cache[column] = self._data.index.get_level_values(column)
        return self._index_cache[column]
    
    @monitor_performance
    def filter_data(self, filters):
        """Optimized filtering with caching and vectorized operations"""
        filter_hash = self._create_filter_hash(filters)
        
        # Return cached result if available
        if filter_hash == self._last_filter_hash and filter_hash in self._cache:
            return self._cache[filter_hash]
            
        # Create initial mask
        mask = pd.Series(True, index=self._data.index)
        
        # Apply filters using vectorized operations
        for col, values in filters.items():
            if values:
                idx = self._create_index(col)
                mask &= idx.isin(values)
        
        # Cache and return result
        filtered_data = self._data[mask]
        self._cache[filter_hash] = filtered_data
        self._last_filter_hash = filter_hash
        
        # Limit cache size
        if len(self._cache) > 10:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            
        return filtered_data

# Initialize the optimizer after loading data
filter_optimizer = FilterOptimizer(data)

@azure_cache_decorator(ttl=300)
@monitor_performance
def preprocess_data(selected_stem_bhase, selected_years, selected_provs, selected_isced, 
                  selected_credentials, selected_institutions, selected_cmas):
    """The central data processing pipeline for filtering and aggregating graduate data."""
    
    filters = {
        'STEM/BHASE': set(selected_stem_bhase),
        'Academic Year': set(selected_years),
        'Province or Territory': set(selected_provs),
        'ISCED Level of Education': set(selected_isced),
        'Credential Type': set(selected_credentials),
        'Institution': set(selected_institutions),
        'CMA/CSD': set(selected_cmas),
        'DGUID': set()
    }

    # Use optimized filtering
    filtered_data = filter_optimizer.filter_data(filters)
    if filtered_data.empty:
        return filtered_data.reset_index(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Use vectorized operations instead of parallel processing
    aggregations = {
        'cma': filtered_data.groupby(["CMA/CSD", "DGUID"], observed=True)['Value'].sum(),
        'isced': filtered_data.groupby("ISCED Level of Education", observed=True)['Value'].sum(),
        'province': filtered_data.groupby("Province or Territory", observed=True)['Value'].sum(),
        'credential': filtered_data.groupby("Credential Type", observed=True)['Value'].sum(),
        'institution': filtered_data.groupby("Institution", observed=True)['Value'].sum()
    }
    
    # Pre-allocate DataFrames for better memory efficiency
    cma_grads = pd.DataFrame()
    isced_grads = pd.DataFrame()
    province_grads = pd.DataFrame()
    credential_grads = pd.DataFrame()
    institution_grads = pd.DataFrame()
    
    # Perform aggregations with optimized settings
    with pd.option_context('mode.chained_assignment', None):
        cma_grads = aggregations['cma'].reset_index(name='graduates')
        isced_grads = aggregations['isced'].reset_index(name='graduates')
        province_grads = aggregations['province'].reset_index(name='graduates')
        credential_grads = aggregations['credential'].reset_index(name='graduates')
        institution_grads = aggregations['institution'].reset_index(name='graduates')

    return (
        filtered_data.reset_index(),
        cma_grads,
        isced_grads,
        province_grads,
        credential_grads,
        institution_grads
    )

def create_geojson_feature(row, colorscale, max_graduates, min_graduates, selected_feature):
    """
    Creates a GeoJSON feature dictionary for a single geographic unit with styling.
    
    This function transforms a row from a GeoDataFrame into a properly formatted 
    GeoJSON feature with styling properties, tooltips, and selection highlighting.
    It applies color scaling based on graduate counts and handles selection state.
    
    Parameters:
        row (pandas.Series): A row from a GeoDataFrame representing one geographic unit (CMA/CSD).
        colorscale (list): A list of color strings for representing graduate counts.
        max_graduates (int): The maximum graduates count in the dataset for normalization.
        min_graduates (int): The minimum graduates count in the dataset for normalization.
        selected_feature (str or None): The DGUID of the currently selected feature, if any.
    
    Returns:
        dict: A dictionary representing a GeoJSON feature with:
            - 'graduates': Number of graduates in this geographic unit
            - 'DGUID': Geographic unit identifier
            - 'CMA/CSD': Name of the Census Metropolitan Area or Census Agglomeration
            - 'style': Visual styling properties (colors, weights, opacity)
            - 'tooltip': HTML content for the hover tooltip
    
    Style Determination:
        1. If the feature is selected, uses red fill and black border
        2. If not selected, color is determined by normalized graduate count:
           - Higher counts get darker red colors
           - Zero counts get light gray
        3. Selected features have thicker borders (2px vs 0.5px)
    
    Geometry Processing:
        Applies geometry simplification (0.01) to reduce complexity and improve
        rendering performance, particularly important for complex polygons.
    
    Tooltip Formatting:
        Creates an HTML tooltip showing the CMA/CSD name and formatted graduate count
        with consistent Open Sans font styling and thousands separators.
    """
    graduates = float(row['graduates'])  # Convert to float for faster comparisons
    dguid = str(row['DGUID'])
    is_selected = selected_feature and dguid == selected_feature
    
    if max_graduates > min_graduates:
        normalized_value = (graduates - min_graduates) / (max_graduates - min_graduates)
        color_index = int(normalized_value * (len(colorscale) - 1))
        color = colorscale[color_index] if graduates > 0 else 'lightgray'
    else:
        color = colorscale[0] if graduates > 0 else 'lightgray'
    
    # Add feature simplification for better performance
    if row.geometry:
        geometry = row.geometry.simplify(0.01).__geo_interface__
    else:
        geometry = None

    return {
        'graduates': int(graduates),
        'DGUID': dguid,
        'CMA/CSD': row['CMA/CSD'],
        'style': {
            'fillColor': bc.MAIN_RED if is_selected else color,
            'color': bc.IIC_BLACK if is_selected else bc.GREY,
            'weight': 2 if is_selected else 0.5,
            'fillOpacity': 0.8
        },
        'tooltip': f"<div style='font-family: Open Sans, sans-serif; font-weight: 600;'>{row['CMA/CSD']}: {int(graduates):,}</div>"
    }

@azure_cache_decorator(ttl=300)
def create_chart(dataframe, x_column, y_column, x_label, selected_value=None):
    """
    Creates a standardized horizontal bar chart for visualization dimensions.
    
    This function transforms aggregated data into interactive Plotly bar charts
    with consistent styling, sorting, and highlighting of selected values. Charts
    are responsive to selections and maintain visual consistency across the dashboard.
    
    Parameters:
        dataframe (pd.DataFrame): Data to plot, typically from preprocess_data() aggregations.
        x_column (str): Column name for category labels (e.g., 'Province or Territory').
        y_column (str): Column name for numeric values (typically 'graduates').
        x_label (str): Label for the chart title (e.g., 'Province/Territory').
        selected_value (str, optional): Currently selected value to highlight in red.
    
    Returns:
        dict: A complete Plotly figure object ready for rendering, or an empty dict if
              the input data is invalid or empty.
    
    Chart Behavior:
        - Sorts data by value in ascending order (smallest to largest)
        - Formats numeric values with thousands separators
        - Uses horizontal orientation for easier category label reading
        - Dynamically adjusts height based on number of items (for Institution and CMA charts)
        - Highlights the selected value in red if provided
        - Uses a red color scale when no selection is active
        - Provides hover tooltips with formatted graduate counts
        - Sets up consistent styling using brand colors
    
    Visual Features:
        - Text labels showing values outside bars
        - Formatted numbers with thousands separators
        - Consistent font family (Open Sans)
        - Optimized margins and layout settings
        - Custom hover templates with clean formatting
        - Simplified modebar with unnecessary controls removed
    
    Selection Highlighting:
        When a selected_value is provided, the corresponding bar is colored red
        (bc.MAIN_RED) while other bars are colored light grey (bc.LIGHT_GREY).
        This provides clear visual feedback on the current selection state.
    
    Error Handling:
        If dataframe is None or empty, returns an empty dict to avoid rendering errors.
    """
    if dataframe is None or dataframe.empty:
        return {}
    
    sorted_data = dataframe.sort_values(y_column, ascending=True)
    sorted_data['text'] = sorted_data[y_column].apply(lambda x: f'{int(x):,}')
    
    chart_height=500
    
    #If x_label = 'Institution' or 'CMA/CSD', then chart_height=1000
    if x_label == 'Institution' or x_label == 'Census Metropolitan Area':
        num_bars =len(sorted_data.index)
        max_height=max(chart_height, 25 * len(sorted_data.index))
        chart_height= max_height
        # filename = replace spaces in x_label with _
        #filename = x_label.replace(' ', '_')
        #sorted_data.to_csv(filename+'.csv', index=False)
    
    # Create figure using go.Figure instead of px.bar
    fig = go.Figure(
        data=go.Bar(
            x=sorted_data[y_column],
            y=sorted_data[x_column],
            orientation='h',
            text=sorted_data['text'],
            textposition='outside',
            cliponaxis=False,
            textfont=dict(color=bc.IIC_BLACK, size=12, family='Open Sans', weight=600),
            hovertemplate='%{y}: %{x:,}<extra></extra>',
            hoverlabel=dict(
                bgcolor='white',
                font_color='black',
                font_size=14,
                font_family='Open Sans',
                bordercolor=bc.IIC_BLACK
            ),
            marker=dict(
                color=sorted_data[y_column] if not selected_value else [bc.MAIN_RED if x == selected_value else bc.LIGHT_GREY for x in sorted_data[x_column]],
                colorscale=COLOUR_SCALE if not selected_value else None
            )
        ), layout = go.Layout(
                              xaxis={'title': 'x-axis','fixedrange':True},
                              yaxis={'title': 'y-axis','fixedrange':True})#, layout={'height': 5000}
    )
    
    '''
    title=dict(
        text=f'Number of Graduates by {x_label}',
        font=dict(size=16,
                  family='Open Sans',
                  weight=600)  # Increased title size
    ),
    '''
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_title=None,
        yaxis_title=None,
        height=chart_height,  # Keeping 500 as default but can be adjusted
        margin=dict(l=5, r=50, t=25, b=5),
        clickmode='event+select',
        plot_bgcolor='#D5DADC',
        paper_bgcolor='white',
        font=dict(
            color=bc.IIC_BLACK,
            family='Open Sans',
            weight=600
        ),
        yaxis=dict(
            ticksuffix='  ',  # Add space after tick labels
            separatethousands=True,
            automargin=True,  # Automatically adjust margins
        ),
        modebar_remove=['zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'lasso2d']
    )
    
    return fig

def create_empty_response():
    """
    Produces a set of empty or default return values for callbacks that must always
    return consistent output structures even when data is unavailable or an error
    occurs. The empty response includes:
    - An empty GeoJSON structure for the map.
    - Empty figure objects for charts.
    - Empty lists for table data and columns.
    - A default viewport centered on Canada.

    Returns:
        tuple: A tuple containing empty or default components:
               - Empty geojson data (dict)
               - Empty figure for ISCED chart (dict)
               - Empty figure for province chart (dict)
               - Empty lists for table data and columns (list, list)
               - A default viewport dictionary with bounds and transition
    """
    empty_geojson = {'type': 'FeatureCollection', 'features': []}
    empty_fig = {}
    empty_row_data = []
    empty_column_defs = []
    default_bounds = [[41, -141], [83, -52]]  # Canada bounds
    
    return (
        empty_geojson,
        empty_fig,
        empty_fig,
        empty_row_data,  # Changed from empty_data
        empty_column_defs,  # Changed from empty_columns
        dict(bounds=default_bounds, transition="flyToBounds")
    )

def update_selected_value(click_data, n_clicks, stored_value, triggered_id, clear_id, chart_id, figure=None):
    """
    A generic helper function that updates a selected value based on chart click interactions
    and a clear selection button.

    Args:
        click_data (dict or None): Data from a chart click event. Contains 'points' with clicked category.
        n_clicks (int): Number of times the clear selection button was clicked.
        stored_value (str or None): Currently stored selected value.
        triggered_id (str): The ID of the component that triggered this callback.
        clear_id (str): The ID of the 'Clear Selection' button.
        chart_id (str): The ID of the chart component that triggers selection updates.
        figure (dict or None): The chart figure dictionary. If provided and the chart is a bar chart,
                               its orientation helps determine which axis value to read.

    Returns:
        str or None: The updated selected value. Returns None to clear it.
    """
    if triggered_id == clear_id:
        return None

    if triggered_id == chart_id and click_data and 'points' in click_data:
        orientation = 'v'
        if figure and figure.get('data') and figure['data'][0].get('orientation') == 'h':
            orientation = 'h'
        
        if orientation == 'h':
            clicked_value = click_data['points'][0]['y']
        else:
            clicked_value = click_data['points'][0]['x']

        return None if stored_value == clicked_value else clicked_value

    return stored_value

# Generate initial filter options
stem_bhase_options_full = [{'label': stem, 'value': stem} for stem in sorted(data.index.get_level_values('STEM/BHASE').unique())]
year_options_full = [{'label': year, 'value': year} for year in sorted(data.index.get_level_values('Academic Year').unique())]
prov_options_full = [{'label': prov, 'value': prov} for prov in sorted(data.index.get_level_values('Province or Territory').unique())]
cma_options_full = [{'label': cma, 'value': cma} for cma in sorted(data.index.get_level_values('CMA/CSD').unique())]
isced_options_full = [{'label': level, 'value': level} for level in sorted(data.index.get_level_values('ISCED Level of Education').unique())]
credential_options_full = [{'label': cred, 'value': cred} for cred in sorted(data.index.get_level_values('Credential Type').unique())]
institution_options_full = [{'label': inst, 'value': inst} for inst in sorted(data.index.get_level_values('Institution').unique())]

# Extract and write values only from option dictionaries
#def write_values_to_file(options, filename):
    #values = [item['value'] for item in options]
    #with open(filename, 'w') as f:
        #for value in values:
            #f.write(f"{value}\n")

# Write all option values to respective files
#write_values_to_file(stem_bhase_options_full, 'stem_bhase_options_full.txt')
#write_values_to_file(year_options_full, 'year_options_full.txt')
#write_values_to_file(prov_options_full, 'prov_options_full.txt')
#write_values_to_file(cma_options_full, 'cma_options_full.txt')
#write_values_to_file(isced_options_full, 'isced_options_full.txt')
#write_values_to_file(credential_options_full, 'credential_options_full.txt')
#write_values_to_file(institution_options_full, 'institution_options_full.txt')

app.layout = html.Div([
    html.Link(
        rel='stylesheet',
        href='/assets/assets/pivottable.css'
    ),
    dcc.Store(id='client-data-store', storage_type='session'),
    dcc.Store(id='client-filters-store', storage_type='local'),
    create_layout(data, stem_bhase_options_full, year_options_full, prov_options_full, isced_options_full, credential_options_full, institution_options_full, cma_options_full)
])

def calculate_viewport_update(triggered_id, cma_data, selected_feature=None):
    """
    Determines an appropriate viewport update based on the user interaction that
    triggered the callback. If the triggered action is selecting a CMA/CSD feature,
    it adjusts the viewport to zoom in on that feature. If filters changed, it
    recalculates bounds to show all visible features. If no adjustments are needed,
    it returns None.

    Args:
        triggered_id (str): The ID of the component that triggered the callback.
        cma_data (geopandas.GeoDataFrame): The spatial data with graduates information.
        selected_feature (str or None): The currently selected CMA/CSD DGUID, if any.

    Returns:
        dict or None: A dictionary defining the new viewport or None if no update
                      is required.
    """
    if triggered_id == 'selected-feature' and selected_feature:
        # For clicked features, zoom to the selected feature
        selected_geometry = cma_data[cma_data['DGUID'] == selected_feature]
        if not selected_geometry.empty:
            bounds = selected_geometry.total_bounds
            viewport = calculate_optimal_viewport(tuple(bounds))
            viewport['zoom'] = 8  # Fixed zoom for selections
            return viewport
    elif triggered_id in ['stem-bhase-filter', 'year-filter', 'prov-filter', 
                          'isced-filter', 'credential-filter', 'institution-filter',
                          'reset-filters', 'clear-selection']:
        # For filter changes, zoom to show all visible features
        if not cma_data.empty:
            bounds = cma_data.total_bounds
            return calculate_optimal_viewport(tuple(bounds))
    
    return None

# Optimized callback for map selection
@app.callback(
    Output('selected-feature', 'data'),
    Input('cma-geojson', 'clickData'),
    Input('clear-selection', 'n_clicks'),
    State('selected-feature', 'data'),
    prevent_initial_call=True
)
def update_selected_feature(click_data, n_clicks, stored_cma):
    """
    Updates the currently selected CMA/CSD feature when the map is clicked. If the
    'Clear Selection' button is clicked, resets the selection. This ensures that
    the selected feature state remains synchronized with user actions on the map.

    Args:
        click_data (dict or None): Data associated with a map feature click.
        n_clicks (int): The number of times the 'Clear Selection' button was clicked.
        stored_cma (str or None): The currently stored CMA/CSD DGUID.

    Returns:
        str or None: The updated selected CMA/CSD DGUID or None if the selection is cleared.
    """
    ctx_manager = CallbackContextManager(callback_context)
    
    if not ctx_manager.is_triggered:
        raise PreventUpdate
        
    if ctx_manager.triggered_id == 'clear-selection':
        return None
        
    if ctx_manager.triggered_id == 'cma-geojson' and click_data and 'points' in click_data:
        clicked_id = click_data['points'][0]['featureId']
        return None if stored_cma == clicked_id else clicked_id
        
    return stored_cma

@app.callback(
    Output({'type': 'store', 'item': MATCH}, 'data'),
    Input({'type': 'graph', 'item': MATCH}, 'clickData'),
    Input('clear-selection', 'n_clicks'),
    State({'type': 'store', 'item': MATCH}, 'data'),
    State({'type': 'graph', 'item': MATCH}, 'figure'),
    prevent_initial_call=True
)
def update_selection(clickData, n_clicks, stored_value, figure):
    """
    Manages selection state for all chart visualizations using pattern matching.
    
    This callback processes clicks on any chart and updates the corresponding selection
    state. It uses Dash's pattern matching to handle all charts with a single callback,
    making the selection system maintainable and consistent. The pattern matching is
    based on the 'item' key which identifies the dimension (e.g., 'isced', 'province').
    
    Triggers:
        - Clicking on any bar in any chart visualization
        - Clicking the "Clear Selection" button
    
    Parameters:
        clickData (dict or None): Data from chart click event containing the clicked value.
        n_clicks (int): Number of times "Clear Selection" button has been clicked.
        stored_value (str or None): Currently stored selected value.
        figure (dict): The current figure object to determine bar orientation.
    
    Returns:
        str or None: The new selected value, or None to clear selection.
    
    Selection Logic:
        1. If "Clear Selection" button is clicked, returns None for all stores.
        2. For chart clicks:
           - Extracts dimension type from pattern-matched component ID
           - Determines if chart is horizontal or vertical
           - Gets clicked value based on orientation (x or y coordinate)
           - If clicked value matches stored value, toggles off (returns None)
           - Otherwise sets the new clicked value as selection
    
    Pattern Matching:
        Uses {'type': 'store', 'item': MATCH} to match with corresponding store.
        The 'item' key connects the graph, store, and data dimension (e.g., 'isced').
    
    This system creates a toggle behavior where:
    - First click on an item selects it
    - Second click on same item deselects it
    - Click on different item changes selection to that item
    - "Clear Selection" button deselects everything
    
    Once a value is stored in the store component, it triggers the update_visualizations
    callback which applies cross-filtering based on the selection.
    """
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    triggered_prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_prop_id == 'clear-selection':
        return None
    
    if clickData and 'points' in clickData:
        try:
            pattern_dict = json.loads(triggered_prop_id.replace("'", "\""))
            item_type = pattern_dict.get('item')
            
            orientation = 'v'
            if figure and 'data' in figure and figure['data'][0].get('orientation') == 'h':
                orientation = 'h'
            
            if orientation == 'h':
                clicked_value = clickData['points'][0]['y']
            else:
                clicked_value = clickData['points'][0]['x']
                
            return None if stored_value == clicked_value else clicked_value
        except Exception as e:
            logger.error(f"Error parsing pattern ID: {e}")
            return stored_value
    
    return stored_value

@app.callback(
    Output('cma-geojson', 'data'),
    Output({'type': 'graph', 'item': 'isced'}, 'figure'),
    Output({'type': 'graph', 'item': 'province'}, 'figure'),
    Output({'type': 'graph', 'item': 'cma'}, 'figure'),
    Output({'type': 'graph', 'item': 'credential'}, 'figure'),
    Output({'type': 'graph', 'item': 'institution'}, 'figure'),
    Output('map', 'viewport'),
    Input('stem-bhase-filter', 'value'),
    Input('year-filter', 'value'),
    Input('prov-filter', 'value'),
    Input('isced-filter', 'value'),
    Input('credential-filter', 'value'),
    Input('institution-filter', 'value'),
    Input('cma-filter', 'value'),
    Input({'type': 'store', 'item': 'isced'}, 'data'),
    Input({'type': 'store', 'item': 'province'}, 'data'),
    Input('selected-feature', 'data'),
    Input({'type': 'store', 'item': 'credential'}, 'data'),
    Input({'type': 'store', 'item': 'institution'}, 'data'),
    Input({'type': 'store', 'item': 'cma'}, 'data'),
    State('map', 'viewport')
)
def update_visualizations(*args):
    """
    The central cross-filtering callback that coordinates all visualizations.
    
    This callback is the heart of the application's interactive cross-filtering system.
    It responds to both filter changes and visualization selections to update all
    visualizations simultaneously, ensuring they remain synchronized and reflect the
    current filtered and selected state.
    
    Triggers:
        - Changes to any filter (STEM/BHASE, Year, Province, etc.)
        - Selection of any element (map feature, chart bar)
        - Clearing of selection
        - Reset of filters
    
    Processing Flow:
        1. Identifies which input triggered the callback
        2. Processes data through preprocess_data() with current filters
        3. If any selection exists (ISCED, Province, etc.), applies cross-filtering:
           - Creates a boolean mask for each active selection
           - Combines masks with logical AND operations
           - Re-aggregates data based on the combined mask
        4. Builds map visualization:
           - Merges filtered data with geographic features
           - Creates GeoJSON with dynamic styling based on graduate counts
           - Highlights selected features
        5. Creates chart visualizations:
           - Generates horizontal bar charts for each dimension
           - Applies consistent styling and highlighting
        6. Updates map viewport if necessary:
           - Centers on selection if a feature was clicked
           - Shows all visible features after filter changes
    
    Outputs:
        - GeoJSON data for the map
        - Figure objects for each chart (ISCED, Province, CMA, Credential, Institution)
        - Map viewport settings
    
    Performance Considerations:
        - Monitors cache usage during major updates
        - Only updates viewport when necessary
        - Uses efficient vectorized operations throughout
        - Uses a try-except pattern to ensure graceful degradation
        - Returns cached empty responses for error states
    
    Cross-Filtering Mechanism:
        When a user clicks any visualization element, this callback:
        1. Receives the selection via store components
        2. Applies the selection as additional filters
        3. Updates all visualizations to reflect the filtered view
        4. Highlights the selected element across all applicable visualizations
    
    This creates a unified visual query system where selections in any component
    affect all other components, enabling powerful exploratory analysis.
    """
    try:
        current_viewport = args[-1]
        (stem_bhase, years, provs, isced, credentials, institutions, cma_filter,
         selected_isced, selected_province, selected_feature, 
         selected_credential, selected_institution, selected_cma) = args[:-1]
        
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
            
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if triggered_id.startswith('{'):
            try:
                pattern_dict = json.loads(triggered_id.replace("'", "\""))
                if pattern_dict.get('type') == 'store':
                    triggered_id = f"selected-{pattern_dict.get('item')}"
            except:
                pass

        # Process data with optimized function
        filtered_data, cma_grads, isced_grads, province_grads, credential_grads, institution_grads = preprocess_data(
            tuple(stem_bhase or []),
            tuple(years or []),
            tuple(provs or []),
            tuple(isced or []),
            tuple(credentials or []),
            tuple(institutions or []),
            tuple(cma_filter or [])  # Add CMA filter to preprocess_data
        )
        
        # Apply cross-filtering with vectorized operations
        if any([selected_isced, selected_province, selected_feature, selected_credential, selected_institution, selected_cma]):
            mask = pd.Series(True, index=filtered_data.index)
            if selected_isced:
                mask &= filtered_data['ISCED Level of Education'] == selected_isced
            if selected_province:
                mask &= filtered_data['Province or Territory'] == selected_province
            if selected_feature:
                mask &= filtered_data['DGUID'] == selected_feature
            if selected_credential:
                mask &= filtered_data['Credential Type'] == selected_credential
            if selected_institution:
                mask &= filtered_data['Institution'] == selected_institution
            if selected_cma:
                mask &= filtered_data['CMA/CSD'] == selected_cma

            filtered_data = filtered_data[mask]
            if filtered_data.empty:
                return create_empty_response()

            # Aggregations with observed=True
            cma_grads = filtered_data.groupby(["CMA/CSD", "DGUID"], observed=True)['Value'].sum().reset_index(name='graduates')
            isced_grads = filtered_data.groupby("ISCED Level of Education", observed=True)['Value'].sum().reset_index(name='graduates')
            province_grads = filtered_data.groupby("Province or Territory", observed=True)['Value'].sum().reset_index(name='graduates')
            # Add aggregations for new charts
            #cma_aggregation = filtered_data.groupby("CMA/CSD", observed=True)['Value'].sum().reset_index(name='graduates')
            credential_grads = filtered_data.groupby("Credential Type", observed=True)['Value'].sum().reset_index(name='graduates')
            institution_grads = filtered_data.groupby("Institution", observed=True)['Value'].sum().reset_index(name='graduates')

        # Prepare map data efficiently
        cma_data = combined_longlat_clean.merge(cma_grads, on='DGUID', how='left')
        cma_data['graduates'] = cma_data['graduates'].fillna(0)
        cma_data = cma_data[cma_data['graduates'] > 0]
        
        if cma_data.empty:
            return create_empty_response()
        
        should_update_viewport = triggered_id in [
            'stem-bhase-filter', 'year-filter', 'prov-filter',
            'isced-filter', 'credential-filter', 'institution-filter',
            'selected-feature', 'clear-selection', 'reset-filters'
        ]
        
        if should_update_viewport:
            bounds = cma_data.total_bounds
            lat_padding = (bounds[3] - bounds[1]) * 0.1
            lon_padding = (bounds[2] - bounds[0]) * 0.1
            map_bounds = [
                [bounds[1] - lat_padding, bounds[0] - lon_padding],
                [bounds[3] + lat_padding, bounds[2] + lon_padding]
            ]
            viewport_output = dict(
                bounds=map_bounds,
                transition=dict(duration=1000)
            )
        else:
            viewport_output = dash.no_update
            
        # Create GeoJSON efficiently - updated to use built-in color scale
        max_graduates = cma_data['graduates'].max()
        min_graduates = cma_data['graduates'].min()
        
        if max_graduates > min_graduates:
            normalized_values = (cma_data['graduates'] - min_graduates) / (max_graduates - min_graduates)
            colors = px.colors.sample_colorscale(COLOUR_SCALE, normalized_values)
        else:
            colors = [COLOUR_SCALE[-1]] * len(cma_data)
        
        features = [
            {
                'type': 'Feature',
                'geometry': row.geometry.__geo_interface__,
                'properties': {
                    'graduates': int(row['graduates']),
                    'DGUID': str(row['DGUID']),
                    'CMA/CSD': row['CMA/CSD'],
                    'style': {
                        'fillColor': color if row['graduates'] > 0 else 'lightgray',
                        'color': bc.IIC_BLACK if row['DGUID'] == selected_feature else bc.GREY,
                        'weight': 2 if row['DGUID'] == selected_feature else 0.75,
                        'fillOpacity': 0.8
                    },
                    'tooltip': f"<div style='font-family: Open Sans, sans-serif; font-weight: 600;'>{row['CMA/CSD']}: {int(row['graduates']):,}</div>"
                }
            }
            for (_, row), color in zip(cma_data.iterrows(), colors)
        ]
        
        geojson_data = {'type': 'FeatureCollection', 'features': features}
        
        # Create charts with selections (simplified without chart type parameter)
        fig_isced = create_chart(
            isced_grads, 
            'ISCED Level of Education', 
            'graduates',
            'ISCED Level of Education', 
            selected_isced
        )
        
        fig_province = create_chart(
            province_grads, 
            'Province or Territory', 
            'graduates',
            'Province/Territory', 
            selected_province
        )

        # Create new charts
        fig_cma = create_chart(
            cma_grads,
            'CMA/CSD',
            'graduates',
            'Census Metropolitan Area',
            selected_feature
        )

        fig_credential = create_chart(
            credential_grads,
            'Credential Type',
            'graduates',
            'Credential Type',
            selected_credential
        )

        fig_institution = create_chart(
            institution_grads,
            'Institution',
            'graduates',
            'Institution',
            selected_institution
        )
        
        # Monitor cache at the start of major updates
        monitor_cache_usage()
        
        return (
            geojson_data,
            fig_isced,
            fig_province,
            fig_cma,
            fig_credential,
            fig_institution,
            viewport_output
        )
        
    except Exception as e:
        logger.error(f"Error in update_visualizations: {str(e)}")
        return create_empty_response()

@app.callback(
    Output('stem-bhase-filter', 'value'),
    Output('year-filter', 'value'),
    Output('prov-filter', 'value'),
    Output('isced-filter', 'value'),
    Output('credential-filter', 'value'),
    Output('institution-filter', 'value'),
    Output('cma-filter', 'value'),  # Add CMA filter output
    Input('reset-filters', 'n_clicks'),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    """
    Resets all data filters back to their default values. This clears any user-applied
    restrictions, ensuring that the dataset and visualizations return to their initial
    state.

    Args:
        n_clicks (int): The number of times the 'Reset Filters' button has been clicked.

    Returns:
        tuple: A tuple containing the default values for each filter component.
    """
    if not n_clicks:
        raise PreventUpdate
        
    return (
        [option['value'] for option in stem_bhase_options_full],
        [option['value'] for option in year_options_full],
        [], [], [], [], []  # Added empty list for CMA filter
    )

@azure_cache_decorator(ttl=300)  # 5 minutes - filter options change with selections
def filter_options(data, column, selected_filters):
    """
    Returns available options for a filter based on currently selected values in other filters.
    
    Args:
        data (pandas.DataFrame): The source data
        column (str): The column name to get options for
        selected_filters (dict): Dictionary of currently selected values for other filters
    
    Returns:
        list: List of available options as dictionaries with 'label' and 'value' keys
    """
    mask = pd.Series(True, index=data.index)
    for col, values in selected_filters.items():
        if values and col != column:
            mask &= data.index.get_level_values(col).isin(values)
    
    available_values = sorted(data[mask].index.get_level_values(column).unique())
    return [{'label': val, 'value': val} for val in available_values]

@app.callback(
    Output('stem-bhase-filter', 'options'),
    Output('year-filter', 'options'),
    Output('prov-filter', 'options'),
    Output('cma-filter', 'options'),
    Output('isced-filter', 'options'),
    Output('credential-filter', 'options'),
    Output('institution-filter', 'options'),
    Input('stem-bhase-filter', 'value'),
    Input('year-filter', 'value'),
    Input('prov-filter', 'value'),
    Input('isced-filter', 'value'),
    Input('credential-filter', 'value'),
    Input('institution-filter', 'value'),
    Input('cma-filter', 'value')
)
def update_filter_options(stem_bhase, years, provs, isced, credentials, institutions, cmas):
    """Updates filter options based on other filter selections"""
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    # Create dictionary of current filter selections
    current_filters = {
        'STEM/BHASE': stem_bhase or [],
        'Academic Year': years or [],
        'Province or Territory': provs or [],
        'CMA/CSD': cmas or [],
        'ISCED Level of Education': isced or [],
        'Credential Type': credentials or [],
        'Institution': institutions or []
    }
    
    # Get updated options for each filter
    stem_options = filter_options(data, 'STEM/BHASE', {k:v for k,v in current_filters.items() if k != 'STEM/BHASE'})
    year_options = filter_options(data, 'Academic Year', {k:v for k,v in current_filters.items() if k != 'Academic Year'})
    prov_options = filter_options(data, 'Province or Territory', {k:v for k,v in current_filters.items() if k != 'Province or Territory'})
    cma_options = filter_options(data, 'CMA/CSD', {k:v for k,v in current_filters.items() if k != 'CMA/CSD'})
    isced_options = filter_options(data, 'ISCED Level of Education', {k:v for k,v in current_filters.items() if k != 'ISCED Level of Education'})
    cred_options = filter_options(data, 'Credential Type', {k:v for k,v in current_filters.items() if k != 'Credential Type'})
    inst_options = filter_options(data, 'Institution', {k:v for k,v in current_filters.items() if k != 'Institution'})
    
    return (
        stem_options,
        year_options,
        prov_options,
        cma_options,
        isced_options,
        cred_options,
        inst_options
    )

@app.callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    State("pivot-table", "data"),
    State("pivot-table", "cols"),
    State("pivot-table", "rows"),
    State("pivot-table", "vals"),
    prevent_initial_call=True,
)
def download_pivot_data(n_clicks, data, cols, rows, vals):
    """
    Creates a downloadable CSV file from the pivot table's current configuration.
    
    Args:
        n_clicks (int): Number of times download button clicked
        data (list): The raw data from the pivot table
        cols (list): Column headers configured in the pivot table
        rows (list): Row headers configured in the pivot table
        vals (list): Value fields configured in the pivot table
        
    Returns:
        dict: Download specification for Dash
    """
    if not n_clicks or not data:
        raise PreventUpdate

    try:
        # Create a pandas DataFrame from the pivot table data
        df = pd.DataFrame(data)
        
        # Get unique values for rows and columns
        row_values = []
        for row in rows:
            if row in df.columns:
                row_values.append(sorted(df[row].unique()))
                
        col_values = []
        for col in cols:
            if col in df.columns:
                col_values.append(sorted(df[col].unique()))
                
        # Create MultiIndex for rows and columns
        if row_values:
            row_index = pd.MultiIndex.from_product(row_values, names=rows)
        else:
            row_index = pd.Index([])
            
        if col_values:
            col_index = pd.MultiIndex.from_product(col_values, names=cols)
        else:
            col_index = pd.Index([])
            
        # Create pivot table
        pivot_df = df.pivot_table(
            index=rows if rows else None,
            columns=cols if cols else None,
            values=vals,
            aggfunc='sum',
            fill_value=0
        )
        
        # Convert to string buffer
        string_buffer = io.StringIO()
        pivot_df.to_csv(string_buffer)
        
        return dict(
            content=string_buffer.getvalue(),
            filename=f"graduates_pivot_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            type="text/csv",
            base64=False
        )
        
    except Exception as e:
        logger.error(f"Error in download_pivot_data: {str(e)}")
        raise PreventUpdate

@app.callback(
    Output("horizontal-collapse", "is_open"),
    [Input("horizontal-collapse-button", "n_clicks")],
    [State("horizontal-collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# we use a callback to toggle the collapse on small screens
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# the same function (toggle_navbar_collapse) is used in all three callbacks
app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)(toggle_navbar_collapse)

app.clientside_callback(
    """
    function set_event(map_id) {
        // On resize event 
        var callback = function() {
            window.dispatchEvent(new Event('resize'));
        }

        new ResizeObserver(callback).observe(document.getElementById(map_id))

        return dash_clientside.no_update;
    }
    """,
    Output("map", "id"),
    Input("map", "id")
)

def load_user_guide():
    """Load user guide markdown"""
    user_guide_path = Path("user_guide.md")
    if (user_guide_path.exists()):
        with open(user_guide_path, "r", encoding="utf-8") as f:
            content = f.read()
            return dcc.Markdown(content)
    return "User guide not available"

@app.callback(
    Output("user-guide-modal", "is_open"),
    Output("user-guide-content", "children"),
    [
        Input("open-guide-button", "n_clicks"),
        Input("close-guide-button", "n_clicks")
    ],
    State("user-guide-modal", "is_open"),
)
def toggle_user_guide(open_clicks, close_clicks, is_open):
    """Toggle user guide modal and load content"""
    if not any(clicks for clicks in [open_clicks, close_clicks]):
        raise PreventUpdate
        
    if open_clicks or close_clicks:
        if not is_open:
            # Only load content when opening
            return not is_open, html.Div(
                #dangerously_allow_html=True,
                children=load_user_guide()
            )
        return not is_open, dash.no_update
        
    return is_open, dash.no_update

@app.callback(
    Output("faq-modal", "is_open"),
    [
        Input("open-faq-button", "n_clicks"),
        Input("close-faq-button", "n_clicks")
    ],
    State("faq-modal", "is_open"),
)
def toggle_faq(open_clicks, close_clicks, is_open):
    """Toggle FAQ modal and load content"""
    if not any(clicks for clicks in [open_clicks, close_clicks]):
        raise PreventUpdate
        
    if open_clicks or close_clicks:
        if not is_open:
            return not is_open
        return not is_open
        
    return is_open

if __name__ == '__main__':
    app.run_server(debug=False)
