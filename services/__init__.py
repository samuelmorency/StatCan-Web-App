"""Services package initialization."""

from services.cache_service import azure_cache, azure_cache_decorator, cache

__all__ = ['azure_cache', 'azure_cache_decorator', 'cache']