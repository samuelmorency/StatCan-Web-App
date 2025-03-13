"""Configuration settings for performance optimizations."""

# Master feature flags for controlling optimizations
OPTIMIZATION_CONFIG = {
    'use_patch': True,          # Master switch for all Patch optimizations
    'highlight_only': True,     # Use Patch for selection highlighting updates
    'viewport_only': True,      # Use Patch for map viewport updates only
    'measure_performance': True # Enable performance logging
}