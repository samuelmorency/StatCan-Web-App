"""Configuration management for the Infozone application."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# App configuration
APP_CONFIG = {
    "title": "Canadian STEM/BHASE Graduates Dashboard",
    "debug": os.getenv("DEBUG", "False").lower() == "true",
    "cache_dir": os.getenv("CACHE_DIR", "cache"),
    "cache_size": float(os.getenv("CACHE_SIZE_MB", "1024")) * 1e6,
    "cache_ttl": int(os.getenv("CACHE_TTL", "3600")),
    "log_level": os.getenv("LOG_LEVEL", "INFO")
}

# Map configuration
MAP_CONFIG = {
    "default_center": [56, -96],
    "default_zoom": 4,
    "bounds": [[36.676556, -141.001735], [68.110626, -52.620422]]
}

# UI configuration
UI_CONFIG = {
    "font_family": "Open Sans, sans-serif",
    "font_weight": "600",
    "tab_style": {
        "backgroundColor": "#cccccc", 
        "borderColor": "#F1F1F1", 
        "color": "black",
        "font-family": 'Open Sans',
        "font-weight": "600"
    },
    "active_tab_style": {
        "backgroundColor": "#F1F1F1", 
        "borderColor": "#F1F1F1", 
        "color": "black",
        "font-family": 'Open Sans',
        "font-weight": "600"
    }
}