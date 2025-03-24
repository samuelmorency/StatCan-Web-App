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

import cache_utils, data_utils

NEW_DATA = True
NEW_SF = True
SIMPLIFIED_SF = True
COLOUR_SCALE = bc.BRIGHT_RED_SCALE

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
server = app.server

cache_utils.initialize_cache()

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

@cache_utils.azure_cache_decorator(ttl=600)
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

@cache_utils.azure_cache_decorator(ttl=600)
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

# Load initial data
combined_longlat_clean = cache_utils.azure_cache_decorator(ttl=3600)(lambda: gpd.read_parquet(
    "data/combined_longlat_simplified.parquet" if SIMPLIFIED_SF == True else "data/combined_longlat_clean.parquet"
))()
data = cache_utils.azure_cache_decorator(ttl=3600)(pd.read_pickle)("data/cleaned_data.pkl")
print("Main DataFrame columns:", data.index.names if data.index.nlevels > 1 else data.columns)
print("GeoDataFrame columns:", combined_longlat_clean.columns)

# Ensure categorical types for filters:
for col in ["STEM/BHASE","Academic Year","Province or Territory","ISCED Level of Education","Credential Type","Institution","CMA/CSD","DGUID"]:
    data[col] = data[col].astype('category')
data['Value'] = data['Value'].astype('float32')
data = data.set_index(["STEM/BHASE","Academic Year","Province or Territory","ISCED Level of Education",
                       "Credential Type","Institution","CMA/CSD","DGUID"]).sort_index()

# Set up filter optimizer with loaded data
data_utils.filter_optimizer = data_utils.FilterOptimizer(data)
# Compute full filter option lists for initial layout
stem_bhase_options_full = [{'label': v, 'value': v} for v in sorted(data.index.get_level_values('STEM/BHASE').unique())]
year_options_full = [{'label': v, 'value': v} for v in sorted(data.index.get_level_values('Academic Year').unique())]
prov_options_full = [{'label': v, 'value': v} for v in sorted(data.index.get_level_values('Province or Territory').unique())]
cma_options_full = [{'label': v, 'value': v} for v in sorted(data.index.get_level_values('CMA/CSD').unique())]
isced_options_full = [{'label': v, 'value': v} for v in sorted(data.index.get_level_values('ISCED Level of Education').unique())]
credential_options_full = [{'label': v, 'value': v} for v in sorted(data.index.get_level_values('Credential Type').unique())]
institution_options_full = [{'label': v, 'value': v} for v in sorted(data.index.get_level_values('Institution').unique())]
# Save options in data_utils (for use in reset_filters callback)
data_utils.stem_bhase_options_full = stem_bhase_options_full
data_utils.year_options_full = year_options_full
data_utils.prov_options_full = prov_options_full
data_utils.cma_options_full = cma_options_full
data_utils.isced_options_full = isced_options_full
data_utils.credential_options_full = credential_options_full
data_utils.institution_options_full = institution_options_full

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

import callbacks

if __name__ == '__main__':
    app.run_server(debug=True)
