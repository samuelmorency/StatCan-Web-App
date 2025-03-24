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

# @app.callback(
#     Output({'type': 'store', 'item': MATCH}, 'data'),
#     Input({'type': 'graph', 'item': MATCH}, 'clickData'),
#     Input('clear-selection', 'n_clicks'),
#     State({'type': 'store', 'item': MATCH}, 'data'),
#     State({'type': 'graph', 'item': MATCH}, 'figure'),
#     prevent_initial_call=True
# )
# def update_selection(clickData, n_clicks, stored_value, figure):
#     """
#     Manages selection state for all chart visualizations using pattern matching.
    
#     This callback processes clicks on any chart and updates the corresponding selection
#     state. It uses Dash's pattern matching to handle all charts with a single callback,
#     making the selection system maintainable and consistent. The pattern matching is
#     based on the 'item' key which identifies the dimension (e.g., 'isced', 'province').
    
#     Triggers:
#         - Clicking on any bar in any chart visualization
#         - Clicking the "Clear Selection" button
    
#     Parameters:
#         clickData (dict or None): Data from chart click event containing the clicked value.
#         n_clicks (int): Number of times "Clear Selection" button has been clicked.
#         stored_value (str or None): Currently stored selected value.
#         figure (dict): The current figure object to determine bar orientation.
    
#     Returns:
#         str or None: The new selected value, or None to clear selection.
    
#     Selection Logic:
#         1. If "Clear Selection" button is clicked, returns None for all stores.
#         2. For chart clicks:
#            - Extracts dimension type from pattern-matched component ID
#            - Determines if chart is horizontal or vertical
#            - Gets clicked value based on orientation (x or y coordinate)
#            - If clicked value matches stored value, toggles off (returns None)
#            - Otherwise sets the new clicked value as selection
    
#     Pattern Matching:
#         Uses {'type': 'store', 'item': MATCH} to match with corresponding store.
#         The 'item' key connects the graph, store, and data dimension (e.g., 'isced').
    
#     This system creates a toggle behavior where:
#     - First click on an item selects it
#     - Second click on same item deselects it
#     - Click on different item changes selection to that item
#     - "Clear Selection" button deselects everything
    
#     Once a value is stored in the store component, it triggers the update_visualizations
#     callback which applies cross-filtering based on the selection.
#     """
#     ctx = callback_context
#     if not ctx.triggered:
#         raise PreventUpdate
        
#     triggered_prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
#     if triggered_prop_id == 'clear-selection':
#         return None
    
#     if clickData and 'points' in clickData:
#         try:
#             pattern_dict = json.loads(triggered_prop_id.replace("'", "\""))
#             item_type = pattern_dict.get('item')
            
#             orientation = 'v'
#             if figure and 'data' in figure and figure['data'][0].get('orientation') == 'h':
#                 orientation = 'h'
            
#             if orientation == 'h':
#                 clicked_value = clickData['points'][0]['y']
#             else:
#                 clicked_value = clickData['points'][0]['x']
                
#             return None if stored_value == clicked_value else clicked_value
#         except Exception as e:
#             cache_utils.logger.error(f"Error parsing pattern ID: {e}")
#             return stored_value
    
#     return stored_value

# @app.callback(
#     Output('cma-geojson', 'data'),
#     Output({'type': 'graph', 'item': 'isced'}, 'figure'),
#     Output({'type': 'graph', 'item': 'province'}, 'figure'),
#     Output({'type': 'graph', 'item': 'cma'}, 'figure'),
#     Output({'type': 'graph', 'item': 'credential'}, 'figure'),
#     Output({'type': 'graph', 'item': 'institution'}, 'figure'),
#     Output('map', 'viewport'),
#     Input('stem-bhase-filter', 'value'),
#     Input('year-filter', 'value'),
#     Input('prov-filter', 'value'),
#     Input('isced-filter', 'value'),
#     Input('credential-filter', 'value'),
#     Input('institution-filter', 'value'),
#     Input('cma-filter', 'value'),
#     Input({'type': 'store', 'item': 'isced'}, 'data'),
#     Input({'type': 'store', 'item': 'province'}, 'data'),
#     Input('selected-feature', 'data'),
#     Input({'type': 'store', 'item': 'credential'}, 'data'),
#     Input({'type': 'store', 'item': 'institution'}, 'data'),
#     Input({'type': 'store', 'item': 'cma'}, 'data'),
#     State('map', 'viewport')
# )
# def update_visualizations(*args):
#     """
#     The central cross-filtering callback that coordinates all visualizations.
    
#     This callback is the heart of the application's interactive cross-filtering system.
#     It responds to both filter changes and visualization selections to update all
#     visualizations simultaneously, ensuring they remain synchronized and reflect the
#     current filtered and selected state.
    
#     Triggers:
#         - Changes to any filter (STEM/BHASE, Year, Province, etc.)
#         - Selection of any element (map feature, chart bar)
#         - Clearing of selection
#         - Reset of filters
    
#     Processing Flow:
#         1. Identifies which input triggered the callback
#         2. Processes data through preprocess_data() with current filters
#         3. If any selection exists (ISCED, Province, etc.), applies cross-filtering:
#            - Creates a boolean mask for each active selection
#            - Combines masks with logical AND operations
#            - Re-aggregates data based on the combined mask
#         4. Builds map visualization:
#            - Merges filtered data with geographic features
#            - Creates GeoJSON with dynamic styling based on graduate counts
#            - Highlights selected features
#         5. Creates chart visualizations:
#            - Generates horizontal bar charts for each dimension
#            - Applies consistent styling and highlighting
#         6. Updates map viewport if necessary:
#            - Centers on selection if a feature was clicked
#            - Shows all visible features after filter changes
    
#     Outputs:
#         - GeoJSON data for the map
#         - Figure objects for each chart (ISCED, Province, CMA, Credential, Institution)
#         - Map viewport settings
    
#     Performance Considerations:
#         - Monitors cache usage during major updates
#         - Only updates viewport when necessary
#         - Uses efficient vectorized operations throughout
#         - Uses a try-except pattern to ensure graceful degradation
#         - Returns cached empty responses for error states
    
#     Cross-Filtering Mechanism:
#         When a user clicks any visualization element, this callback:
#         1. Receives the selection via store components
#         2. Applies the selection as additional filters
#         3. Updates all visualizations to reflect the filtered view
#         4. Highlights the selected element across all applicable visualizations
    
#     This creates a unified visual query system where selections in any component
#     affect all other components, enabling powerful exploratory analysis.
#     """
#     try:
#         current_viewport = args[-1]
#         (stem_bhase, years, provs, isced, credentials, institutions, cma_filter,
#          selected_isced, selected_province, selected_feature, 
#          selected_credential, selected_institution, selected_cma) = args[:-1]
        
#         ctx = callback_context
#         if not ctx.triggered:
#             raise PreventUpdate
            
#         triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
#         if triggered_id.startswith('{'):
#             try:
#                 pattern_dict = json.loads(triggered_id.replace("'", "\""))
#                 if pattern_dict.get('type') == 'store':
#                     triggered_id = f"selected-{pattern_dict.get('item')}"
#             except:
#                 pass

#         # Process data with optimized function
#         filtered_data, cma_grads, isced_grads, province_grads, credential_grads, institution_grads = data_utils.preprocess_data(
#             tuple(stem_bhase or []),
#             tuple(years or []),
#             tuple(provs or []),
#             tuple(isced or []),
#             tuple(credentials or []),
#             tuple(institutions or []),
#             tuple(cma_filter or [])  # Add CMA filter to preprocess_data
#         )
        
#         # Apply cross-filtering with vectorized operations
#         if any([selected_isced, selected_province, selected_feature, selected_credential, selected_institution, selected_cma]):
#             mask = pd.Series(True, index=filtered_data.index)
#             if selected_isced:
#                 mask &= filtered_data['ISCED Level of Education'] == selected_isced
#             if selected_province:
#                 mask &= filtered_data['Province or Territory'] == selected_province
#             if selected_feature:
#                 mask &= filtered_data['DGUID'] == selected_feature
#             if selected_credential:
#                 mask &= filtered_data['Credential Type'] == selected_credential
#             if selected_institution:
#                 mask &= filtered_data['Institution'] == selected_institution
#             if selected_cma:
#                 mask &= filtered_data['CMA/CSD'] == selected_cma

#             filtered_data = filtered_data[mask]
#             if filtered_data.empty:
#                 return create_empty_response()

#             # Aggregations with observed=True
#             cma_grads = filtered_data.groupby(["CMA/CSD", "DGUID"], observed=True)['Value'].sum().reset_index(name='graduates')
#             isced_grads = filtered_data.groupby("ISCED Level of Education", observed=True)['Value'].sum().reset_index(name='graduates')
#             province_grads = filtered_data.groupby("Province or Territory", observed=True)['Value'].sum().reset_index(name='graduates')
#             # Add aggregations for new charts
#             #cma_aggregation = filtered_data.groupby("CMA/CSD", observed=True)['Value'].sum().reset_index(name='graduates')
#             credential_grads = filtered_data.groupby("Credential Type", observed=True)['Value'].sum().reset_index(name='graduates')
#             institution_grads = filtered_data.groupby("Institution", observed=True)['Value'].sum().reset_index(name='graduates')

#         # Prepare map data efficiently
#         cma_data = combined_longlat_clean.merge(cma_grads, on='DGUID', how='left')
#         cma_data['graduates'] = cma_data['graduates'].fillna(0)
#         cma_data = cma_data[cma_data['graduates'] > 0]
        
#         if cma_data.empty:
#             return create_empty_response()
        
#         should_update_viewport = triggered_id in [
#             'stem-bhase-filter', 'year-filter', 'prov-filter',
#             'isced-filter', 'credential-filter', 'institution-filter',
#             'selected-feature', 'clear-selection', 'reset-filters'
#         ]
        
#         if should_update_viewport:
#             bounds = cma_data.total_bounds
#             lat_padding = (bounds[3] - bounds[1]) * 0.1
#             lon_padding = (bounds[2] - bounds[0]) * 0.1
#             map_bounds = [
#                 [bounds[1] - lat_padding, bounds[0] - lon_padding],
#                 [bounds[3] + lat_padding, bounds[2] + lon_padding]
#             ]
#             viewport_output = dict(
#                 bounds=map_bounds,
#                 transition=dict(duration=1000)
#             )
#         else:
#             viewport_output = dash.no_update
            
#         # Create GeoJSON efficiently - updated to use built-in color scale
#         max_graduates = cma_data['graduates'].max()
#         min_graduates = cma_data['graduates'].min()
        
#         if max_graduates > min_graduates:
#             normalized_values = (cma_data['graduates'] - min_graduates) / (max_graduates - min_graduates)
#             colors = px.colors.sample_colorscale(COLOUR_SCALE, normalized_values)
#         else:
#             colors = [COLOUR_SCALE[-1]] * len(cma_data)
        
#         features = [
#             {
#                 'type': 'Feature',
#                 'geometry': row.geometry.__geo_interface__,
#                 'properties': {
#                     'graduates': int(row['graduates']),
#                     'DGUID': str(row['DGUID']),
#                     'CMA/CSD': row['CMA/CSD'],
#                     'style': {
#                         'fillColor': color if row['graduates'] > 0 else 'lightgray',
#                         'color': bc.IIC_BLACK if row['DGUID'] == selected_feature else bc.GREY,
#                         'weight': 2 if row['DGUID'] == selected_feature else 0.75,
#                         'fillOpacity': 0.8
#                     },
#                     'tooltip': f"<div style='font-family: Open Sans, sans-serif; font-weight: 600;'>{row['CMA/CSD']}: {int(row['graduates']):,}</div>"
#                 }
#             }
#             for (_, row), color in zip(cma_data.iterrows(), colors)
#         ]
        
#         geojson_data = {'type': 'FeatureCollection', 'features': features}
        
#         # Create charts with selections (simplified without chart type parameter)
#         fig_isced = data_utils.create_chart(
#             isced_grads, 
#             'ISCED Level of Education', 
#             'graduates',
#             'ISCED Level of Education', 
#             selected_isced
#         )
        
#         fig_province = data_utils.create_chart(
#             province_grads, 
#             'Province or Territory', 
#             'graduates',
#             'Province/Territory', 
#             selected_province
#         )

#         # Create new charts
#         fig_cma = data_utils.create_chart(
#             cma_grads,
#             'CMA/CSD',
#             'graduates',
#             'Census Metropolitan Area',
#             selected_feature
#         )

#         fig_credential = data_utils.create_chart(
#             credential_grads,
#             'Credential Type',
#             'graduates',
#             'Credential Type',
#             selected_credential
#         )

#         fig_institution = data_utils.create_chart(
#             institution_grads,
#             'Institution',
#             'graduates',
#             'Institution',
#             selected_institution
#         )
        
#         # Monitor cache at the start of major updates
#         cache_utils.monitor_cache_usage()
        
#         return (
#             geojson_data,
#             fig_isced,
#             fig_province,
#             fig_cma,
#             fig_credential,
#             fig_institution,
#             viewport_output
#         )
        
#     except Exception as e:
#         cache_utils.logger.error(f"Error in update_visualizations: {str(e)}")
#         return create_empty_response()

# @app.callback(
#     Output('stem-bhase-filter', 'value'),
#     Output('year-filter', 'value'),
#     Output('prov-filter', 'value'),
#     Output('isced-filter', 'value'),
#     Output('credential-filter', 'value'),
#     Output('institution-filter', 'value'),
#     Output('cma-filter', 'value'),  # Add CMA filter output
#     Input('reset-filters', 'n_clicks'),
#     prevent_initial_call=True
# )
# def reset_filters(n_clicks):
#     """
#     Resets all data filters back to their default values. This clears any user-applied
#     restrictions, ensuring that the dataset and visualizations return to their initial
#     state.

#     Args:
#         n_clicks (int): The number of times the 'Reset Filters' button has been clicked.

#     Returns:
#         tuple: A tuple containing the default values for each filter component.
#     """
#     if not n_clicks:
#         raise PreventUpdate
        
#     return (
#         [option['value'] for option in stem_bhase_options_full],
#         [option['value'] for option in year_options_full],
#         [], [], [], [], []  # Added empty list for CMA filter
#     )

# @app.callback(
#     Output('stem-bhase-filter', 'options'),
#     Output('year-filter', 'options'),
#     Output('prov-filter', 'options'),
#     Output('cma-filter', 'options'),
#     Output('isced-filter', 'options'),
#     Output('credential-filter', 'options'),
#     Output('institution-filter', 'options'),
#     Input('stem-bhase-filter', 'value'),
#     Input('year-filter', 'value'),
#     Input('prov-filter', 'value'),
#     Input('isced-filter', 'value'),
#     Input('credential-filter', 'value'),
#     Input('institution-filter', 'value'),
#     Input('cma-filter', 'value'),
#     # Chart selection stores
#     Input({'type': 'store', 'item': 'isced'}, 'data'),
#     Input({'type': 'store', 'item': 'province'}, 'data'),
#     Input('selected-feature', 'data'),
#     Input({'type': 'store', 'item': 'credential'}, 'data'),
#     Input({'type': 'store', 'item': 'institution'}, 'data'),
#     Input({'type': 'store', 'item': 'cma'}, 'data'),
#     Input('clear-selection', 'n_clicks')
# )
# def update_filter_options(stem_bhase, years, provs, isced, credentials, institutions, cmas,
#                          selected_isced, selected_province, selected_feature,
#                          selected_credential, selected_institution, selected_cma,
#                          clear_clicks):
#     """
#     Updates filter options based on both dropdown selections and chart selections.
#     Works with MultiIndex data structure.
#     """
#     ctx = callback_context
#     if not ctx.triggered:
#         raise PreventUpdate
    
#     # Start with dropdown selections
#     updated_filters = {
#         'STEM/BHASE': set(stem_bhase or []),
#         'Academic Year': set(years or []),
#         'Province or Territory': set(provs or []),
#         'CMA/CSD': set(cmas or []),
#         'ISCED Level of Education': set(isced or []),
#         'Credential Type': set(credentials or []),
#         'Institution': set(institutions or [])
#     }
    
#     # Add chart selections if they're not already in the corresponding filter
#     if selected_isced and selected_isced not in updated_filters['ISCED Level of Education']:
#         updated_filters['ISCED Level of Education'].add(selected_isced)
    
#     if selected_province and selected_province not in updated_filters['Province or Territory']:
#         updated_filters['Province or Territory'].add(selected_province)
    
#     if selected_credential and selected_credential not in updated_filters['Credential Type']:
#         updated_filters['Credential Type'].add(selected_credential)
    
#     if selected_institution and selected_institution not in updated_filters['Institution']:
#         updated_filters['Institution'].add(selected_institution)
    
#     if selected_cma and selected_cma not in updated_filters['CMA/CSD']:
#         updated_filters['CMA/CSD'].add(selected_cma)
    
#     # Handle map feature selection
#     if selected_feature:
#         try:
#             feature_cma = combined_longlat_clean[combined_longlat_clean['DGUID'] == selected_feature]['NAME'].iloc[0]
#             if feature_cma and feature_cma not in updated_filters['CMA/CSD']:
#                 updated_filters['CMA/CSD'].add(feature_cma)
#         except (IndexError, KeyError):
#             pass  # Silently handle the case where feature_cma can't be found
    
#     # Generate filter options by excluding the target dimension from filters
#     stem_options = data_utils.filter_options(data, 'STEM/BHASE', 
#                                             {k: v for k, v in updated_filters.items() if k != 'STEM/BHASE'})
    
#     year_options = data_utils.filter_options(data, 'Academic Year', 
#                                            {k: v for k, v in updated_filters.items() if k != 'Academic Year'})
    
#     prov_options = data_utils.filter_options(data, 'Province or Territory', 
#                                            {k: v for k, v in updated_filters.items() if k != 'Province or Territory'})
    
#     cma_options = data_utils.filter_options(data, 'CMA/CSD', 
#                                           {k: v for k, v in updated_filters.items() if k != 'CMA/CSD'})
    
#     isced_options = data_utils.filter_options(data, 'ISCED Level of Education', 
#                                             {k: v for k, v in updated_filters.items() if k != 'ISCED Level of Education'})
    
#     cred_options = data_utils.filter_options(data, 'Credential Type', 
#                                           {k: v for k, v in updated_filters.items() if k != 'Credential Type'})
    
#     inst_options = data_utils.filter_options(data, 'Institution', 
#                                           {k: v for k, v in updated_filters.items() if k != 'Institution'})
    
#     return (
#         stem_options,
#         year_options,
#         prov_options,
#         cma_options,
#         isced_options,
#         cred_options,
#         inst_options
#     )

# @app.callback(
#     Output("download-data", "data"),
#     Input("download-button", "n_clicks"),
#     State("pivot-table", "data"),
#     State("pivot-table", "cols"),
#     State("pivot-table", "rows"),
#     State("pivot-table", "vals"),
#     prevent_initial_call=True,
# )
# def download_pivot_data(n_clicks, data, cols, rows, vals):
#     """
#     Creates a downloadable CSV file from the pivot table's current configuration.
    
#     Args:
#         n_clicks (int): Number of times download button clicked
#         data (list): The raw data from the pivot table
#         cols (list): Column headers configured in the pivot table
#         rows (list): Row headers configured in the pivot table
#         vals (list): Value fields configured in the pivot table
        
#     Returns:
#         dict: Download specification for Dash
#     """
#     if not n_clicks or not data:
#         raise PreventUpdate

#     try:
#         # Create a pandas DataFrame from the pivot table data
#         df = pd.DataFrame(data)
        
#         # Get unique values for rows and columns
#         row_values = []
#         for row in rows:
#             if row in df.columns:
#                 row_values.append(sorted(df[row].unique()))
                
#         col_values = []
#         for col in cols:
#             if col in df.columns:
#                 col_values.append(sorted(df[col].unique()))
                
#         # Create MultiIndex for rows and columns
#         if row_values:
#             row_index = pd.MultiIndex.from_product(row_values, names=rows)
#         else:
#             row_index = pd.Index([])
            
#         if col_values:
#             col_index = pd.MultiIndex.from_product(col_values, names=cols)
#         else:
#             col_index = pd.Index([])
            
#         # Create pivot table
#         pivot_df = df.pivot_table(
#             index=rows if rows else None,
#             columns=cols if cols else None,
#             values=vals,
#             aggfunc='sum',
#             fill_value=0
#         )
        
#         # Convert to string buffer
#         string_buffer = io.StringIO()
#         pivot_df.to_csv(string_buffer)
        
#         return dict(
#             content=string_buffer.getvalue(),
#             filename=f"graduates_pivot_{time.strftime('%Y%m%d_%H%M%S')}.csv",
#             type="text/csv",
#             base64=False
#         )
        
#     except Exception as e:
#         cache_utils.logger.error(f"Error in download_pivot_data: {str(e)}")
#         raise PreventUpdate

# @app.callback(
#     Output("horizontal-collapse", "is_open"),
#     [Input("horizontal-collapse-button", "n_clicks")],
#     [State("horizontal-collapse", "is_open")],
# )
# def toggle_collapse(n, is_open):
#     if n:
#         return not is_open
#     return is_open

# # we use a callback to toggle the collapse on small screens
# def toggle_navbar_collapse(n, is_open):
#     if n:
#         return not is_open
#     return is_open


# # the same function (toggle_navbar_collapse) is used in all three callbacks
# app.callback(
#     Output("navbar-collapse", "is_open"),
#     [Input("navbar-toggler", "n_clicks")],
#     [State("navbar-collapse", "is_open")],
# )(toggle_navbar_collapse)

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

# def load_user_guide():
#     """Load user guide markdown"""
#     user_guide_path = Path("user_guide.md")
#     if (user_guide_path.exists()):
#         with open(user_guide_path, "r", encoding="utf-8") as f:
#             content = f.read()
#             return dcc.Markdown(content)
#     return "User guide not available"

# @app.callback(
#     Output("user-guide-modal", "is_open"),
#     Output("user-guide-content", "children"),
#     [
#         Input("open-guide-button", "n_clicks"),
#         Input("close-guide-button", "n_clicks")
#     ],
#     State("user-guide-modal", "is_open"),
# )
# def toggle_user_guide(open_clicks, close_clicks, is_open):
#     """Toggle user guide modal and load content"""
#     if not any(clicks for clicks in [open_clicks, close_clicks]):
#         raise PreventUpdate
        
#     if open_clicks or close_clicks:
#         if not is_open:
#             # Only load content when opening
#             return not is_open, html.Div(
#                 #dangerously_allow_html=True,
#                 children=load_user_guide()
#             )
#         return not is_open, dash.no_update
        
#     return is_open, dash.no_update

import callbacks

if __name__ == '__main__':
    app.run_server(debug=True)
