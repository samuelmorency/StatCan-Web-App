import tempfile
import os
from dash import Dash, html, dcc, callback, Output, Input, State, callback_context, dash_table, Patch, no_update
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
from dotenv import load_dotenv

load_dotenv()
#mapbox_api_token = os.getenv("MAPBOX_ACCESS_TOKEN")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureCache:
    def __init__(self):
        self._CACHE_DIR = os.path.join(tempfile.gettempdir(), 'dash_cache')
        self._DEFAULT_SIZE = 1024e6  # 1GB
        self._FALLBACK_SIZE = 256e6
        self._TTL = 3600
        self._cache = None
        self._memory_cache = {}
        self._last_access = {}
        self._MAX_MEMORY_ITEMS = 1000
        self.initialize_cache()

    def _prune_memory_cache(self):
        if len(self._memory_cache) > self._MAX_MEMORY_ITEMS:
            sorted_items = sorted(self._last_access.items(), key=lambda x: x[1])
            to_remove = len(self._memory_cache) - self._MAX_MEMORY_ITEMS
            for key, _ in sorted_items[:to_remove]:
                self._memory_cache.pop(key, None)
                self._last_access.pop(key, None)

    def get_cache(self):
        return self._cache

    def get_cache_value(self, key):
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
        self._memory_cache.clear()
        self._last_access.clear()
        if self._cache:
            try:
                self._cache.clear()
                logger.info("Cache cleared successfully")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")

# Initialize Azure-specific cache
azure_cache = AzureCache()
cache = azure_cache.get_cache()

# Register cleanup
atexit.register(azure_cache.clear_cache)


def azure_cache_decorator(ttl=3600):
    """
    Decorator for caching function results with configurable TTL.

    Args:
        ttl (int): Time-to-live in seconds for cached results

    Returns:
        function: Decorated function with caching capability
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



# Initialize the app
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        # Add Google Fonts link for Open Sans SemiBold
        'https://fonts.googleapis.com/css2?family=Open+Sans:wght@600&display=swap'
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
    Maintains the current state of the map, including viewport bounds, zoom level,
    selected and hovered features, color scale, timing of last updates, and whether
    the viewport is locked from updates. This class provides methods to determine
    if the viewport should be updated based on timing constraints and user actions,
    as well as methods to update the stored state as user interactions occur.
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

@azure_cache_decorator(ttl=600)  # 10 minutes - calculated values
def calculate_optimal_viewport(bounds, padding_factor=0.1):
    """
    Computes an optimal viewport given geographic bounds. If the provided bounds
    are invalid or zero-sized, returns default bounds covering Canada.

    Args:
        bounds (tuple): A tuple (min_lat, min_lon, max_lat, max_lon) specifying
                        the geographic area to fit in the viewport.
        padding_factor (float): A factor to apply as padding around the given bounds
                                to avoid overly tight fitting.

    Returns:
        dict: A dictionary containing:
              - 'bounds': A 2D list of adjusted bounding coordinates.
              - 'center': A 2-element list representing the center [lat, lon].
              - 'zoom': An integer representing the computed zoom level.
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

@azure_cache_decorator(ttl=600)  # 10 minutes - calculated values
def calculate_zoom_level(min_lat, min_lon, max_lat, max_lon):
    """
    Determines a suitable zoom level based on the size of a geographic bounding box.
    The zoom level is higher (more zoomed in) for smaller areas and lower (more zoomed out)
    for larger areas. The logic considers the maximum dimension of the latitude or
    longitude span and maps it to a discrete zoom level.

    Args:
        min_lat (float): Minimum latitude of the bounding box.
        min_lon (float): Minimum longitude of the bounding box.
        max_lat (float): Maximum latitude of the bounding box.
        max_lon (float): Maximum longitude of the bounding box.

    Returns:
        int: A zoom level. Larger values indicate a closer view.
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

@azure_cache_decorator(ttl=300)  # 5 minutes - color calculations
def create_color_scale(max_val, min_val, n_colors=9):
    """
    Creates a dictionary mapping a range of values to a corresponding set of colors
    arranged in a sequential color scale. If the range is zero (max_val == min_val),
    a single color is returned. The colors are based on a 'Reds' sequential scheme.

    Args:
        max_val (float): The maximum value of the data range.
        min_val (float): The minimum value of the data range.
        n_colors (int): The number of colors to use in the scale.

    Returns:
        dict: A dictionary where keys are numeric breakpoints and values are color strings.
              Each key-value pair represents a segment of the color scale.
    """
    if max_val == min_val:
        return {min_val: cl.scales[str(n_colors)]['seq']['Reds'][-1]}
    breaks = np.linspace(min_val, max_val, n_colors)
    colors = cl.scales[str(n_colors)]['seq']['Reds']
    return dict(zip(breaks, colors))

class CallbackContextManager:
    """
    A helper class to interpret Dash callback contexts. It simplifies the identification
    of which input triggered the callback and whether the callback was triggered at all.
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
    Loads geospatial data for provinces and combined CMA/CA polygons from parquet files.
    This includes coordinates and identifiers necessary for geographic rendering and
    linking data to specific regions. The result is cached to avoid repeated disk reads.

    Returns:
        tuple: A tuple containing:
               - A GeoDataFrame of provinces with cleaned longitude/latitude.
               - A GeoDataFrame of combined CMA/CA regions with cleaned coordinates.
    """
    province_longlat_clean = gpd.read_parquet("data/province_longlat_clean.parquet")
    combined_longlat_clean = gpd.read_parquet("data/combined_longlat_clean.parquet")
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
    data = pd.read_pickle("data/cleaned_data.pkl")
    
    categorical_cols = ["STEM/BHASE", "year", "Province_Territory", "ISCED_level_of_education", "Credential_Type", "Institution", "CMA_CA", "DGUID"]
    for col in categorical_cols:
        data[col] = data[col].astype('category')
    data['value'] = data['value'].astype('float32')

    # Set a multi-index for direct indexing
    data = data.set_index(["STEM/BHASE", "year", "Province_Territory", "ISCED_level_of_education", "Credential_Type", "Institution", "CMA_CA", "DGUID"]).sort_index()

    return data

# Load initial data
province_longlat_clean, combined_longlat_clean = load_spatial_data()
data = load_and_process_educational_data()

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

# Add optimized filtering class
class FilterOptimizer:
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

def filter_data(data, filters):
    """
    Filters the multi-indexed DataFrame using direct indexing. The filters dictionary
    contains sets of allowed values for each dimension. If a set is empty, all values
    for that level are included. This approach uses direct .loc indexing on a multi-index.

    Args:
        data (pandas.DataFrame): The original dataset with a multi-level index.
        filters (dict): A dictionary where keys are column names (in the index) and
                        values are sets of acceptable values.

    Returns:
        pandas.DataFrame: The filtered DataFrame containing only rows that meet
                          all specified conditions.
    """
    mask = pd.Series(True, index=data.index)
    
    for col, values in filters.items():
        if values:
            mask &= data.index.get_level_values(col).isin(values)
    
    return data[mask]

@azure_cache_decorator(ttl=300)  # 5 minutes - results depend on user filters
@monitor_performance
def preprocess_data(selected_stem_bhase, selected_years, selected_provs, selected_isced, 
                   selected_credentials, selected_institutions, selected_cmas):
    """Optimized data preprocessing with caching and vectorized operations

    Args:
        selected_stem_bhase (_type_): _description_
        selected_years (_type_): _description_
        selected_provs (_type_): _description_
        selected_isced (_type_): _description_
        selected_credentials (_type_): _description_
        selected_institutions (_type_): _description_
        selected_cmas (_type_): _description_

    Returns:
        _type_: _description_
    """    """"""
    filters = {
        'STEM/BHASE': set(selected_stem_bhase),
        'year': set(selected_years),
        'Province_Territory': set(selected_provs),
        'ISCED_level_of_education': set(selected_isced),
        'Credential_Type': set(selected_credentials),
        'Institution': set(selected_institutions),
        'CMA_CA': set(selected_cmas),
        'DGUID': set()
    }

    # Use optimized filtering
    filtered_data = filter_optimizer.filter_data(filters)
    if filtered_data.empty:
        return filtered_data.reset_index(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Use vectorized operations instead of parallel processing
    aggregations = {
        'cma': filtered_data.groupby(["CMA_CA", "DGUID"], observed=True)['value'].sum(),
        'isced': filtered_data.groupby("ISCED_level_of_education", observed=True)['value'].sum(),
        'province': filtered_data.groupby("Province_Territory", observed=True)['value'].sum(),
        'credential': filtered_data.groupby("Credential_Type", observed=True)['value'].sum(),
        'institution': filtered_data.groupby("Institution", observed=True)['value'].sum()
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
    Creates a GeoJSON feature dictionary for a single geographic unit (e.g., a CMA/CA).
    Each feature includes property fields for graduates count, DGUID, and CMA/CA name.
    The style properties are determined by the number of graduates and whether the
    feature is currently selected. If the data range is valid, it uses a normalized
    index into the color scale; otherwise, it falls back to a default color.

    Args:
        row (pandas.Series): A row from a GeoDataFrame representing one geographic unit.
        colorscale (list): A list of color strings for representing graduate counts.
        max_graduates (int): The maximum graduates count in the dataset to map.
        min_graduates (int): The minimum graduates count in the dataset to map.
        selected_feature (str or None): The DGUID of the currently selected feature, if any.

    Returns:
        dict: A dictionary representing a GeoJSON feature with 'properties' and 'style'
              attributes suitable for rendering on a Leaflet map via Dash Leaflet.
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
        'CMA_CA': row['CMA_CA'],
        'style': {
            'fillColor': bc.MAIN_RED if is_selected else color,
            'color': bc.IIC_BLACK if is_selected else bc.GREY,
            'weight': 2 if is_selected else 0.5,
            'fillOpacity': 0.8
        },
        'tooltip': f"CMA/CA: {row['CMA_CA']}<br>Graduates: {int(graduates)}"
    }

@azure_cache_decorator(ttl=300)  # 5 minutes - visualization data
def create_chart(dataframe, x_column, y_column, x_label, selected_value=None):
    """
    Creates a horizontal bar chart with error handling for invalid values.

    Args:
        dataframe (pd.DataFrame): Data to plot
        x_column (str): Column name for x-axis
        y_column (str): Column name for y values
        x_label (str): Label for chart title
        selected_value (str, optional): Currently selected value to highlight

    Returns:
        dict: Plotly figure dictionary or empty dict if data invalid
    """
    if dataframe is None or dataframe.empty:
        return {}
    
    sorted_data = dataframe.sort_values(y_column, ascending=True)
    sorted_data['text'] = sorted_data[y_column].apply(lambda x: f'{int(x):,}')
    
    chart_height=500
    
    #If x_label = 'Institution' or 'CMA_CA', then chart_height=1000
    if x_label == 'Institution' or x_label == 'Census Metropolitan Area':
        chart_height= 25 * len(sorted_data.index)
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
                bordercolor=bc.IIC_BLACK
            ),
            marker=dict(
                color=[bc.LIGHT_GREY if x != selected_value else bc.MAIN_RED for x in sorted_data[x_column]] if selected_value else sorted_data[y_column],
                colorscale='Reds' if not selected_value else None
            )
        )#, layout={'height': 5000}
    )
    
    fig.update_layout(
        title=dict(
            text=f'Number of Graduates by {x_label}',
            font=dict(size=16,
                      family='Open Sans',
                      weight=600)  # Increased title size
        ),
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_title=None,
        yaxis_title=None,
        height=chart_height,  # Keeping 500 as default but can be adjusted
        margin=dict(l=5, r=50, t=50, b=5),
        clickmode='event+select',
        plot_bgcolor='#D5DADC',
        paper_bgcolor='white',
        font=dict(
            color=bc.IIC_BLACK,
            family='Open Sans',
            weight=600
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
year_options_full = [{'label': year, 'value': year} for year in sorted(data.index.get_level_values('year').unique())]
prov_options_full = [{'label': prov, 'value': prov} for prov in sorted(data.index.get_level_values('Province_Territory').unique())]
cma_options_full = [{'label': cma, 'value': cma} for cma in sorted(data.index.get_level_values('CMA_CA').unique())]
isced_options_full = [{'label': level, 'value': level} for level in sorted(data.index.get_level_values('ISCED_level_of_education').unique())]
credential_options_full = [{'label': cred, 'value': cred} for cred in sorted(data.index.get_level_values('Credential_Type').unique())]
institution_options_full = [{'label': inst, 'value': inst} for inst in sorted(data.index.get_level_values('Institution').unique())]


app.layout = html.Div([
    dcc.Store(id='client-data-store', storage_type='session'),
    dcc.Store(id='client-filters-store', storage_type='local'),
    create_layout(data, stem_bhase_options_full, year_options_full, prov_options_full, isced_options_full, credential_options_full, institution_options_full, cma_options_full)
])

def calculate_viewport_update(triggered_id, cma_data, selected_feature=None):
    """
    Determines an appropriate viewport update based on the user interaction that
    triggered the callback. If the triggered action is selecting a CMA/CA feature,
    it adjusts the viewport to zoom in on that feature. If filters changed, it
    recalculates bounds to show all visible features. If no adjustments are needed,
    it returns None.

    Args:
        triggered_id (str): The ID of the component that triggered the callback.
        cma_data (geopandas.GeoDataFrame): The spatial data with graduates information.
        selected_feature (str or None): The currently selected CMA/CA DGUID, if any.

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
    Updates the currently selected CMA/CA feature when the map is clicked. If the
    'Clear Selection' button is clicked, resets the selection. This ensures that
    the selected feature state remains synchronized with user actions on the map.

    Args:
        click_data (dict or None): Data associated with a map feature click.
        n_clicks (int): The number of times the 'Clear Selection' button was clicked.
        stored_cma (str or None): The currently stored CMA/CA DGUID.

    Returns:
        str or None: The updated selected CMA/CA DGUID or None if the selection is cleared.
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
    Output('selected-isced', 'data'),
    Input('graph-isced', 'clickData'),
    Input('clear-selection', 'n_clicks'),
    State('selected-isced', 'data'),
    State('graph-isced', 'figure'),
    prevent_initial_call=True
)
def update_selected_isced(clickData, n_clicks, stored_isced, figure):
    """
    Updates the selected ISCED level when the user interacts with the ISCED chart.
    If a slice or bar is clicked, this updates the selected ISCED level. If 'Clear
    Selection' is clicked, it resets the selection. Maintains synchronization with
    chart interactions.

    Args:
        clickData (dict or None): Data from a chart click event that includes the selected category.
        n_clicks (int): The count of 'Clear Selection' button clicks.
        stored_isced (str or None): The currently selected ISCED level.
        figure (dict): The current figure object to determine chart orientation.

    Returns:
        str or None: The updated selected ISCED level or None if the selection is cleared.
    """
    ctx_manager = CallbackContextManager(callback_context)
    if not ctx_manager.is_triggered:
        raise PreventUpdate

    return update_selected_value(
        click_data=clickData,
        n_clicks=n_clicks,
        stored_value=stored_isced,
        triggered_id=ctx_manager.triggered_id,
        clear_id='clear-selection',
        chart_id='graph-isced',
        figure=figure
    )

@app.callback(
    Output('selected-province', 'data'),
    Input('graph-province', 'clickData'),
    Input('clear-selection', 'n_clicks'),
    State('selected-province', 'data'),
    State('graph-province', 'figure'),
    prevent_initial_call=True
)
def update_selected_province(clickData, n_clicks, stored_province, figure):
    """
    Updates the selected province when the user interacts with the province chart.
    If a bar or slice is clicked, the selection updates accordingly. If 'Clear Selection'
    is clicked, the selection is reset. The function also interprets chart orientation
    to determine which axis value corresponds to the province.

    Args:
        clickData (dict or None): Data from a chart click event that includes the clicked province.
        n_clicks (int): The count of 'Clear Selection' button clicks.
        stored_province (str or None): The currently selected province.
        figure (dict): The current figure object to determine chart orientation.

    Returns:
        str or None: The updated selected province or None if the selection is cleared.
    """
    ctx_manager = CallbackContextManager(callback_context)
    if not ctx_manager.is_triggered:
        raise PreventUpdate

    return update_selected_value(
        click_data=clickData,
        n_clicks=n_clicks,
        stored_value=stored_province,
        triggered_id=ctx_manager.triggered_id,
        clear_id='clear-selection',
        chart_id='graph-province',
        figure=figure
    )

# Add new callback for credential selection
@app.callback(
    Output('selected-credential', 'data'),
    Input('graph-credential', 'clickData'),
    Input('clear-selection', 'n_clicks'),
    State('selected-credential', 'data'),
    State('graph-credential', 'figure'),
    prevent_initial_call=True
)
def update_selected_credential(clickData, n_clicks, stored_credential, figure):
    ctx_manager = CallbackContextManager(callback_context)
    if not ctx_manager.is_triggered:
        raise PreventUpdate

    return update_selected_value(
        click_data=clickData,
        n_clicks=n_clicks,
        stored_value=stored_credential,
        triggered_id=ctx_manager.triggered_id,
        clear_id='clear-selection',
        chart_id='graph-credential',
        figure=figure
    )

@app.callback(
    Output('selected-cma', 'data'),
    Input('graph-cma', 'clickData'),
    Input('clear-selection', 'n_clicks'),
    State('selected-cma', 'data'),
    State('graph-cma', 'figure'),
    prevent_initial_call=True
)
def update_selected_cma(clickData, n_clicks, stored_credential, figure):
    ctx_manager = CallbackContextManager(callback_context)
    if not ctx_manager.is_triggered:
        raise PreventUpdate

    return update_selected_value(
        click_data=clickData,
        n_clicks=n_clicks,
        stored_value=stored_credential,
        triggered_id=ctx_manager.triggered_id,
        clear_id='clear-selection',
        chart_id='graph-cma',
        figure=figure
    )

# Add new callback for institution selection
@app.callback(
    Output('selected-institution', 'data'),
    Input('graph-institution', 'clickData'),
    Input('clear-selection', 'n_clicks'),
    State('selected-institution', 'data'),
    State('graph-institution', 'figure'),
    prevent_initial_call=True
)
def update_selected_institution(clickData, n_clicks, stored_institution, figure):
    ctx_manager = CallbackContextManager(callback_context)
    if not ctx_manager.is_triggered:
        raise PreventUpdate

    return update_selected_value(
        click_data=clickData,
        n_clicks=n_clicks,
        stored_value=stored_institution,
        triggered_id=ctx_manager.triggered_id,
        clear_id='clear-selection',
        chart_id='graph-institution',
        figure=figure
    )

def update_map_style(geojson_data, colorscale, selected_feature=None):
    """
    Create a Patch object for updating map styles.
    """
    patched_geojson = Patch()
    
    if not geojson_data or 'features' not in geojson_data:
        return patched_geojson
        
    feature_dguids = [f['properties']['DGUID'] for f in geojson_data['features']]
    is_selected = [selected_feature and dguid == selected_feature for dguid in feature_dguids]
    
    for i, selected in enumerate(is_selected):
        if selected or geojson_data['features'][i]['properties']['style'].get('weight') != 0.5:
            style_updates = {
                'fillColor': 'yellow' if selected else geojson_data['features'][i]['properties']['style']['fillColor'],
                'color': 'black' if selected else 'gray',
                'weight': 2 if selected else 0.5,
            }
            patched_geojson['features'][i]['properties']['style'].update(style_updates)
    
    return patched_geojson

@app.callback(
    Output('cma-geojson', 'data', allow_duplicate=True),
    Input('cma-geojson', 'hover_feature'),
    State('cma-geojson', 'data'),
    prevent_initial_call=True
)
def update_hover_style(hover_feature, current_geojson):
    """
    Adjusts the styling of GeoJSON features on the map when a user hovers over them.
    The hovered feature is highlighted to provide visual feedback, and previously
    hovered features are reverted to their original style.

    Args:
        hover_feature (dict or None): The feature currently hovered, containing its 'id'.
        current_geojson (dict): The current GeoJSON data displayed on the map.

    Returns:
        dash_table.Patch: A Patch object representing the updated styling instructions.
    """
    if not hover_feature or not current_geojson:
        raise PreventUpdate
        
    patched_geojson = Patch()
    hover_id = hover_feature.get('id')
    
    for i, feature in enumerate(current_geojson['features']):
        feature_id = feature['properties']['DGUID']
        if feature_id == hover_id:
            patched_geojson['features'][i]['properties']['style'].update({
                'weight': 3,
                'color': bc.DARK_BLUE,
                'fillOpacity': 0.9
            })
        elif feature['properties']['style'].get('weight') == 3:
            patched_geojson['features'][i]['properties']['style'].update({
                'weight': 0.5,
                'fillOpacity': 0.8
            })
            
    return patched_geojson

@app.callback(
    Output('map', 'viewport', allow_duplicate=True),
    Input('map', 'viewport'),
    prevent_initial_call=True
)
def handle_map_movement(viewport):
    """
    Updates the global map state when the user moves the map (pans or zooms). If the
    viewport contains updated bounds, these are stored so that future updates can be
    managed based on the current state. Does not modify the viewport further, only
    updates state tracking.

    Args:
        viewport (dict): The current viewport state of the map, including bounds and zoom.

    Returns:
        no_update: Indicates no modification to outputs.
    """
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    if viewport and 'bounds' in viewport:
        map_state.update_state(bounds=viewport['bounds'])
    
    return no_update

# Add cache monitoring
def monitor_cache_usage():
    """Monitor cache size and usage"""
    if not cache:
        return
        
    try:
        total_size = sum(os.path.getsize(os.path.join(cache.directory, f))
                        for f in os.listdir(cache.directory)
                        if os.path.isfile(os.path.join(cache.directory, f)))
        
        usage_mb = total_size / 1e6
        logger.info(f"Current cache usage: {usage_mb:.2f}MB")
        
        if usage_mb > (cache.size_limit / 1e6) * 0.9:  # 90% threshold
            logger.warning("Cache usage approaching limit")
            cache.expire()
    except Exception as e:
        logger.error(f"Error monitoring cache: {e}")

@app.callback(
    Output('cma-geojson', 'data'),
    Output('graph-isced', 'figure'),
    Output('graph-province', 'figure'),
    Output('graph-cma', 'figure'),
    Output('graph-credential', 'figure'),
    Output('graph-institution', 'figure'),
    Output('map', 'viewport'),
    Input('stem-bhase-filter', 'value'),
    Input('year-filter', 'value'),
    Input('prov-filter', 'value'),
    Input('isced-filter', 'value'),
    Input('credential-filter', 'value'),
    Input('institution-filter', 'value'),
    Input('cma-filter', 'value'),  # Add CMA filter input
    Input('selected-isced', 'data'),
    Input('selected-province', 'data'),
    Input('selected-feature', 'data'),
    Input('selected-credential', 'data'),
    Input('selected-institution', 'data'),
    Input('selected-cma', 'data'),
    State('map', 'viewport')
)
def update_visualizations(*args):
    """
    The primary callback that responds to user input changes from filters, charts,
    and map selections. It:
    - Filters and preprocesses the data based on selected criteria.
    - Applies cross-filtering based on the currently selected ISCED level, province,
      and CMA/CA feature.
    - Updates the GeoJSON map layer, ISCED chart, province chart, and the CMA/CA table.
    - Adjusts the map viewport if needed.

    If no data matches the filters or an error occurs, it returns empty structures.

    Args:
        *args: A list of inputs including filter values, chart types, selected ISCED,
               selected province, selected CMA, and the current map viewport.

    Returns:
        tuple: A tuple of updated components for the map, charts, table, and viewport.
               - GeoJSON data for the map
               - Figure for ISCED chart
               - Figure for province chart
               - Updated viewport settings
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
                mask &= filtered_data['ISCED_level_of_education'] == selected_isced
            if selected_province:
                mask &= filtered_data['Province_Territory'] == selected_province
            if selected_feature:
                mask &= filtered_data['DGUID'] == selected_feature
            if selected_credential:
                mask &= filtered_data['Credential_Type'] == selected_credential
            if selected_institution:
                mask &= filtered_data['Institution'] == selected_institution
            if selected_cma:
                mask &= filtered_data['CMA_CA'] == selected_cma

            filtered_data = filtered_data[mask]
            if filtered_data.empty:
                return create_empty_response()

            # Aggregations with observed=True
            cma_grads = filtered_data.groupby(["CMA_CA", "DGUID"], observed=True)['value'].sum().reset_index(name='graduates')
            isced_grads = filtered_data.groupby("ISCED_level_of_education", observed=True)['value'].sum().reset_index(name='graduates')
            province_grads = filtered_data.groupby("Province_Territory", observed=True)['value'].sum().reset_index(name='graduates')
            # Add aggregations for new charts
            #cma_aggregation = filtered_data.groupby("CMA_CA", observed=True)['value'].sum().reset_index(name='graduates')
            credential_grads = filtered_data.groupby("Credential_Type", observed=True)['value'].sum().reset_index(name='graduates')
            institution_grads = filtered_data.groupby("Institution", observed=True)['value'].sum().reset_index(name='graduates')

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
            colors = px.colors.sample_colorscale(px.colors.sequential.Reds, normalized_values)
        else:
            colors = [px.colors.sequential.Reds[-1]] * len(cma_data)
        
        features = [
            {
                'type': 'Feature',
                'geometry': row.geometry.__geo_interface__,
                'properties': {
                    'graduates': int(row['graduates']),
                    'DGUID': str(row['DGUID']),
                    'CMA_CA': row['CMA_CA'],
                    'style': {
                        'fillColor': color if row['graduates'] > 0 else 'lightgray',
                        'color': bc.IIC_BLACK if row['DGUID'] == selected_feature else bc.GREY,
                        'weight': 2 if row['DGUID'] == selected_feature else 0.5,
                        'fillOpacity': 0.8
                    },
                    'tooltip': f"CMA/CA: {row['CMA_CA']}<br>Graduates: {int(row['graduates'])}"
                }
            }
            for (_, row), color in zip(cma_data.iterrows(), colors)
        ]
        
        geojson_data = {'type': 'FeatureCollection', 'features': features}
        
        # Create charts with selections (simplified without chart type parameter)
        fig_isced = create_chart(
            isced_grads, 
            'ISCED_level_of_education', 
            'graduates',
            'ISCED Level of Education', 
            selected_isced
        )
        
        fig_province = create_chart(
            province_grads, 
            'Province_Territory', 
            'graduates',
            'Province/Territory', 
            selected_province
        )

        # Create new charts
        fig_cma = create_chart(
            cma_grads,
            'CMA_CA',
            'graduates',
            'Census Metropolitan Area',
            selected_feature
        )

        fig_credential = create_chart(
            credential_grads,
            'Credential_Type',
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
        'year': years or [],
        'Province_Territory': provs or [],
        'CMA_CA': cmas or [],
        'ISCED_level_of_education': isced or [],
        'Credential_Type': credentials or [],
        'Institution': institutions or []
    }
    
    # Get updated options for each filter
    stem_options = filter_options(data, 'STEM/BHASE', {k:v for k,v in current_filters.items() if k != 'STEM/BHASE'})
    year_options = filter_options(data, 'year', {k:v for k,v in current_filters.items() if k != 'year'})
    prov_options = filter_options(data, 'Province_Territory', {k:v for k,v in current_filters.items() if k != 'Province_Territory'})
    cma_options = filter_options(data, 'CMA_CA', {k:v for k,v in current_filters.items() if k != 'CMA_CA'})
    isced_options = filter_options(data, 'ISCED_level_of_education', {k:v for k,v in current_filters.items() if k != 'ISCED_level_of_education'})
    cred_options = filter_options(data, 'Credential_Type', {k:v for k,v in current_filters.items() if k != 'Credential_Type'})
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

if __name__ == '__main__':
    app.run_server(debug=True)

