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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure App Service cache configuration
class AzureCache:
    def __init__(self):
        self.CACHE_DIR = os.path.join(tempfile.gettempdir(), 'dash_cache')
        self.DEFAULT_SIZE = 512e6  # 512MB default
        self.FALLBACK_SIZE = 128e6  # 128MB fallback
        self.cache = None
        self.initialize_cache()

    def initialize_cache(self):
        """Initialize cache with fallback options"""
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            self.cache = Cache(self.CACHE_DIR, size_limit=self.DEFAULT_SIZE)
            logger.info(f"Cache initialized at {self.CACHE_DIR} with size {self.DEFAULT_SIZE/1e6}MB")
        except Exception as e:
            logger.warning(f"Primary cache initialization failed: {e}")
            try:
                self.cache = Cache(self.CACHE_DIR, size_limit=self.FALLBACK_SIZE)
                logger.info(f"Fallback cache initialized with size {self.FALLBACK_SIZE/1e6}MB")
            except Exception as e:
                logger.error(f"Cache initialization completely failed: {e}")
                self.cache = None

    def clear_cache(self):
        """Clear cache on shutdown"""
        if self.cache:
            try:
                self.cache.clear()
                logger.info("Cache cleared successfully")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")

    def get_cache(self):
        """Get cache instance with initialization check"""
        if not self.cache:
            self.initialize_cache()
        return self.cache

# Initialize Azure-specific cache
azure_cache = AzureCache()
cache = azure_cache.get_cache()

# Register cleanup
atexit.register(azure_cache.clear_cache)

# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Modified cache decorator with retry logic
def azure_cache_decorator(func):
    """Decorator for Azure-friendly caching with retry logic"""
    def wrapper(*args, **kwargs):
        if not cache:
            return func(*args, **kwargs)
        
        try:
            return cache.memoize(expire=3600)(func)(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Cache operation failed for {func.__name__}: {e}")
            return func(*args, **kwargs)
    return wrapper

# Enhanced caching configuration
cache = Cache('/tmp/dash_cache', size_limit=5e9)  # Increased to 5GB cache limit

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
        self.current_bounds = None
        self.current_zoom = None
        self.selected_feature = None
        self.hover_feature = None
        self.color_scale = None
        self.last_update_time = 0
        self.viewport_locked = False
        
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
        if not self.current_bounds or force:
            return True
            
        # Reset viewport lock periodically
        current_time = time.time()
        if current_time - self.last_update_time > 2.0:  # Reset lock after 2 seconds
            self.viewport_locked = False
            
        if self.viewport_locked and not force:
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
            self.current_bounds = bounds
            self.last_update_time = time.time()
        if zoom is not None:
            self.current_zoom = zoom
        if selected is not None:
            self.selected_feature = selected
        if hover is not None:
            self.hover_feature = hover
        if color_scale is not None:
            self.color_scale = color_scale

# Initialize map state
map_state = MapState()

@lru_cache(maxsize=32)
def calculate_optimal_viewport(bounds, padding_factor=0.1):
    """
    Computes an optimal viewport given geographic bounds. If the provided bounds
    are too small, they are adjusted to ensure a minimum viewing area. The function
    returns a dictionary including padded bounds, a calculated center point, and a
    suitable zoom level. If an error occurs, it returns default viewport parameters
    centered over Canada.

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
        
        # Handle invalid or zero-size bounds
        if abs(max_lat - min_lat) < 0.1:
            center_lat = (max_lat + min_lat) / 2
            min_lat = center_lat - 0.05
            max_lat = center_lat + 0.05
        if abs(max_lon - min_lon) < 0.1:
            center_lon = (max_lon + min_lon) / 2
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
        return {
            'bounds': [[41, -141], [83, -52]],
            'center': [56, -96],
            'zoom': 4
        }

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

@lru_cache(maxsize=128)
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
        self.ctx = context
        self.triggered = self.ctx.triggered[0] if self.ctx.triggered else None
        self.triggered_id = self.triggered['prop_id'].split('.')[0] if self.triggered else None

    @property
    def is_triggered(self):
        """
        Indicates whether the callback was triggered by any input.

        Returns:
            bool: True if the callback was triggered, False otherwise.
        """
        return bool(self.ctx.triggered)

@functools.lru_cache(maxsize=64)  # Increased cache size
@azure_cache_decorator
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

@functools.lru_cache(maxsize=1)
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

@cache.memoize(expire=3600)  # Cache for 1 hour
@azure_cache_decorator
def preprocess_data(selected_stem_bhase, selected_years, selected_provs, selected_isced, 
                   selected_credentials, selected_institutions):
    """
    Applies a series of filters to the pre-indexed global data. After filtering,
    aggregates are computed by summing over the relevant index levels rather than
    re-grouping the entire dataset. This approach uses direct indexing for efficient
    slicing, reducing overhead for frequent filter changes.

    Args:
        selected_stem_bhase (tuple): A tuple of STEM/BHASE categories to include.
        selected_years (tuple): A tuple of years to include.
        selected_provs (tuple): A tuple of provinces/territories to include.
        selected_isced (tuple): A tuple of ISCED levels to include.
        selected_credentials (tuple): A tuple of credential types to include.
        selected_institutions (tuple): A tuple of institutions to include.

    Returns:
        tuple: A tuple containing:
               - filtered_data (pandas.DataFrame): The filtered dataset at the granular level.
               - cma_grads (pandas.DataFrame): Aggregation of graduates by CMA/CA and DGUID.
               - isced_grads (pandas.DataFrame): Aggregation of graduates by ISCED level.
               - province_grads (pandas.DataFrame): Aggregation of graduates by province/territory.
    """
    filters = {
        'STEM/BHASE': set(selected_stem_bhase),
        'year': set(selected_years),
        'Province_Territory': set(selected_provs),
        'ISCED_level_of_education': set(selected_isced),
        'Credential_Type': set(selected_credentials),
        'Institution': set(selected_institutions),
        'CMA_CA': set(),
        'DGUID': set()
    }

    filtered_data = filter_data(data, filters)
    if filtered_data.empty:
        return filtered_data.reset_index(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Vectorized operations for aggregations
    aggregations = {
        'cma': filtered_data.groupby(["CMA_CA", "DGUID"], observed=True)['value'].sum(),
        'isced': filtered_data.groupby("ISCED_level_of_education", observed=True)['value'].sum(),
        'province': filtered_data.groupby("Province_Territory", observed=True)['value'].sum()
    }

    return (
        filtered_data.reset_index(),
        aggregations['cma'].reset_index(name='graduates'),
        aggregations['isced'].reset_index(name='graduates'),
        aggregations['province'].reset_index(name='graduates')
    )

# Update color constants
PRIMARY_RED = '#C01823'
DARK_BLUE = '#00758D'
MAIN_BLUE = '#008FBE'
LIGHT_BLUE = '#B8D8EB'
IIC_BLACK = '#24272A'
GREY = '#54575A'
LIGHT_GREY = '#97989A'

# Custom color sequence for choropleth maps
CUSTOM_REDS = [
    '#FFE5E6',  # Lightest
    '#FFCCCD',
    '#FF9999',
    '#FF6666',
    '#FF3333',
    '#FF0000',
    '#CC0000',
    '#990000',
    '#C01823'   # Darkest - matches brand red
]

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
            'fillColor': MAIN_BLUE if is_selected else color,
            'color': IIC_BLACK if is_selected else GREY,
            'weight': 2 if is_selected else 0.5,
            'fillOpacity': 0.8
        },
        'tooltip': f"CMA/CA: {row['CMA_CA']}<br>Graduates: {int(graduates)}"
    }

@cache.memoize(expire=300)  # Cache for 5 minutes
@azure_cache_decorator
def create_chart(dataframe, x_column, y_column, x_label, selected_value=None):
    """Creates a horizontal bar chart."""
    if dataframe.empty:
        return {}
    
    sorted_data = dataframe.sort_values(y_column, ascending=True)
    stats = {
        'vmin': sorted_data[y_column].quantile(0.01),
        'vmax': sorted_data[y_column].max() or 0
    }
    
    fig = px.bar(
        sorted_data,
        x=y_column,
        y=x_column,
        orientation='h',
        title=f'Number of Graduates by {x_label}',
        labels={y_column: 'Number of Graduates', x_column: x_label},
        color=y_column,
        color_continuous_scale=[LIGHT_BLUE, MAIN_BLUE, DARK_BLUE],
        range_color=[stats['vmin'], stats['vmax']]
    )
    
    if selected_value:
        colors = [LIGHT_GREY if x != selected_value else PRIMARY_RED for x in sorted_data[x_column]]
        fig.data[0].marker.color = colors
        fig.update_coloraxes(showscale=False)
    
    fig.update_layout(
        xaxis_title='Number of Graduates',
        yaxis_title=x_label,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        clickmode='event+select',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': IIC_BLACK}
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
               - Empty data for the table (list)
               - Empty columns for the table (list)
               - A default viewport dictionary with bounds and transition
    """
    empty_geojson = {'type': 'FeatureCollection', 'features': []}
    empty_fig = {}
    empty_data = []
    empty_columns = []
    default_bounds = [[41, -141], [83, -52]]  # Canada bounds
    
    return (
        empty_geojson,
        empty_fig,
        empty_fig,
        empty_data,
        empty_columns,
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
isced_options_full = [{'label': level, 'value': level} for level in sorted(data.index.get_level_values('ISCED_level_of_education').unique())]
credential_options_full = [{'label': cred, 'value': cred} for cred in sorted(data.index.get_level_values('Credential_Type').unique())]
institution_options_full = [{'label': inst, 'value': inst} for inst in sorted(data.index.get_level_values('Institution').unique())]

# Create the app layout
app.layout = dbc.Container([
    html.H1("Interactive Choropleth Map of STEM/BHASE Graduates in Canada", className="my-4"),
    dbc.Row([
        dbc.Col([
            html.H5("Filters"),
            dbc.Button(
                html.I(className="fas fa-info-circle"),
                id="filter-info",
                color="link",
                className="p-0 mb-2"
            ),
            dcc.Tooltip(
                id="filter-tooltip",
                children="Use these filters to narrow down the data displayed on the map and charts.",
                style={
                    'backgroundColor': 'white',
                    'color': 'black',
                    'border': '1px solid #ccc',
                    'padding': '10px',
                    'fontSize': '12px'
                }
            ),
            html.Label([
                "STEM/BHASE ",
                dbc.Button(
                    html.I(className="fas fa-info-circle"),
                    id="stem-info",
                    color="link",
                    size="sm",
                    className="p-0"
                ),
            ]),
            dcc.Tooltip(
                id="stem-tooltip",
                children=[
                    html.P("STEM: Science, Technology, Engineering, Mathematics"),
                    html.P("BHASE: Business, Humanities, Arts, Social Science, Education")
                ],
                style={
                    'backgroundColor': 'white',
                    'color': 'black',
                    'border': '1px solid #ccc',
                    'padding': '10px',
                    'fontSize': '12px'
                }
            ),
            html.Label("STEM/BHASE:"),
            dcc.Checklist(
                id='stem-bhase-filter',
                options=stem_bhase_options_full,
                value=[option['value'] for option in stem_bhase_options_full],
                inputStyle={"margin-right": "5px", "margin-left": "20px"},
                style={"margin-bottom": "15px"}
            ),
            html.Label("Academic Year:"),
            dcc.Checklist(
                id='year-filter',
                options=year_options_full,
                value=[option['value'] for option in year_options_full],
                inputStyle={"margin-right": "5px", "margin-left": "20px"},
                style={"margin-bottom": "15px"}
            ),
            html.Label("Province:"),
            dcc.Dropdown(
                id='prov-filter',
                options=prov_options_full,
                value=[],
                multi=True,
                placeholder="All Provinces",
                searchable=True,
                style={"margin-bottom": "15px"}
            ),
            html.Label("ISCED Level:"),
            dcc.Dropdown(
                id='isced-filter',
                options=isced_options_full,
                value=[],
                multi=True,
                placeholder="All Levels",
                searchable=True,
                style={"margin-bottom": "15px"}
            ),
            html.Label("Credential Type:"),
            dcc.Dropdown(
                id='credential-filter',
                options=credential_options_full,
                value=[],
                multi=True,
                placeholder="All Credential Types",
                searchable=True,
                style={"margin-bottom": "15px"}
            ),
            html.Label("Institution:"),
            dcc.Dropdown(
                id='institution-filter',
                options=institution_options_full,
                value=[],
                multi=True,
                placeholder="All Institutions",
                searchable=True,
                style={"margin-bottom": "15px"}
            ),
            html.Button('Reset Filters', 
                id='reset-filters', 
                n_clicks=0, 
                style={
                    "margin-top": "15px",
                    "background-color": MAIN_BLUE,
                    "color": "white",
                    "border": "none",
                    "padding": "10px 20px",
                    "border-radius": "5px",
                    "margin-right": "10px"
                }
            ),
            html.Button('Clear Selection', 
                id='clear-selection', 
                n_clicks=0, 
                style={
                    "margin-top": "15px",
                    "background-color": LIGHT_GREY,
                    "color": IIC_BLACK,
                    "border": "none",
                    "padding": "10px 20px",
                    "border-radius": "5px"
                }
            ),
            # Add dcc.Store components to store selected data for cross-filtering
            dcc.Store(id='selected-isced', data=None),
            dcc.Store(id='selected-province', data=None),
            dcc.Store(id='selected-cma', data=None),
            # Remove download buttons and components
        ], width=3, style={
            "background-color": LIGHT_GREY,
            "padding": "20px",
            "border-radius": "5px"
        }),

        dbc.Col([
            # Remove Spinner from map, keep direct map component
            html.Div([
                dl.Map(
                    id='map',
                    center=[56, -96],
                    zoom=4,
                    children=[
                        dl.TileLayer(),
                        dl.GeoJSON(
                            id='cma-geojson',
                            data=None,
                            style=assign("""
                            function(feature) {
                                return feature.properties.style;
                            }
                            """),
                            zoomToBounds=True,
                            hoverStyle=dict(
                                weight=2, color='black', dashArray='',
                                fillOpacity=0.7
                            ),
                            onEachFeature=assign("""
                            function(feature, layer) {
                                if (feature.properties && feature.properties.tooltip) {
                                    layer.bindTooltip(feature.properties.tooltip);
                                }
                            }
                            """),
                            options=dict(interactive=True),
                            eventHandlers=dict(
                                click=assign("""
                                function(e, ctx) {
                                    e.originalEvent._stopped = true;
                                    const clickData = {
                                        feature: e.sourceTarget.feature.properties.DGUID,
                                        points: [{
                                            featureId: e.sourceTarget.feature.properties.DGUID
                                        }]
                                    };
                                    ctx.setProps({ 
                                        clickData: clickData,
                                        clickedFeature: null  // Reset the clicked feature
                                    });
                                }
                                """)
                            ),
                        ),
                    ],
                    style={'width': '100%', 'height': '600px'},
                ),
            ], style={"height": "600px"}),

            # Keep existing spinners for charts
            # Arrange the two graphs side by side with chart type selection
            dbc.Row([
                dbc.Col([
                    dbc.Spinner(
                        dcc.Graph(id='graph-isced'),
                        color="primary",
                        type="border",
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Spinner(
                        dcc.Graph(id='graph-province'),
                        color="primary",
                        type="border",
                    ),
                ], width=6)
            ]),

            # Add the scrollable table at the bottom
            html.H3("Number of Graduates by CMA/CA"),
            dash_table.DataTable(
                id='table-cma',
                columns=[],  # Placeholder for table columns
                data=[],  # Placeholder for table data
                style_table={'height': '400px', 'overflowY': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'color': IIC_BLACK,
                    'backgroundColor': 'white'
                },
                style_header={
                    'backgroundColor': LIGHT_BLUE,
                    'color': IIC_BLACK,
                    'fontWeight': 'bold'
                },
                page_action='none',  # Disable pagination
                sort_action='native',  # Enable sorting
                filter_action='native',  # Enable filtering
            ),
        ], width=9)
    ])
], fluid=True)

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
    if triggered_id == 'selected-cma' and selected_feature:
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
    Output('selected-cma', 'data'),
    Input('cma-geojson', 'clickData'),
    Input('clear-selection', 'n_clicks'),
    State('selected-cma', 'data'),
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
                'color': DARK_BLUE,
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
    Output('table-cma', 'data'),
    Output('table-cma', 'columns'),
    Output('map', 'viewport'),
    Input('stem-bhase-filter', 'value'),
    Input('year-filter', 'value'),
    Input('prov-filter', 'value'),
    Input('isced-filter', 'value'),
    Input('credential-filter', 'value'),
    Input('institution-filter', 'value'),
    Input('selected-isced', 'data'),
    Input('selected-province', 'data'),
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
               - Data for the CMA/CA table
               - Columns for the CMA/CA table
               - Updated viewport settings
    """
    try:
        current_viewport = args[-1]
        (stem_bhase, years, provs, isced, credentials, institutions, 
         selected_isced, selected_province, selected_feature) = args[:-1]
        
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
            
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Process data with optimized function
        filtered_data, cma_grads, isced_grads, province_grads = preprocess_data(
            tuple(stem_bhase or []),
            tuple(years or []),
            tuple(provs or []),
            tuple(isced or []),
            tuple(credentials or []),
            tuple(institutions or [])
        )
        
        # Apply cross-filtering with vectorized operations
        if any([selected_isced, selected_province, selected_feature]):
            mask = pd.Series(True, index=filtered_data.index)
            if selected_isced:
                mask &= filtered_data['ISCED_level_of_education'] == selected_isced
            if selected_province:
                mask &= filtered_data['Province_Territory'] == selected_province
            if selected_feature:
                mask &= filtered_data['DGUID'] == selected_feature

            filtered_data = filtered_data[mask]
            if filtered_data.empty:
                return create_empty_response()

            # Aggregations with observed=True
            cma_grads = filtered_data.groupby(["CMA_CA", "DGUID"], observed=True)['value'].sum().reset_index(name='graduates')
            isced_grads = filtered_data.groupby("ISCED_level_of_education", observed=True)['value'].sum().reset_index(name='graduates')
            province_grads = filtered_data.groupby("Province_Territory", observed=True)['value'].sum().reset_index(name='graduates')

        # Prepare map data efficiently
        cma_data = combined_longlat_clean.merge(cma_grads, on='DGUID', how='left')
        cma_data['graduates'] = cma_data['graduates'].fillna(0)
        cma_data = cma_data[cma_data['graduates'] > 0]
        
        if cma_data.empty:
            return create_empty_response()
        
        should_update_viewport = triggered_id in [
            'stem-bhase-filter', 'year-filter', 'prov-filter',
            'isced-filter', 'credential-filter', 'institution-filter',
            'selected-cma', 'clear-selection', 'reset-filters'
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
            
        # Create GeoJSON efficiently
        colorscale = CUSTOM_REDS
        max_graduates = cma_data['graduates'].max()
        min_graduates = cma_data['graduates'].min()
        
        features = [
            {
                'type': 'Feature',
                'geometry': row.geometry.__geo_interface__,
                'properties': create_geojson_feature(
                    row, colorscale, max_graduates, min_graduates, selected_feature
                )
            }
            for _, row in cma_data.iterrows()
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
        
        # Prepare table data
        table_data = cma_grads.sort_values('graduates', ascending=False).to_dict('records')
        table_columns = [{"name": i, "id": i} for i in cma_grads.columns]
        
        # Monitor cache at the start of major updates
        monitor_cache_usage()
        
        return (
            geojson_data,
            fig_isced,
            fig_province,
            table_data,
            table_columns,
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
        [], [], [], []
    )

if __name__ == '__main__':
    app.run_server(debug=False)
