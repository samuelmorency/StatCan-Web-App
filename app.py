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


# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

cache = Cache('/tmp/dash_cache')

class MapState:
    def __init__(self):
        self.current_bounds = None
        self.current_zoom = None
        self.selected_feature = None
        self.hover_feature = None
        self.color_scale = None
        self.last_update_time = 0
        self.viewport_locked = False
        
    def should_update_viewport(self, new_bounds, force=False):
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

# Add optimized helper functions
@lru_cache(maxsize=32)
def calculate_optimal_viewport(bounds, padding_factor=0.1):
    """
    Calculate optimal viewport settings for given bounds.
    
    Args:
        bounds (tuple): (min_lat, min_lon, max_lat, max_lon)
        padding_factor (float): Padding factor for bounds
        
    Returns:
        dict: Viewport settings including bounds, center, and zoom
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
    if max_val == min_val:
        return {min_val: cl.scales[str(n_colors)]['seq']['Reds'][-1]}
    breaks = np.linspace(min_val, max_val, n_colors)
    colors = cl.scales[str(n_colors)]['seq']['Reds']
    return dict(zip(breaks, colors))

class CallbackContextManager:
    def __init__(self, context):
        self.ctx = context
        self.triggered = self.ctx.triggered[0] if self.ctx.triggered else None
        self.triggered_id = self.triggered['prop_id'].split('.')[0] if self.triggered else None

    @property
    def is_triggered(self):
        return bool(self.ctx.triggered)

# Load spatial data
@functools.lru_cache(maxsize=1)
def load_spatial_data():
    province_longlat_clean = gpd.read_parquet("province_longlat_clean.parquet")
    combined_longlat_clean = gpd.read_parquet("combined_longlat_clean.parquet")

    return province_longlat_clean, combined_longlat_clean

@functools.lru_cache(maxsize=1)
def load_and_process_educational_data():
    data = pd.read_pickle("cleaned_data.pkl")
    
    return data

# Load initial data
province_longlat_clean, combined_longlat_clean = load_spatial_data()
data = load_and_process_educational_data()

def filter_data(data, filters):
    """Efficiently filter data using vectorized operations"""
    mask = pd.Series(True, index=data.index)
    
    for column, values in filters.items():
        if values:
            mask &= data[column].isin(values)
    
    return data[mask]

@functools.lru_cache(maxsize=128)
def preprocess_data(selected_stem_bhase, selected_years, selected_provs, selected_isced, 
                   selected_credentials, selected_institutions):
    # Create filters dictionary with frozensets for hashability
    filters = {
        'STEM/BHASE': frozenset(selected_stem_bhase) if selected_stem_bhase else frozenset(),
        'year': frozenset(selected_years) if selected_years else frozenset(),
        'Province_Territory': frozenset(selected_provs) if selected_provs else frozenset(),
        'ISCED_level_of_education': frozenset(selected_isced) if selected_isced else frozenset(),
        'Credential_Type': frozenset(selected_credentials) if selected_credentials else frozenset(),
        'Institution': frozenset(selected_institutions) if selected_institutions else frozenset()
    }
    
    # Get filtered data
    filtered_data = filter_data(data, filters)
    
    # Optimize aggregations using named aggregation
    aggregations = {
        'cma': filtered_data.groupby(["CMA_CA", "DGUID"])
              .agg(graduates=('value', 'sum'))
              .reset_index(),
        'isced': filtered_data.groupby("ISCED_level_of_education")
                .agg(graduates=('value', 'sum'))
                .reset_index(),
        'province': filtered_data.groupby("Province_Territory")
                   .agg(graduates=('value', 'sum'))
                   .reset_index()
    }
    
    return filtered_data, aggregations['cma'], aggregations['isced'], aggregations['province']

def create_geojson_feature(row, colorscale, max_graduates, min_graduates, selected_cma):
    graduates = row['graduates']
    cmapuid = str(row['DGUID'])
    
    # Optimize color calculation
    if max_graduates > min_graduates:
        normalized_value = (graduates - min_graduates) / (max_graduates - min_graduates)
        color_index = int(normalized_value * (len(colorscale) - 1))
        color = colorscale[color_index] if graduates > 0 else 'lightgray'
    else:
        color = colorscale[0] if graduates > 0 else 'lightgray'
    
    is_selected = selected_cma and cmapuid == selected_cma
    
    return {
        'graduates': int(graduates),
        'DGUID': cmapuid,
        'CMA_CA': row['NAME'],
        'style': {
            'fillColor': 'yellow' if is_selected else color,
            'color': 'black' if is_selected else 'gray',
            'weight': 2 if is_selected else 0.5,
            'fillOpacity': 0.8
        },
        'tooltip': f"CMA/CA: {row['NAME']}<br>Graduates: {int(graduates)}"
    }



def create_chart(dataframe, x_column, y_column, chart_type, x_label, colorscale, selected_value):
    """Optimized chart creation with cached calculations"""
    if dataframe.empty:
        return {}
    
    # Pre-calculate statistics once
    stats = {
        'vmin': dataframe[y_column].quantile(0.01),
        'vmax': dataframe[y_column].max()
    }
    
    if stats['vmin'] == stats['vmax']:
        stats['vmin'] = 0
        
    # Create charts based on type
    if chart_type == 'bar':
        return create_bar_chart(dataframe, x_column, y_column, x_label, stats, selected_value)
    elif chart_type == 'line':
        return create_line_chart(dataframe, x_column, y_column, x_label)
    elif chart_type == 'pie':
        return create_pie_chart(dataframe, x_column, y_column, x_label, selected_value)
    
    return {}

def create_bar_chart(dataframe, x_column, y_column, x_label, stats, selected_value):
    """Optimized bar chart creation"""
    sorted_data = dataframe.sort_values(y_column, ascending=True)
    
    fig = px.bar(
        sorted_data,
        x=y_column,
        y=x_column,
        orientation='h',
        title=f'Number of Graduates by {x_label}',
        labels={y_column: 'Number of Graduates', x_column: x_label},
        color=y_column,
        color_continuous_scale=px.colors.sequential.Reds,
        range_color=[stats['vmin'], stats['vmax']]
    )
    
    if selected_value:
        colors = ['grey'] * len(sorted_data)
        idx = sorted_data[sorted_data[x_column] == selected_value].index
        if not idx.empty:
            colors[idx[0]] = 'red'
        fig.data[0].marker.color = colors
        fig.update_coloraxes(showscale=False)
    
    fig.update_layout(
        xaxis_title='Number of Graduates',
        yaxis_title=x_label,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        clickmode='event+select'
    )
    
    return fig

def create_line_chart(dataframe, x_column, y_column, x_label):
    """Optimized line chart creation"""
    sorted_data = dataframe.sort_values(x_column, ascending=True)
    
    fig = px.line(
        sorted_data,
        x=x_column,
        y=y_column,
        title=f'Number of Graduates by {x_label}',
        labels={y_column: 'Number of Graduates', x_column: x_label},
    )
    
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title='Number of Graduates',
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        clickmode='event+select'
    )
    
    return fig

def create_pie_chart(dataframe, x_column, y_column, x_label, selected_value):
    """Optimized pie chart creation"""
    fig = px.pie(
        dataframe,
        names=x_column,
        values=y_column,
        title=f'Number of Graduates by {x_label}',
        color_discrete_sequence=px.colors.sequential.Reds,
    )
    
    if selected_value:
        pull = [0.1 if name == selected_value else 0 for name in dataframe[x_column]]
        fig.update_traces(pull=pull)
    
    fig.update_layout(
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        clickmode='event+select'
    )
    
    return fig

def create_empty_response():
    """Create empty response for error cases"""
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

# Generate initial filter options
stem_bhase_options_full = [{'label': stem, 'value': stem} for stem in sorted(data['STEM/BHASE'].unique())]
year_options_full = [{'label': year, 'value': year} for year in sorted(data['year'].unique())]
prov_options_full = [{'label': prov, 'value': prov} for prov in sorted(data['Province_Territory'].unique())]
isced_options_full = [{'label': level, 'value': level} for level in sorted(data['ISCED_level_of_education'].unique())]
credential_options_full = [{'label': cred, 'value': cred} for cred in sorted(data['Credential_Type'].unique())]
institution_options_full = [{'label': inst, 'value': inst} for inst in sorted(data['Institution'].unique())]

# Create the app layout
app.layout = dbc.Container([
    html.H1("Interactive Choropleth Map of STEM/BHASE Graduates in Canada", className="my-4"),

    # Define the layout for filters and map
    dbc.Row([
        dbc.Col([
            html.H5("Filters"),
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
            html.Button('Reset Filters', id='reset-filters', n_clicks=0, style={"margin-top": "15px"}),
            html.Button('Clear Selection', id='clear-selection', n_clicks=0, style={"margin-top": "15px"}),
            # Add dcc.Store components to store selected data for cross-filtering
            dcc.Store(id='selected-isced', data=None),
            dcc.Store(id='selected-province', data=None),
            dcc.Store(id='selected-cma', data=None),
        ], width=3, style={"background-color": "#f8f9fa", "padding": "20px"}),

        dbc.Col([
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
                    # Remove updateTrigger as it's not needed
                ),
            ], style={"height": "600px"}),

            # Arrange the two graphs side by side with chart type selection
            dbc.Row([
                dbc.Col([
                    html.Label("Chart Type:"),
                    dcc.RadioItems(
                        id='chart-type-isced',
                        options=[
                            {'label': 'Bar', 'value': 'bar'},
                            {'label': 'Line', 'value': 'line'},
                            {'label': 'Pie', 'value': 'pie'}
                        ],
                        value='bar',
                        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                    ),
                    dcc.Graph(id='graph-isced'),  # Graph for ISCED level of education
                ], width=6),
                dbc.Col([
                    html.Label("Chart Type:"),
                    dcc.RadioItems(
                        id='chart-type-province',
                        options=[
                            {'label': 'Bar', 'value': 'bar'},
                            {'label': 'Line', 'value': 'line'},
                            {'label': 'Pie', 'value': 'pie'}
                        ],
                        value='bar',
                        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                    ),
                    dcc.Graph(id='graph-province'),  # Graph for provinces
                ], width=6)
            ]),

            # Add the scrollable table at the bottom
            html.H3("Number of Graduates by CMA/CA"),
            dash_table.DataTable(
                id='table-cma',
                columns=[],  # Placeholder for table columns
                data=[],  # Placeholder for table data
                style_table={'height': '400px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left'},
                page_action='none',  # Disable pagination
                sort_action='native',  # Enable sorting
                filter_action='native',  # Enable filtering
            ),
        ], width=9)
    ])
], fluid=True)

def calculate_viewport_update(triggered_id, cma_data, selected_cma=None):
    """Helper function to determine viewport updates"""
    if triggered_id == 'selected-cma' and selected_cma:
        # For clicked features, zoom to the selected feature
        selected_geometry = cma_data[cma_data['DGUID'] == selected_cma]
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
def update_selected_cma(click_data, n_clicks, stored_cma):
    ctx_manager = CallbackContextManager(callback_context)
    
    if not ctx_manager.is_triggered:
        raise PreventUpdate
        
    if ctx_manager.triggered_id == 'clear-selection':
        return None
        
    if ctx_manager.triggered_id == 'cma-geojson' and click_data and 'points' in click_data:
        clicked_id = click_data['points'][0]['featureId']
        return None if stored_cma == clicked_id else clicked_id
        
    return None

# Optimized callback for ISCED selection with better error handling
@app.callback(
    Output('selected-isced', 'data'),
    Input('graph-isced', 'clickData'),
    Input('clear-selection', 'n_clicks'),
    State('selected-isced', 'data'),
    prevent_initial_call=True
)
def update_selected_isced(clickData, n_clicks, stored_isced):
    ctx_manager = CallbackContextManager(callback_context)
    
    if not ctx_manager.is_triggered:
        raise dash.exceptions.PreventUpdate
        
    if ctx_manager.triggered_id == 'clear-selection':
        return None
        
    if ctx_manager.triggered_id == 'graph-isced' and clickData and 'points' in clickData:
        clicked_value = clickData['points'][0]['y']
        return None if stored_isced == clicked_value else clicked_value
        
    return stored_isced

# Optimized callback for province selection
@app.callback(
    Output('selected-province', 'data'),
    Input('graph-province', 'clickData'),
    Input('clear-selection', 'n_clicks'),
    State('selected-province', 'data'),
    State('graph-province', 'figure'),
    prevent_initial_call=True
)
def update_selected_province(clickData, n_clicks, stored_province, figure):
    ctx_manager = CallbackContextManager(callback_context)
    
    if not ctx_manager.is_triggered:
        raise dash.exceptions.PreventUpdate
        
    if ctx_manager.triggered_id == 'clear-selection':
        return None
        
    if ctx_manager.triggered_id == 'graph-province' and clickData and 'points' in clickData:
        orientation = figure['data'][0].get('orientation', 'v')
        clicked_value = clickData['points'][0]['y' if orientation == 'h' else 'x']
        return None if stored_province == clicked_value else clicked_value
        
    return stored_province

def update_map_style(geojson_data, colorscale, selected_cma=None):
    """Create a Patch object for updating map styles"""
    patched_geojson = Patch()
    
    if not geojson_data or 'features' not in geojson_data:
        return patched_geojson
        
    for i, feature in enumerate(geojson_data['features']):
        cmapuid = feature['properties']['DGUID']
        is_selected = selected_cma and cmapuid == selected_cma
        
        # Update only style properties that need to change
        style_updates = {
            'fillColor': 'yellow' if is_selected else feature['properties']['style']['fillColor'],
            'color': 'black' if is_selected else 'gray',
            'weight': 2 if is_selected else 0.5,
        }
        
        patched_geojson['features'][i]['properties']['style'].update(style_updates)
    
    return patched_geojson

# Add hover callback
@app.callback(
    Output('cma-geojson', 'data', allow_duplicate=True),
    Input('cma-geojson', 'hover_feature'),
    State('cma-geojson', 'data'),
    prevent_initial_call=True
)
def update_hover_style(hover_feature, current_geojson):
    """Update map styles on hover"""
    if not hover_feature or not current_geojson:
        raise PreventUpdate
        
    patched_geojson = Patch()
    hover_id = hover_feature.get('id')
    
    for i, feature in enumerate(current_geojson['features']):
        feature_id = feature['properties']['DGUID']
        if feature_id == hover_id:
            patched_geojson['features'][i]['properties']['style'].update({
                'weight': 3,
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
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    # Update map state with the new viewport
    if viewport and 'bounds' in viewport:
        map_state.update_state(bounds=viewport['bounds'])
    
    return no_update

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
    Input('chart-type-isced', 'value'),
    Input('chart-type-province', 'value'),
    Input('selected-isced', 'data'),
    Input('selected-province', 'data'),
    Input('selected-cma', 'data'),
    State('map', 'viewport')
)
def update_visualizations(*args):
    try:
        current_viewport = args[-1]  # Get the current viewport state
        (stem_bhase, years, provs, isced, credentials, institutions, 
         chart_type_isced, chart_type_province, selected_isced, 
         selected_province, selected_cma) = args[:-1]  # Get other inputs
        
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
        if any([selected_isced, selected_province, selected_cma]):
            filter_conditions = pd.Series(True, index=filtered_data.index)
            
            if selected_isced:
                filter_conditions &= filtered_data['ISCED_level_of_education'] == selected_isced
            if selected_province:
                filter_conditions &= filtered_data['Province_Territory'] == selected_province
            if selected_cma:
                filter_conditions &= filtered_data['DGUID'] == selected_cma
                
            filtered_data = filtered_data[filter_conditions]
            
            # Recalculate aggregations efficiently
            cma_grads = filtered_data.groupby(["CMA_CA", "DGUID"]).agg(graduates=('value', 'sum')).reset_index()
            isced_grads = filtered_data.groupby("ISCED_level_of_education").agg(graduates=('value', 'sum')).reset_index()
            province_grads = filtered_data.groupby("Province_Territory").agg(graduates=('value', 'sum')).reset_index()
        
        # Prepare map data efficiently
        cma_data = combined_longlat_clean.merge(cma_grads, on='DGUID', how='left')
        cma_data['graduates'] = cma_data['graduates'].fillna(0)
        cma_data = cma_data[cma_data['graduates'] > 0]
        
        if cma_data.empty:
            return create_empty_response()
        
        # Only update viewport for specific triggers
        should_update_viewport = triggered_id in [
            'stem-bhase-filter', 'year-filter', 'prov-filter',
            'isced-filter', 'credential-filter', 'institution-filter',
            'selected-cma', 'clear-selection', 'reset-filters'
        ]
        
        if should_update_viewport:
            # Calculate bounds with padding
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
        colorscale = cl.scales['9']['seq']['Reds']
        max_graduates = cma_data['graduates'].max()
        min_graduates = cma_data['graduates'].min()
        
        features = [
            {
                'type': 'Feature',
                'geometry': row.geometry.__geo_interface__,
                'properties': create_geojson_feature(
                    row, colorscale, max_graduates, min_graduates, selected_cma
                )
            }
            for _, row in cma_data.iterrows()
        ]
        
        geojson_data = {'type': 'FeatureCollection', 'features': features}
        
        # Create charts with selections
        fig_isced = create_chart(
            isced_grads, 
            'ISCED_level_of_education', 
            'graduates',
            chart_type_isced, 
            'ISCED Level of Education', 
            colorscale, 
            selected_isced
        )
        
        fig_province = create_chart(
            province_grads, 
            'Province_Territory', 
            'graduates',
            chart_type_province, 
            'Province/Territory', 
            colorscale, 
            selected_province
        )
        
        # Prepare table data
        table_data = cma_grads.sort_values('graduates', ascending=False).to_dict('records')
        table_columns = [{"name": i, "id": i} for i in cma_grads.columns]
        
        return (
            geojson_data,
            fig_isced,
            fig_province,
            table_data,
            table_columns,
            viewport_output
        )
        
    except Exception as e:
        print(f"Error in update_visualizations: {str(e)}")
        return create_empty_response()

def create_patched_selection(data, column, selected_value, chart_type):
    patched_figure = Patch()
    
    if chart_type == 'bar':
        colors = ['grey' if x != selected_value else 'red' 
                 for x in data[column]]
        patched_figure['data'][0]['marker']['color'] = colors
        
    elif chart_type == 'pie':
        pull = [0.1 if x == selected_value else 0 
                for x in data[column]]
        patched_figure['data'][0]['pull'] = pull
        
    return patched_figure

def create_empty_response():
    empty_geojson = {'type': 'FeatureCollection', 'features': []}
    empty_fig = {}
    empty_data = []
    empty_columns = []
    default_bounds = [[41, -141], [83, -52]]
    
    return (
        empty_geojson,
        empty_fig,
        empty_fig,
        empty_data,
        empty_columns,
        dict(bounds=default_bounds, transition=dict(duration=1000))
    )

# Add a callback for filter reset
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
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
        
    return (
        [option['value'] for option in stem_bhase_options_full],
        [option['value'] for option in year_options_full],
        [], [], [], []
    )

if __name__ == '__main__':
    app.run_server(debug=False)