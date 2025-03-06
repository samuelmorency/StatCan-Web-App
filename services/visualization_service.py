"""Visualization service for creating charts and maps."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import colorlover as cl
import brand_colours as bc
from services.cache_service import azure_cache_decorator

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
    
    chart_height = 500
    
    # If x_label = 'Institution' or 'CMA_CA', then chart_height=1000
    if x_label == 'Institution' or x_label == 'Census Metropolitan Area':
        num_bars = len(sorted_data.index)
        max_height = max(chart_height, 25 * len(sorted_data.index))
        chart_height = max_height
    
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
                color=[bc.LIGHT_GREY if x != selected_value else bc.MAIN_RED for x in sorted_data[x_column]] if selected_value else sorted_data[y_column],
                colorscale='Reds' if not selected_value else None
            )
        )
    )
    
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
        modebar_remove=['zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'lasso2d']
    )
    
    return fig

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
        'tooltip': f"<div style='font-family: Open Sans, sans-serif; font-weight: 600;'>CMA/CA: {row['CMA_CA']}<br>Graduates: {int(graduates):,}</div>"
    }

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
        empty_row_data,
        empty_column_defs,
        dict(bounds=default_bounds, transition="flyToBounds")
    )