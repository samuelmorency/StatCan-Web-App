"""Utility functions and classes for the Infozone application."""

import time
import logging
from functools import wraps
from collections import defaultdict
from dash import callback_context

# Configure logging
logger = logging.getLogger(__name__)

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


def monitor_performance(func):
    """
    Decorator to monitor the performance of functions.
    Logs average execution time after every 10 calls.
    
    Args:
        func (callable): The function to monitor
        
    Returns:
        callable: Wrapped function with performance monitoring
    """
    metrics = defaultdict(list)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        metrics[func.__name__].append(execution_time)
        
        # Log average performance every 10 calls
        if len(metrics[func.__name__]) >= 10:
            avg_time = sum(metrics[func.__name__]) / len(metrics[func.__name__])
            logger.info(f"{func.__name__} average execution time: {avg_time:.4f}s")
            metrics[func.__name__] = []
            
        return result
    return wrapper


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
    from services.visualization_service import calculate_optimal_viewport
    
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