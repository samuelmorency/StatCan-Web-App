# data_utils.py – Data filtering and chart utilities
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import brand_colours as bc
from cache_utils import azure_cache_decorator, logger

# Performance monitoring decorator (log average exec time every 10 calls)
def monitor_performance(func):
    metrics = []
    def wrapper(*args, **kwargs):
        import time
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        metrics.append(duration)
        if len(metrics) >= 10:
            avg = sum(metrics)/len(metrics)
            print(f"{func.__name__} avg execution time: {avg:.4f}s")  # simple console output
            metrics.clear()
        return result
    return wrapper

# (The FilterOptimizer class from original code, for optimized multi-index filtering)
class FilterOptimizer:
    def __init__(self, data):
        self._data = data
        self._cache = {}
        self._index_cache = {}
        self._last_hash = None
    def _make_hash(self, filters):
        return __import__('hashlib').md5(str(sorted(filters.items())).encode()).hexdigest()
    def _get_index(self, col):
        if col not in self._index_cache:
            self._index_cache[col] = self._data.index.get_level_values(col)
        return self._index_cache[col]
    @monitor_performance
    def filter_data(self, filters: dict):
        # Return cached if same filter combo
        f_hash = self._make_hash(filters)
        if f_hash == self._last_hash and f_hash in self._cache:
            return self._cache[f_hash]
        # Compute mask across multi-index levels
        mask = pd.Series(True, index=self._data.index)
        for col, values in filters.items():
            if values:
                idx = self._get_index(col)
                mask &= idx.isin(values)
        filtered = self._data[mask]
        # Cache result and trim cache size
        self._cache[f_hash] = filtered
        self._last_hash = f_hash
        if len(self._cache) > 10:
            self._cache.pop(next(iter(self._cache)))
        return filtered

# These will be set in app.py after data is loaded
filter_optimizer: FilterOptimizer = None
# Option lists for filters (populated in app.py)
stem_bhase_options_full = year_options_full = []
prov_options_full = cma_options_full = []
isced_options_full = credential_options_full = institution_options_full = []

@azure_cache_decorator(ttl=300)
@monitor_performance
def preprocess_data(sel_stem, sel_years, sel_provs, sel_isced, sel_creds, sel_institutions, sel_cmas):
    """
    Filter the main DataFrame according to selections and aggregate the results.
    Returns a tuple: (filtered_df, cma_grads, isced_grads, province_grads, credential_grads, institution_grads).
    """
    # Build filter criteria
    filters = {
        'STEM/BHASE': set(sel_stem),
        'Academic Year': set(sel_years),
        'Province or Territory': set(sel_provs),
        'ISCED Level of Education': set(sel_isced),
        'Credential Type': set(sel_creds),
        'Institution': set(sel_institutions),
        'CMA/CSD': set(sel_cmas),
        'DGUID': set()  # always empty (acts as a placeholder for selection filtering)
    }
    # Apply optimized filtering
    filtered_data = filter_optimizer.filter_data(filters)
    if filtered_data.empty:
        # Return empty dataframes for each aggregation if nothing passes the filters
        empty_df = pd.DataFrame()
        return (filtered_data.reset_index(), empty_df, empty_df, empty_df, empty_df, empty_df)
    logger.info(f"filtered_data shape: {filtered_data.shape}")
    logger.info(f"filtered_data sample:\n{filtered_data.head()}")
    # Aggregate graduate counts by various dimensions
    cma_grads = filtered_data.groupby(["CMA/CSD", "DGUID"], observed=True)['Value'].sum().reset_index(name='graduates')
    logger.info(f"cma_grads shape: {cma_grads.shape}")
    logger.info(f"cma_grads sample:\n{cma_grads.head()}")
    isced_grads = filtered_data.groupby("ISCED Level of Education", observed=True)['Value'].sum().reset_index(name='graduates')
    province_grads = filtered_data.groupby("Province or Territory", observed=True)['Value'].sum().reset_index(name='graduates')
    credential_grads = filtered_data.groupby("Credential Type", observed=True)['Value'].sum().reset_index(name='graduates')
    institution_grads = filtered_data.groupby("Institution", observed=True)['Value'].sum().reset_index(name='graduates')
    return (filtered_data.reset_index(), cma_grads, isced_grads, province_grads, credential_grads, institution_grads)

@azure_cache_decorator(ttl=300)
def create_chart(df, x_column, y_column, x_label, selected_value=None):
    """
    Create a horizontal bar chart figure from an aggregated DataFrame.
    Highlights the bar for `selected_value` in red if provided.
    """
    if df is None or df.empty:
        return {}
    # Sort data for consistent order (smallest to largest)
    sorted_df = df.sort_values(y_column, ascending=True)
    sorted_df['text'] = sorted_df[y_column].apply(lambda v: f"{int(v):,}")
    # Determine bar colors – all red scale if no selection, otherwise highlight one
    if selected_value:
        colors = [bc.MAIN_RED if cat == selected_value else bc.LIGHT_GREY for cat in sorted_df[x_column]]
        color_scale = None  # no continuous colorscale when highlighting one
    else:
        colors = sorted_df[y_column]  # use values (will apply a continuous colorscale)
        color_scale = bc.BRIGHT_RED_SCALE  # custom red color scale from brand_colours
    fig = go.Figure(go.Bar(
        x=sorted_df[y_column], y=sorted_df[x_column],
        orientation='h',
        text=sorted_df['text'], textposition='outside', cliponaxis=False,
        marker=dict(color=colors, colorscale=color_scale),
        hovertemplate='%{y}: %{x:,}<extra></extra>'
    ))
    # Layout adjustments for readability
    fig.update_layout(
        showlegend=False, xaxis_title=None, yaxis_title=None,
        font=dict(family='Open Sans', size=12, color=bc.IIC_BLACK),
        plot_bgcolor='#D5DADC', paper_bgcolor='white',
        margin=dict(l=5, r=50, t=25, b=5),
        height=1000 if x_label in ['Institution', 'Census Metropolitan Area'] else 500,
        # Remove unnecessary modebar buttons:
        modebar_remove=['zoom','pan','select','zoomIn','zoomOut','autoScale','resetScale','lasso2d']
    )
    return fig

@azure_cache_decorator(ttl=300)
def filter_options(data, column, selected_filters):
    """
    Compute the available options for `column` given current selections in other filters.
    Returns a list of {'label': ..., 'value': ...} for the dropdown options.
    """
    if data is None or data.empty:
        return []
    mask = pd.Series(True, index=data.index)
    for col, vals in selected_filters.items():
        if vals and col != column:
            mask &= data.index.get_level_values(col).isin(vals)
    available_vals = sorted(data[mask].index.get_level_values(column).unique())
    return [{'label': v, 'value': v} for v in available_vals]
