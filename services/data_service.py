"""Data service for loading and processing educational data."""

import pandas as pd
import geopandas as gpd
import logging
import numpy as np
import hashlib
from functools import lru_cache
from services.cache_service import azure_cache_decorator

# Configure logging
logger = logging.getLogger(__name__)

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
    
    categorical_cols = ["STEM/BHASE", "year", "Province_Territory", "ISCED_level_of_education", 
                        "Credential_Type", "Institution", "CMA_CA", "DGUID"]
    for col in categorical_cols:
        data[col] = data[col].astype('category')
    data['value'] = data['value'].astype('float32')

    # Set a multi-index for direct indexing
    data = data.set_index(["STEM/BHASE", "year", "Province_Territory", "ISCED_level_of_education", 
                           "Credential_Type", "Institution", "CMA_CA", "DGUID"]).sort_index()

    return data

class FilterOptimizer:
    """Optimized data filtering with caching and vectorized operations."""
    
    def __init__(self, data):
        """
        Initialize the FilterOptimizer with data.
        
        Args:
            data (pandas.DataFrame): The dataset to filter
        """
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
    
    def filter_data(self, filters):
        """
        Optimized filtering with caching and vectorized operations.
        
        Args:
            filters (dict): Dictionary of column names and allowed values
            
        Returns:
            pandas.DataFrame: The filtered dataset
        """
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

@azure_cache_decorator(ttl=300)  # 5 minutes - results depend on user filters
def preprocess_data(filter_optimizer, selected_stem_bhase, selected_years, selected_provs, selected_isced, 
                   selected_credentials, selected_institutions, selected_cmas):
    """
    Optimized data preprocessing with caching and vectorized operations.

    Args:
        filter_optimizer (FilterOptimizer): Optimizer instance for filtering
        selected_stem_bhase (tuple): Selected STEM/BHASE categories
        selected_years (tuple): Selected academic years
        selected_provs (tuple): Selected provinces/territories
        selected_isced (tuple): Selected ISCED education levels
        selected_credentials (tuple): Selected credential types
        selected_institutions (tuple): Selected institutions
        selected_cmas (tuple): Selected CMA/CA regions

    Returns:
        tuple: A tuple containing:
               - Filtered data (DataFrame)
               - CMA graduates data (DataFrame)
               - ISCED graduates data (DataFrame)
               - Province graduates data (DataFrame)
               - Credential graduates data (DataFrame)
               - Institution graduates data (DataFrame)
    """
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