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

# Add Clarity tracking script to the head
app.index_string = '''<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script type="text/javascript">
            (function(c,l,a,r,i,t,y){
                c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
                t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
                y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
            })(window, document, "clarity", "script", "r5p3eqn5xb");
        </script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>'''

cache_utils.initialize_cache()

# Load initial data
combined_longlat_clean = cache_utils.azure_cache_decorator(ttl=3600)(gpd.read_parquet)("data/combined_longlat_simplified.parquet")
# Set DGUID as index for faster merges, keep it as a column too
if 'DGUID' in combined_longlat_clean.columns and not combined_longlat_clean.index.name == 'DGUID':
    combined_longlat_clean = combined_longlat_clean.set_index('DGUID', drop=False)
data = cache_utils.azure_cache_decorator(ttl=3600)(pd.read_parquet)("data/cleaned_data.parquet")

# Rename Quebec to Québec if necessary
if 'Province or Territory' in data.columns:
    data['Province or Territory'] = data['Province or Territory'].replace('Quebec', 'Québec')
elif 'Province or Territory' in data.index.names:
    # Handle case where it might be in the index after loading
    data.index = data.index.set_levels(data.index.levels[data.index.names.index('Province or Territory')].str.replace('Quebec', 'Québec'), level='Province or Territory')

#print("Main DataFrame columns:", data.index.names if data.index.nlevels > 1 else data.columns)
#print("GeoDataFrame columns:", combined_longlat_clean.columns)

# Ensure categorical types for filters:
for col in ["STEM/BHASE","Academic Year","Province or Territory","ISCED Level of Education",
           "Credential Type","Institution","CMA/CA/CSD","DGUID","CIP Name"]:
    data[col] = data[col].astype('category')
data['Value'] = data['Value'].astype('float32')
data = data.set_index(["STEM/BHASE","Academic Year","Province or Territory","ISCED Level of Education",
                       "Credential Type","Institution","CMA/CA/CSD","DGUID","CIP Name"]).sort_index()

# Set up filter optimizer with loaded data
data_utils.filter_optimizer = data_utils.FilterOptimizer(data)
# Compute full filter option lists for initial layout
stem_bhase_options_full = [{'label': v, 'value': v} for v in sorted(data.index.get_level_values('STEM/BHASE').unique())]
year_options_full = [{'label': v, 'value': v} for v in sorted(data.index.get_level_values('Academic Year').unique())]
prov_options_full = [{'label': v, 'value': v} for v in sorted(data.index.get_level_values('Province or Territory').unique())]
cma_options_full = [{'label': v, 'value': v} for v in sorted(data.index.get_level_values('CMA/CA/CSD').unique())]
isced_options_full = [{'label': v, 'value': v} for v in sorted(data.index.get_level_values('ISCED Level of Education').unique())]
credential_options_full = [{'label': v, 'value': v} for v in sorted(data.index.get_level_values('Credential Type').unique())]
institution_options_full = [{'label': v, 'value': v} for v in sorted(data.index.get_level_values('Institution').unique())]
cip_options_full = [{'label': v, 'value': v} for v in sorted(data.index.get_level_values('CIP Name').unique())]

# Save options in data_utils (for use in reset_filters callback)
data_utils.stem_bhase_options_full = stem_bhase_options_full
data_utils.year_options_full = year_options_full
data_utils.prov_options_full = prov_options_full
data_utils.cma_options_full = cma_options_full
data_utils.isced_options_full = isced_options_full
data_utils.credential_options_full = credential_options_full
data_utils.institution_options_full = institution_options_full
data_utils.cip_options_full = cip_options_full

app.layout = html.Div([
    html.Link(
        rel='stylesheet',
        href='/assets/assets/pivottable.css'
    ),
    dcc.Store(id='client-data-store', storage_type='session'),
    dcc.Store(id='client-filters-store', storage_type='local'),
    create_layout(data, stem_bhase_options_full, year_options_full, prov_options_full, isced_options_full, 
                 credential_options_full, institution_options_full, cma_options_full, cip_options_full)
])

import callbacks

if __name__ == '__main__':
    app.run_server(debug=False)
