"""Main entry point for the Infozone application."""

import dash
from dash import Dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import logging
import atexit
import pandas as pd

# Import from our modular structure
from config.settings import APP_CONFIG
from services.cache_service import azure_cache
from services.data_service import load_spatial_data, load_and_process_educational_data, FilterOptimizer
from controllers import register_callbacks
from models.layout import create_app_layout

# Configure logging
logging.basicConfig(level=logging.getLevelName(APP_CONFIG["log_level"]))
logger = logging.getLogger(__name__)

# Register cleanup
atexit.register(azure_cache.clear_cache)

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

# Set React version to ensure compatibility
dash._dash_renderer._set_react_version("18.2.0")

def main():
    """Main function to start the application."""
    try:
        # Load initial data
        logger.info("Loading spatial data...")
        province_longlat_clean, combined_longlat_clean = load_spatial_data()
        
        logger.info("Loading educational data...")
        data = load_and_process_educational_data()
        
        logger.info("Initializing filter optimizer...")
        filter_optimizer = FilterOptimizer(data)
        
        # Generate initial filter options
        logger.info("Preparing filter options...")
        stem_bhase_options = [{'label': stem, 'value': stem} for stem in 
                             sorted(data.index.get_level_values('STEM/BHASE').unique())]
        year_options = [{'label': year, 'value': year} for year in 
                       sorted(data.index.get_level_values('year').unique())]
        prov_options = [{'label': prov, 'value': prov} for prov in 
                       sorted(data.index.get_level_values('Province_Territory').unique())]
        cma_options = [{'label': cma, 'value': cma} for cma in 
                      sorted(data.index.get_level_values('CMA_CA').unique())]
        isced_options = [{'label': level, 'value': level} for level in 
                        sorted(data.index.get_level_values('ISCED_level_of_education').unique())]
        credential_options = [{'label': cred, 'value': cred} for cred in 
                            sorted(data.index.get_level_values('Credential_Type').unique())]
        institution_options = [{'label': inst, 'value': inst} for inst in 
                             sorted(data.index.get_level_values('Institution').unique())]
        
        # Create the app layout using our models
        logger.info("Creating application layout...")
        app.layout = dmc.MantineProvider(
            theme={
                "fontFamily": "'Open Sans', sans-serif",
                "primaryColor": "red",
                "components": {
                    "MultiSelect": {
                        "styles": {
                            "input": {"fontWeight": 600},
                            "item": {"fontWeight": 600}
                        }
                    }
                }
            },
            children=[
                # Include CSS for pivot table
                dash.html.Link(
                    rel='stylesheet',
                    href='/assets/pivottable.css'
                ),
                # Client-side stores for state management
                dash.dcc.Store(id='client-data-store', storage_type='session'),
                dash.dcc.Store(id='client-filters-store', storage_type='local'),
                # Create main layout
                create_app_layout(
                    data=data,
                    stem_bhase_options=stem_bhase_options,
                    year_options=year_options, 
                    prov_options=prov_options,
                    isced_options=isced_options, 
                    credential_options=credential_options,
                    institution_options=institution_options, 
                    cma_options=cma_options
                )
            ]
        )
        
        # Register callbacks using our controller
        logger.info("Registering application callbacks...")
        register_callbacks(app, filter_optimizer, combined_longlat_clean, province_longlat_clean)
        
        # Start the app
        logger.info(f"Starting server with debug={APP_CONFIG['debug']}...")
        app.run_server(debug=APP_CONFIG["debug"])
        
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        raise

if __name__ == '__main__':
    main()

