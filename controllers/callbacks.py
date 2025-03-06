"""Controllers for application callbacks."""

import pandas as pd
import numpy as np
import json
from dash import callback_context, Output, Input, State
import dash_leaflet as dl
import colorlover as cl

from utils.helper_utils import MapState, CallbackContextManager, update_selected_value, calculate_viewport_update
from services.data_service import preprocess_data, filter_options
from services.visualization_service import create_chart, create_color_scale, create_geojson_feature, create_empty_response

# Initialize map state
map_state = MapState()

def register_callbacks(app, filter_optimizer, gdf_combined, gdf_province):
    """
    Register all Dash callbacks for the application.
    
    Args:
        app: The Dash application instance
        filter_optimizer: The FilterOptimizer instance for data filtering
        gdf_combined: GeoDataFrame with combined spatial data
        gdf_province: GeoDataFrame with province spatial data
    """
    
    @app.callback(
        [
            # Map outputs
            Output("cma-geojson", "data"),
            # Chart outputs
            Output("graph-isced", "figure"),
            Output("graph-province", "figure"),
            Output("graph-cma", "figure"),
            Output("graph-credential", "figure"),
            Output("graph-institution", "figure"),
            # Map viewport
            Output("map", "viewport"),
            # Store outputs
            Output("selected-isced", "data"),
            Output("selected-province", "data"),
            Output("selected-cma", "data"),
            Output("selected-credential", "data"),
            Output("selected-institution", "data"),
        ],
        [
            # Filter inputs
            Input("stem-bhase-filter", "value"),
            Input("year-filter", "value"),
            Input("prov-filter", "value"),
            Input("isced-filter", "value"),
            Input("credential-filter", "value"),
            Input("institution-filter", "value"),
            Input("cma-filter", "value"),
            # Selection inputs
            Input("graph-isced", "clickData"),
            Input("graph-province", "clickData"),
            Input("graph-cma", "clickData"),
            Input("graph-credential", "clickData"),
            Input("graph-institution", "clickData"),
            Input("cma-geojson", "clickData"),
            Input("clear-selection", "n_clicks"),
            # Store inputs
            State("selected-isced", "data"),
            State("selected-province", "data"),
            State("selected-feature", "data"),
            State("selected-cma", "data"),
            State("selected-credential", "data"),
            State("selected-institution", "data"),
            # Reset filters
            Input("reset-filters", "n_clicks"),
        ],
    )
    def update_data_and_charts(
        selected_stem_bhase, selected_years, selected_provs, selected_isced, selected_credentials,
        selected_institutions, selected_cmas, isced_click, prov_click, cma_click, credential_click,
        institution_click, feature_click, clear_clicks, stored_isced, stored_prov, stored_feature,
        stored_cma, stored_credential, stored_institution, reset_clicks
    ):
        # Get callback context to determine which input triggered the callback
        ctx = CallbackContextManager(callback_context)
        
        # Handle reset filters
        if ctx.triggered_id == "reset-filters":
            # Reset all filters to their defaults (all selected)
            selected_stem_bhase = ["STEM", "BHASE"]
            selected_years = ["2015/2016", "2016/2017", "2017/2018", "2018/2019", "2019/2020"]
            selected_provs = []
            selected_isced = []
            selected_credentials = []
            selected_institutions = []
            selected_cmas = []
            
            # Clear all selections
            stored_isced = None
            stored_prov = None
            stored_feature = None
            stored_cma = None
            stored_credential = None
            stored_institution = None
        
        # Process data based on filters
        (
            filtered_data,
            cma_grads,
            isced_grads,
            province_grads,
            credential_grads,
            institution_grads,
        ) = preprocess_data(
            filter_optimizer,
            selected_stem_bhase,
            selected_years,
            selected_provs,
            selected_isced,
            selected_credentials,
            selected_institutions,
            selected_cmas,
        )
        
        # If no data matches filters, return empty response
        if filtered_data.empty or cma_grads.empty:
            empty_response = create_empty_response()
            return (*empty_response, None, None, None, None, None, None)
        
        # Update selections based on click events
        stored_isced = update_selected_value(
            isced_click, clear_clicks, stored_isced, ctx.triggered_id, "clear-selection", "graph-isced"
        )
        stored_prov = update_selected_value(
            prov_click, clear_clicks, stored_prov, ctx.triggered_id, "clear-selection", "graph-province"
        )
        stored_cma = update_selected_value(
            cma_click, clear_clicks, stored_cma, ctx.triggered_id, "clear-selection", "graph-cma"
        )
        stored_credential = update_selected_value(
            credential_click, clear_clicks, stored_credential, ctx.triggered_id, "clear-selection", "graph-credential"
        )
        stored_institution = update_selected_value(
            institution_click, clear_clicks, stored_institution, ctx.triggered_id, "clear-selection", "graph-institution"
        )
        
        # Update feature selection from map clicks
        if ctx.triggered_id == "cma-geojson" and feature_click:
            stored_feature = feature_click.get("feature")
            # If clicked on already selected feature, deselect it
            if stored_feature == feature_click.get("feature"):
                stored_feature = None
        
        # Clear selection if clear button clicked
        if ctx.triggered_id == "clear-selection":
            stored_feature = None
        
        # Create charts based on filtered data
        isced_fig = create_chart(isced_grads, "ISCED_level_of_education", "graduates", "ISCED Level", stored_isced)
        province_fig = create_chart(province_grads, "Province_Territory", "graduates", "Province", stored_prov)
        cma_fig = create_chart(cma_grads, "CMA_CA", "graduates", "Census Metropolitan Area", stored_cma)
        credential_fig = create_chart(credential_grads, "Credential_Type", "graduates", "Credential Type", stored_credential)
        institution_fig = create_chart(institution_grads, "Institution", "graduates", "Institution", stored_institution)
        
        # Generate GeoJSON for map
        # Merge spatial data with graduate counts
        cma_spatial = pd.merge(
            gdf_combined,
            cma_grads,
            left_on=["DGUID"],
            right_on=["DGUID"],
            how="inner"
        )
        
        if cma_spatial.empty:
            empty_response = create_empty_response()
            return (*empty_response, None, None, None, None, None, None)
        
        # Calculate color scale based on graduate counts
        max_graduates = cma_spatial["graduates"].max()
        min_graduates = cma_spatial["graduates"].min()
        colorscale = cl.scales["9"]["seq"]["Reds"]
        
        # Create GeoJSON features
        features = []
        for _, row in cma_spatial.iterrows():
            feature = {
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": create_geojson_feature(
                    row, colorscale, max_graduates, min_graduates, stored_feature
                ),
            }
            features.append(feature)
        
        geojson_data = {"type": "FeatureCollection", "features": features}
        
        # Calculate viewport updates if needed
        viewport_update = calculate_viewport_update(
            ctx.triggered_id, cma_spatial, stored_feature
        )
        
        if viewport_update:
            map_viewport = {
                "center": viewport_update["center"],
                "zoom": viewport_update["zoom"],
                "transition": "flyTo"
            }
        else:
            map_viewport = None
        
        return (
            geojson_data,
            isced_fig,
            province_fig,
            cma_fig,
            credential_fig,
            institution_fig,
            map_viewport,
            stored_isced,
            stored_prov,
            stored_cma,
            stored_credential,
            stored_institution,
        )
    
    @app.callback(
        Output("horizontal-collapse", "is_open"),
        Input("horizontal-collapse-button", "n_clicks"),
        State("horizontal-collapse", "is_open"),
    )
    def toggle_collapse(n, is_open):
        """Toggle collapse of the filters panel"""
        if n:
            return not is_open
        return is_open
    
    @app.callback(
        Output("download-data", "data"),
        Input("download-button", "n_clicks"),
        State("pivot-table", "tableData"),
        prevent_initial_call=True,
    )
    def download_data(n_clicks, table_data):
        """Download pivot table data as CSV"""
        if n_clicks is None or table_data is None:
            return None
        
        try:
            # Convert table data to dataframe and CSV
            df = pd.DataFrame(table_data)
            return dict(
                content=df.to_csv(index=False),
                filename="graduates_data.csv",
            )
        except Exception as e:
            print(f"Error during download: {e}")
            return None