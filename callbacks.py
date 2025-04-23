# callbacks.py – All Dash callbacks
from dash import callback_context, no_update, dcc
from dash.dependencies import Input, Output, State, MATCH
from dash.exceptions import PreventUpdate
from app import data, combined_longlat_clean#, app
import cache_utils, data_utils
import json, pandas as pd
import brand_colours as bc
import plotly.express as px
from pathlib import Path
#from cache_utils import logger
import dash
import numpy as np

@dash.callback(
    Output('selected-feature', 'data'),
    Input('cma-geojson', 'clickData'),
    Input('clear-selection', 'n_clicks'),
    State('selected-feature', 'data'),
    prevent_initial_call=True
)
def update_selected_feature(click_data, n_clicks, current_selection):
    """Update the stored selected feature (region) based on map clicks or clear button."""
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger == 'clear-selection':
        return None
    if trigger == 'cma-geojson' and click_data:
        # featureId was set in geojson click handler to the DGUID of the feature
        clicked_id = click_data.get('points',[{}])[0].get('featureId')
        return None if current_selection == clicked_id else clicked_id
    return current_selection

@dash.callback(
    Output({'type': 'store', 'item': MATCH}, 'data'),
    Input({'type': 'graph', 'item': MATCH}, 'clickData'),
    Input('clear-selection', 'n_clicks'),
    State({'type': 'store', 'item': MATCH}, 'data'),
    State({'type': 'graph', 'item': MATCH}, 'figure'),
    prevent_initial_call=True
)
def update_chart_selection(clickData, clear_clicks, stored_value, figure):
    """Generic callback to update the selection store for any chart (using pattern matching)."""
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'clear-selection':
        return None
    if clickData and 'points' in clickData:
        # Parse the pattern-matched component ID (which comes as a JSON string)
        try:
            pattern = json.loads(trigger_id.replace("'", "\""))
            # Determine orientation of the clicked bar from the figure
            orientation = 'h'
            if figure and figure.get('data') and figure['data'][0].get('orientation') != 'h':
                orientation = 'v'
            clicked_val = (clickData['points'][0]['y'] if orientation == 'h' else 
                           clickData['points'][0]['x'])
            return None if stored_value == clicked_val else clicked_val
        except Exception as e:
            cache_utils.logger.error(f"Error parsing selection pattern: {e}")
    return stored_value

@dash.callback(
    Output('cma-geojson', 'data'),
    Output({'type': 'graph', 'item': 'isced'}, 'figure'),
    Output({'type': 'graph', 'item': 'province'}, 'figure'),
    Output({'type': 'graph', 'item': 'cma'}, 'figure'),
    Output({'type': 'graph', 'item': 'credential'}, 'figure'),
    Output({'type': 'graph', 'item': 'institution'}, 'figure'),
    Output({'type': 'graph', 'item': 'cip'}, 'figure'),
    Output('map', 'viewport'),
    Input('stem-bhase-filter', 'value'),
    Input('year-filter', 'value'),
    Input('prov-filter', 'value'),
    Input('isced-filter', 'value'),
    Input('credential-filter', 'value'),
    Input('institution-filter', 'value'),
    Input('cma-filter', 'value'),
    Input('cip-filter', 'value'),
    Input({'type': 'store', 'item': 'isced'}, 'data'),
    Input({'type': 'store', 'item': 'province'}, 'data'),
    Input('selected-feature', 'data'),
    Input({'type': 'store', 'item': 'credential'}, 'data'),
    Input({'type': 'store', 'item': 'institution'}, 'data'),
    Input({'type': 'store', 'item': 'cma'}, 'data'),
    Input({'type': 'store', 'item': 'cip'}, 'data'),
    State('map', 'viewport')
)
def update_visualizations(stem_vals, year_vals, prov_vals, isced_vals, cred_vals, inst_vals, cma_vals, cip_vals,
                          sel_isced, sel_prov, selected_feature, sel_cred, sel_inst, sel_cma, sel_cip,
                          current_viewport):
    """
    Main callback to update all visualizations (map and charts) based on filter inputs and any selected items.
    """
    # Determine what triggered the update
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # Normalize pattern IDs (e.g., store selections) for logic
    if trigger_id.startswith('{'):
        try:
            trig = json.loads(trigger_id.replace("'", "\""))
            if trig.get('type') == 'store':
                trigger_id = f"selected-{trig.get('item')}"
        except Exception:
            pass

    # Step 1: Apply primary filters and aggregate data
    (filtered_df, cma_grads_agg, isced_grads, province_grads, credential_grads, institution_grads, cip_grads) = (
        data_utils.preprocess_data(
            tuple(stem_vals or []),
            tuple(year_vals or []),
            tuple(prov_vals or []),
            tuple(isced_vals or []),
            tuple(cred_vals or []),
            tuple(inst_vals or []),
            tuple(cma_vals or []),
            tuple(cip_vals or [])
        )
    )
    # Set DGUID index for faster merge
    cma_grads = cma_grads_agg.set_index('DGUID')

    # Step 2: Apply cross-filters if any selection is active (chart or map selections)
    if any([sel_isced, sel_prov, selected_feature, sel_cred, sel_inst, sel_cma]):
        df = filtered_df  # this is the row-level DataFrame from preprocess_data (reset_index form)
        if sel_isced:
            df = df[df['ISCED Level of Education'] == sel_isced]
        if sel_prov:
            df = df[df['Province or Territory'] == sel_prov]
        if selected_feature:
            df = df[df['DGUID'] == selected_feature]
        if sel_cred:
            df = df[df['Credential Type'] == sel_cred]
        if sel_inst:
            df = df[df['Institution'] == sel_inst]
        if sel_cma:
            df = df[df['CMA/CA/CSD'] == sel_cma]
        if sel_cip:
            df = df[df['CIP Name'] == sel_cip]
        if df.empty:
            return create_empty_response()
        # Recalculate aggregations on this cross-filtered subset
        cma_grads_agg = df.groupby(["CMA/CA/CSD", "DGUID"], observed=True)['Value'].sum().reset_index(name='graduates')
        # Set DGUID index for faster merge
        cma_grads = cma_grads_agg.set_index('DGUID')
        isced_grads = df.groupby("ISCED Level of Education", observed=True)['Value'].sum().reset_index(name='graduates')
        province_grads = df.groupby("Province or Territory", observed=True)['Value'].sum().reset_index(name='graduates')
        credential_grads = df.groupby("Credential Type", observed=True)['Value'].sum().reset_index(name='graduates')
        institution_grads = df.groupby("Institution", observed=True)['Value'].sum().reset_index(name='graduates')
        cip_grads = df.groupby("CIP Name", observed=True)['Value'].sum().reset_index(name='graduates')
        
    # Step 3: Prepare GeoJSON data for the map
    # Merge on indices for efficiency
    cma_data = combined_longlat_clean.merge(cma_grads, left_index=True, right_index=True, how='inner')
    if cma_data.empty:
        return create_empty_response()

    # Determine if we should update map viewport (on filter changes or new selection)
    update_view = trigger_id in ['stem-bhase-filter','year-filter','prov-filter','isced-filter',
                                 'credential-filter','institution-filter', 'cip-filter', 'selected-feature',
                                 'clear-selection','reset-filters']
    if update_view:
        bounds = cma_data.total_bounds  # (minx, miny, maxx, maxy)
        lat_pad = (bounds[3] - bounds[1]) * 0.1
        lon_pad = (bounds[2] - bounds[0]) * 0.1
        new_bounds = [[bounds[1] - lat_pad, bounds[0] - lon_pad],
                      [bounds[3] + lat_pad, bounds[2] + lon_pad]]
        viewport = {'bounds': new_bounds, 'transition': {'duration': 1000}}
    else:
        viewport = no_update

    # --- Optimized GeoJSON Generation ---

    # Ensure DGUID is string for comparisons - Access the index directly
    cma_data['DGUID_str'] = cma_data.index.astype(str)
    selected_feature_str = str(selected_feature) if selected_feature else None

    # 1. Vectorized Color Calculation
    max_val = cma_data['graduates'].max()
    min_val = cma_data['graduates'].min()
    if max_val > min_val:
        # Normalize graduates count (handle potential division by zero)
        norm = (cma_data['graduates'] - min_val) / (max_val - min_val)
        # Use the correct reference to brand_colours
        fill_colors = pd.Series(px.colors.sample_colorscale(bc.BRIGHT_RED_SCALE, norm), index=cma_data.index)
    else:
        # Assign the top color if all values are the same
        fill_colors = pd.Series(bc.BRIGHT_RED_SCALE[-1], index=cma_data.index)

    # 2. Vectorized Style Properties
    line_colors = np.where(cma_data['DGUID_str'] == selected_feature_str, bc.IIC_BLACK, bc.GREY)
    line_weights = np.where(cma_data['DGUID_str'] == selected_feature_str, 2, 0.75)
    fill_opacity = 0.8 # Constant for all features

    # 3. Vectorized Tooltip Generation
    tooltips = (f"<div style='font-family: Open Sans; font-weight:600;'>{name}: {int(grads):,}</div>" 
                for name, grads in zip(cma_data['CMA/CA/CSD'], cma_data['graduates']))

    # 4. Construct Properties Dictionary Series
    # Combine style elements into a dictionary for each feature
    styles = [{"fillColor": fc, "color": lc, "weight": lw, "fillOpacity": fill_opacity} 
              for fc, lc, lw in zip(fill_colors, line_colors, line_weights)]

    # Combine all properties needed for the GeoJSON feature
    properties = [{"graduates": int(grad), "DGUID": dguid_str, "CMA/CA/CSD": name, "style": style, "tooltip": tip} 
                  for grad, dguid_str, name, style, tip in zip(cma_data['graduates'], cma_data['DGUID_str'], cma_data['CMA/CA/CSD'], styles, tooltips)]

    # 5. Build GeoJSON features list using geometry and properties
    # Accessing __geo_interface__ is generally efficient for GeoPandas geometries
    features = [
        {
            'type': 'Feature',
            'geometry': geom.__geo_interface__,
            'properties': prop
        }
        for geom, prop in zip(cma_data.geometry, properties)
    ]

    geojson = {'type': 'FeatureCollection', 'features': features}
    # --- End Optimized GeoJSON Generation ---

    # Step 4: Generate chart figures for each dimension
    fig_isced = data_utils.create_chart(isced_grads, 'ISCED Level of Education', 'graduates', 'ISCED Level of Education', sel_isced)
    fig_province = data_utils.create_chart(province_grads, 'Province or Territory', 'graduates', 'Province/Territory', sel_prov)
    fig_cma = data_utils.create_chart(cma_grads, 'CMA/CA/CSD', 'graduates', 'Census Metropolitan Area', selected_feature)
    fig_credential = data_utils.create_chart(credential_grads, 'Credential Type', 'graduates', 'Credential Type', sel_cred)
    fig_institution = data_utils.create_chart(institution_grads, 'Institution', 'graduates', 'Institution', sel_inst)
    fig_cip = data_utils.create_chart(cip_grads, 'CIP Name', 'graduates', 'CIP Name', sel_cip)
    
    # (Optional) Monitor cache usage for debugging performance
    cache_utils.monitor_cache_usage()

    return geojson, fig_isced, fig_province, fig_cma, fig_credential, fig_institution, fig_cip, viewport

@dash.callback(
    Output('stem-bhase-filter', 'value'),
    Output('year-filter', 'value'),
    Output('prov-filter', 'value'),
    Output('isced-filter', 'value'),
    Output('credential-filter', 'value'),
    Output('institution-filter', 'value'),
    Output('cma-filter', 'value'),
    Output('cip-filter', 'value'),
    Input('reset-filters', 'n_clicks'),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    """Reset all filters to their default (All) values."""
    if not n_clicks:
        raise PreventUpdate
    return (
        [opt['value'] for opt in data_utils.stem_bhase_options_full],
        [opt['value'] for opt in data_utils.year_options_full],
        [], [], [], [], [], []  # empty lists = "All" for multi-select filters
    )

@dash.callback(
    Output('stem-bhase-filter', 'options'),
    Output('year-filter', 'options'),
    Output('prov-filter', 'options'),
    Output('cma-filter', 'options'),
    Output('isced-filter', 'options'),
    Output('credential-filter', 'options'),
    Output('institution-filter', 'options'),
    Output('cip-filter', 'options'),
    Input('stem-bhase-filter', 'value'),
    Input('year-filter', 'value'),
    Input('prov-filter', 'value'),
    Input('isced-filter', 'value'),
    Input('credential-filter', 'value'),
    Input('institution-filter', 'value'),
    Input('cma-filter', 'value'),
    Input('cip-filter', 'value'),
    # Chart selection stores
    Input({'type': 'store', 'item': 'isced'}, 'data'),
    Input({'type': 'store', 'item': 'province'}, 'data'),
    Input('selected-feature', 'data'), # Map selection (DGUID)
    Input({'type': 'store', 'item': 'credential'}, 'data'),
    Input({'type': 'store', 'item': 'institution'}, 'data'),
    Input({'type': 'store', 'item': 'cma'}, 'data'), # Chart selection (Name)
    Input({'type': 'store', 'item': 'cip'}, 'data'),
    Input('clear-selection', 'n_clicks')
)
def update_filter_options(stem_bhase, years, provs, isced, credentials, institutions, cmas, cips,
                         selected_isced, selected_province, selected_feature_dguid,
                         selected_credential, selected_institution, selected_cma_name, selected_cip,
                         clear_clicks):
    """
    Optimized: Updates filter options based on dropdowns and selections.
    Filters the preprocessed subset instead of the full dataset repeatedly.
    """
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # 1. Get the data subset based on primary dropdown filters
    (filtered_df, _, _, _, _, _, _) = data_utils.preprocess_data(
        tuple(stem_bhase or []), tuple(years or []), tuple(provs or []),
        tuple(isced or []), tuple(credentials or []), tuple(institutions or []),
        tuple(cmas or []), tuple(cips or [])
    )

    if filtered_df.empty:
        # If primary filters yield no data, return empty options for multi-selects, full for checklists
        empty_opts = []
        return (data_utils.stem_bhase_options_full, data_utils.year_options_full,
                empty_opts, empty_opts, empty_opts, empty_opts, empty_opts, empty_opts)

    # 2. Define active cross-filters (from chart/map selections)
    cross_filters = {
        'ISCED Level of Education': selected_isced,
        'Province or Territory': selected_province,
        'DGUID': selected_feature_dguid, # Map selection uses DGUID
        'Credential Type': selected_credential,
        'Institution': selected_institution,
        'CMA/CA/CSD': selected_cma_name, # CMA chart selection uses Name
        'CIP Name': selected_cip
    }
    active_cross_filters = {k: v for k, v in cross_filters.items() if v is not None}

    # 3. Helper function to get options for a target column from the subset
    def get_options(target_column):
        temp_df = filtered_df
        # Apply cross-filters, *excluding* the target column's own selection
        filters_to_apply = {k: v for k, v in active_cross_filters.items() if k != target_column}
        
        for col, value in filters_to_apply.items():
            if not temp_df.empty and col in temp_df.columns:
                 # Handle potential mismatch (e.g., DGUID vs CMA Name) - needs adjustment if complex cases arise
                if col == 'DGUID' and target_column == 'CMA/CA/CSD': continue # Don't filter CMA options by DGUID selection
                if col == 'CMA/CA/CSD' and target_column == 'DGUID': continue # Don't filter DGUID options by CMA name selection (though DGUID isn't a dropdown)

                temp_df = temp_df[temp_df[col] == value]
            elif not temp_df.empty:
                 cache_utils.logger.warning(f"Column '{col}' not found in filtered_df for cross-filtering options for '{target_column}'.")


        if not temp_df.empty and target_column in temp_df.columns:
            unique_vals = sorted(temp_df[target_column].unique())
            return [{'label': v, 'value': v} for v in unique_vals]
        else:
            return [] # No options available if df becomes empty or column missing

    # 4. Calculate options for each filter using the helper
    # Checklists (STEM, Year) should ideally show all options unless cross-filtered?
    # For simplicity now, let's filter them too, but could revert to full list if needed.
    stem_options = get_options('STEM/BHASE')
    year_options = get_options('Academic Year')
    prov_options = get_options('Province or Territory')
    cma_options = get_options('CMA/CA/CSD')
    isced_options = get_options('ISCED Level of Education')
    cred_options = get_options('Credential Type')
    inst_options = get_options('Institution')
    cip_options = get_options('CIP Name')

    # Ensure checklists always have options if the initial filtered_df wasn't empty
    # This might be desired UX - decide based on requirements.
    # if not stem_options and not filtered_df.empty: stem_options = data_utils.stem_bhase_options_full
    # if not year_options and not filtered_df.empty: year_options = data_utils.year_options_full


    return (
        stem_options, year_options, prov_options, cma_options,
        isced_options, cred_options, inst_options, cip_options
    )

@dash.callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    State("pivot-table", "data"),
    State("pivot-table", "cols"),
    State("pivot-table", "rows"),
    State("pivot-table", "vals"),
    prevent_initial_call=True,
)
def download_pivot_data(n_clicks, records, cols, rows, vals):
    """Generate a CSV download of the current pivot table view."""
    if not n_clicks or records is None:
        raise PreventUpdate
    try:
        df = pd.DataFrame(records)
        pivot_df = df.pivot_table(index=rows or None, columns=cols or None, values=vals, aggfunc='sum', fill_value=0)
        csv_string = pivot_df.to_csv(index=True)
        return dcc.send_string(csv_string, filename=f"graduates_pivot_{__import__('time').strftime('%Y%m%d_%H%M%S')}.csv")
    except Exception as e:
        cache_utils.logger.error(f"Error generating pivot CSV: {e}")
        raise PreventUpdate

# Callbacks for toggling UI elements (nav collapse, modals) remain largely unchanged:
@dash.callback(
    Output("horizontal-collapse", "is_open"),
    Input("horizontal-collapse-button", "n_clicks"),
    State("horizontal-collapse", "is_open")
)
def toggle_collapse(n, is_open):
    return not is_open if n else is_open

def toggle_navbar_collapse(n, is_open):
    return not is_open if n else is_open

dash.callback(Output("navbar-collapse", "is_open"),
             Input("navbar-toggler", "n_clicks"),
             State("navbar-collapse", "is_open"))(toggle_navbar_collapse)

@dash.callback(
    Output("user-guide-modal", "is_open"),
    Output("user-guide-content", "children"),
    Input("open-guide-button", "n_clicks"),
    Input("close-guide-button", "n_clicks"),
    State("user-guide-modal", "is_open")
)
def toggle_user_guide(open_clicks, close_clicks, is_open):
    if not (open_clicks or close_clicks):
        raise PreventUpdate
    if (open_clicks or close_clicks) and not is_open:
        # Load guide content on open
        try:
            guide_md = Path("user_guide.md").read_text(encoding="utf-8")
        except Exception:
            guide_md = "User guide not available."
        return True, dcc.Markdown(guide_md)
    return False if is_open else is_open, no_update

@dash.callback(
    Output("faq-modal", "is_open"),
    Input("open-faq-button", "n_clicks"),
    Input("close-faq-button", "n_clicks"),
    State("faq-modal", "is_open")
)
def toggle_faq(open_clicks, close_clicks, is_open):
    if not (open_clicks or close_clicks):
        raise PreventUpdate
    return (not is_open) if (open_clicks or close_clicks) else is_open

dash.clientside_callback(
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

def create_empty_response():
    """Produce empty outputs (geojson, 5 figs, viewport) for no-data cases."""
    empty_geojson = {'type': 'FeatureCollection', 'features': []}
    empty_fig = {}
    default_bounds = [[41, -141], [83, -52]]  # Canada bounding box
    default_viewport = {'bounds': default_bounds, 'transition': "flyToBounds"}
    return (empty_geojson, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, default_viewport)
