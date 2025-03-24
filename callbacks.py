# callbacks.py – All Dash callbacks
from dash import callback_context, no_update, dcc
from dash.dependencies import Input, Output, State, MATCH
from dash.exceptions import PreventUpdate
#from app import app, data, combined_longlat_clean
import cache_utils, data_utils
import json, pandas as pd
import brand_colours as bc
import plotly.express as px
from pathlib import Path
from cache_utils import logger
import dash

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

# @app.callback(
#     Output({'type': 'store', 'item': MATCH}, 'data'),
#     Input({'type': 'graph', 'item': MATCH}, 'clickData'),
#     Input('clear-selection', 'n_clicks'),
#     State({'type': 'store', 'item': MATCH}, 'data'),
#     State({'type': 'graph', 'item': MATCH}, 'figure'),
#     prevent_initial_call=True
# )
# def update_chart_selection(clickData, clear_clicks, stored_value, figure):
#     """Generic callback to update the selection store for any chart (using pattern matching)."""
#     ctx = callback_context
#     if not ctx.triggered:
#         raise PreventUpdate
#     trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
#     if trigger_id == 'clear-selection':
#         return None
#     if clickData and 'points' in clickData:
#         # Parse the pattern-matched component ID (which comes as a JSON string)
#         try:
#             pattern = json.loads(trigger_id.replace("'", "\""))
#             # Determine orientation of the clicked bar from the figure
#             orientation = 'h'
#             if figure and figure.get('data') and figure['data'][0].get('orientation') != 'h':
#                 orientation = 'v'
#             clicked_val = (clickData['points'][0]['y'] if orientation == 'h' else 
#                            clickData['points'][0]['x'])
#             return None if stored_value == clicked_val else clicked_val
#         except Exception as e:
#             cache_utils.logger.error(f"Error parsing selection pattern: {e}")
#     return stored_value

# @app.callback(
#     Output('cma-geojson', 'data'),
#     Output({'type': 'graph', 'item': 'isced'}, 'figure'),
#     Output({'type': 'graph', 'item': 'province'}, 'figure'),
#     Output({'type': 'graph', 'item': 'cma'}, 'figure'),
#     Output({'type': 'graph', 'item': 'credential'}, 'figure'),
#     Output({'type': 'graph', 'item': 'institution'}, 'figure'),
#     Output('map', 'viewport'),
#     Input('stem-bhase-filter', 'value'),
#     Input('year-filter', 'value'),
#     Input('prov-filter', 'value'),
#     Input('isced-filter', 'value'),
#     Input('credential-filter', 'value'),
#     Input('institution-filter', 'value'),
#     Input('cma-filter', 'value'),
#     Input({'type': 'store', 'item': 'isced'}, 'data'),
#     Input({'type': 'store', 'item': 'province'}, 'data'),
#     Input('selected-feature', 'data'),
#     Input({'type': 'store', 'item': 'credential'}, 'data'),
#     Input({'type': 'store', 'item': 'institution'}, 'data'),
#     Input({'type': 'store', 'item': 'cma'}, 'data'),
#     State('map', 'viewport')
# )
# def update_visualizations(stem_vals, year_vals, prov_vals, isced_vals, cred_vals, inst_vals, cma_vals,
#                           sel_isced, sel_prov, selected_feature, sel_cred, sel_inst, sel_cma,
#                           current_viewport):
#     """
#     Main callback to update all visualizations (map and charts) based on filter inputs and any selected items.
#     """
#     # Determine what triggered the update
#     ctx = callback_context
#     if not ctx.triggered:
#         raise PreventUpdate
#     trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
#     # Normalize pattern IDs (e.g., store selections) for logic
#     if trigger_id.startswith('{'):
#         try:
#             trig = json.loads(trigger_id.replace("'", "\""))
#             if trig.get('type') == 'store':
#                 trigger_id = f"selected-{trig.get('item')}"
#         except Exception:
#             pass

#     # Step 1: Apply primary filters and aggregate data
#     (filtered_df, cma_grads, isced_grads, province_grads, credential_grads, institution_grads) = (
#         data_utils.preprocess_data(
#             tuple(stem_vals or []),
#             tuple(year_vals or []),
#             tuple(prov_vals or []),
#             tuple(isced_vals or []),
#             tuple(cred_vals or []),
#             tuple(inst_vals or []),
#             tuple(cma_vals or [])
#         )
#     )

#     # Step 2: Apply cross-filters if any selection is active (chart or map selections)
#     if any([sel_isced, sel_prov, selected_feature, sel_cred, sel_inst, sel_cma]):
#         df = filtered_df  # this is the row-level DataFrame from preprocess_data (reset_index form)
#         if sel_isced:
#             df = df[df['ISCED Level of Education'] == sel_isced]
#         if sel_prov:
#             df = df[df['Province or Territory'] == sel_prov]
#         if selected_feature:
#             df = df[df['DGUID'] == selected_feature]
#         if sel_cred:
#             df = df[df['Credential Type'] == sel_cred]
#         if sel_inst:
#             df = df[df['Institution'] == sel_inst]
#         if sel_cma:
#             df = df[df['CMA/CSD'] == sel_cma]
#         if df.empty:
#             return create_empty_response()
#         # Recalculate aggregations on this cross-filtered subset
#         cma_grads = df.groupby(["CMA/CSD", "DGUID"], observed=True)['Value'].sum().reset_index(name='graduates')
#         isced_grads = df.groupby("ISCED Level of Education", observed=True)['Value'].sum().reset_index(name='graduates')
#         province_grads = df.groupby("Province or Territory", observed=True)['Value'].sum().reset_index(name='graduates')
#         credential_grads = df.groupby("Credential Type", observed=True)['Value'].sum().reset_index(name='graduates')
#         institution_grads = df.groupby("Institution", observed=True)['Value'].sum().reset_index(name='graduates')

#     # Step 3: Prepare GeoJSON data for the map
#     # Merge geometry with aggregated data (inner join to include only regions with data)
#     logger.info(f"combined_longlat_clean shape: {combined_longlat_clean.shape}")
#     logger.info(f"combined_longlat_clean sample:\n{combined_longlat_clean.head()}")
#     cma_data = combined_longlat_clean.merge(cma_grads, on='DGUID', how='inner')
#     logger.info(f"cma_data shape after merge: {cma_data.shape}")
#     logger.info(f"cma_data sample:\n{cma_data.head()}")
#     if cma_data.empty:
#         return create_empty_response()
#     # Determine if we should update map viewport (on filter changes or new selection)
#     update_view = trigger_id in ['stem-bhase-filter','year-filter','prov-filter','isced-filter',
#                                  'credential-filter','institution-filter','selected-feature',
#                                  'clear-selection','reset-filters']
#     if update_view:
#         bounds = cma_data.total_bounds  # (minx, miny, maxx, maxy)
#         lat_pad = (bounds[3] - bounds[1]) * 0.1
#         lon_pad = (bounds[2] - bounds[0]) * 0.1
#         new_bounds = [[bounds[1] - lat_pad, bounds[0] - lon_pad],
#                       [bounds[3] + lat_pad, bounds[2] + lon_pad]]
#         viewport = {'bounds': new_bounds, 'transition': {'duration': 1000}}
#     else:
#         viewport = no_update

#     # Ensure geometry JSON is computed (to avoid repeated shapely -> dict conversion)
#     if 'geometry_json' not in combined_longlat_clean:
#         combined_longlat_clean['geometry_json'] = combined_longlat_clean.geometry.apply(lambda geom: geom.__geo_interface__)
#     # Assign colors to each region based on graduates count
#     max_val = cma_data['graduates'].max()
#     min_val = cma_data['graduates'].min()
#     if max_val > min_val:
#         norm = (cma_data['graduates'] - min_val) / (max_val - min_val)
#         colors = px.colors.sample_colorscale(data_utils.bc.BRIGHT_RED_SCALE if hasattr(data_utils, 'bc') else bc.BRIGHT_RED_SCALE, norm)
#     else:
#         colors = [bc.BRIGHT_RED_SCALE[-1]] * len(cma_data)
#     # Build GeoJSON features list
#     features = []
#     for (_, row), color in zip(cma_data.iterrows(), colors):
#         features.append({
#             'type': 'Feature',
#             'geometry': row.geometry.__geo_interface__,  # use geometry_json if desired
#             'properties': {
#                 'graduates': int(row['graduates']),
#                 'DGUID': str(row['DGUID']),
#                 'CMA/CSD': row['CMA/CSD'],
#                 'style': {
#                     'fillColor': color,
#                     'color': bc.IIC_BLACK if str(row['DGUID']) == selected_feature else bc.GREY,
#                     'weight': 2 if str(row['DGUID']) == selected_feature else 0.75,
#                     'fillOpacity': 0.8
#                 },
#                 'tooltip': f"<div style='font-family: Open Sans; font-weight:600;'>{row['CMA/CSD']}: {int(row['graduates']):,}</div>"
#             }
#         })
#     geojson = {'type': 'FeatureCollection', 'features': features}

#     # Step 4: Generate chart figures for each dimension
#     fig_isced = data_utils.create_chart(isced_grads, 'ISCED Level of Education', 'graduates', 'ISCED Level of Education', sel_isced)
#     fig_province = data_utils.create_chart(province_grads, 'Province or Territory', 'graduates', 'Province/Territory', sel_prov)
#     fig_cma = data_utils.create_chart(cma_grads, 'CMA/CSD', 'graduates', 'Census Metropolitan Area', selected_feature)
#     fig_credential = data_utils.create_chart(credential_grads, 'Credential Type', 'graduates', 'Credential Type', sel_cred)
#     fig_institution = data_utils.create_chart(institution_grads, 'Institution', 'graduates', 'Institution', sel_inst)

#     # (Optional) Monitor cache usage for debugging performance
#     cache_utils.monitor_cache_usage()

#     return geojson, fig_isced, fig_province, fig_cma, fig_credential, fig_institution, viewport

# @app.callback(
#     Output('stem-bhase-filter', 'value'),
#     Output('year-filter', 'value'),
#     Output('prov-filter', 'value'),
#     Output('isced-filter', 'value'),
#     Output('credential-filter', 'value'),
#     Output('institution-filter', 'value'),
#     Output('cma-filter', 'value'),
#     Input('reset-filters', 'n_clicks'),
#     prevent_initial_call=True
# )
# def reset_filters(n_clicks):
#     """Reset all filters to their default (All) values."""
#     if not n_clicks:
#         raise PreventUpdate
#     return (
#         [opt['value'] for opt in data_utils.stem_bhase_options_full],
#         [opt['value'] for opt in data_utils.year_options_full],
#         [], [], [], [], []  # empty lists = "All" for multi-select filters
#     )

# @app.callback(
#     Output('stem-bhase-filter', 'options'),
#     Output('year-filter', 'options'),
#     Output('prov-filter', 'options'),
#     Output('cma-filter', 'options'),
#     Output('isced-filter', 'options'),
#     Output('credential-filter', 'options'),
#     Output('institution-filter', 'options'),
#     Input('stem-bhase-filter', 'value'),
#     Input('year-filter', 'value'),
#     Input('prov-filter', 'value'),
#     Input('isced-filter', 'value'),
#     Input('credential-filter', 'value'),
#     Input('institution-filter', 'value'),
#     Input('cma-filter', 'value')
# )
# def update_filter_options(stem_vals, year_vals, prov_vals, isced_vals, cred_vals, inst_vals, cma_vals):
#     """Update filter dropdown options based on the current selections (cascading filters)."""
#     ctx = callback_context
#     if not ctx.triggered:
#         raise PreventUpdate
#     current = {
#         'STEM/BHASE': stem_vals or [],
#         'Academic Year': year_vals or [],
#         'Province or Territory': prov_vals or [],
#         'CMA/CSD': cma_vals or [],
#         'ISCED Level of Education': isced_vals or [],
#         'Credential Type': cred_vals or [],
#         'Institution': inst_vals or []
#     }
#     stem_options = data_utils.filter_options(data, 'STEM/BHASE', {k: v for k, v in current.items() if k != 'STEM/BHASE'})
#     year_options = data_utils.filter_options(data, 'Academic Year', {k: v for k, v in current.items() if k != 'Academic Year'})
#     prov_options = data_utils.filter_options(data, 'Province or Territory', {k: v for k, v in current.items() if k != 'Province or Territory'})
#     cma_options = data_utils.filter_options(data, 'CMA/CSD', {k: v for k, v in current.items() if k != 'CMA/CSD'})
#     isced_options = data_utils.filter_options(data, 'ISCED Level of Education', {k: v for k, v in current.items() if k != 'ISCED Level of Education'})
#     cred_options = data_utils.filter_options(data, 'Credential Type', {k: v for k, v in current.items() if k != 'Credential Type'})
#     inst_options = data_utils.filter_options(data, 'Institution', {k: v for k, v in current.items() if k != 'Institution'})
#     return stem_options, year_options, prov_options, cma_options, isced_options, cred_options, inst_options

# @app.callback(
#     Output("download-data", "data"),
#     Input("download-button", "n_clicks"),
#     State("pivot-table", "data"),
#     State("pivot-table", "cols"),
#     State("pivot-table", "rows"),
#     State("pivot-table", "vals"),
#     prevent_initial_call=True,
# )
# def download_pivot_data(n_clicks, records, cols, rows, vals):
#     """Generate a CSV download of the current pivot table view."""
#     if not n_clicks or records is None:
#         raise PreventUpdate
#     try:
#         df = pd.DataFrame(records)
#         pivot_df = df.pivot_table(index=rows or None, columns=cols or None, values=vals, aggfunc='sum', fill_value=0)
#         csv_string = pivot_df.to_csv(index=True)
#         return dcc.send_string(csv_string, filename=f"graduates_pivot_{__import__('time').strftime('%Y%m%d_%H%M%S')}.csv")
#     except Exception as e:
#         cache_utils.logger.error(f"Error generating pivot CSV: {e}")
#         raise PreventUpdate

# # Callbacks for toggling UI elements (nav collapse, modals) remain largely unchanged:
# @app.callback(
#     Output("horizontal-collapse", "is_open"),
#     Input("horizontal-collapse-button", "n_clicks"),
#     State("horizontal-collapse", "is_open")
# )
# def toggle_collapse(n, is_open):
#     return not is_open if n else is_open

# def toggle_navbar_collapse(n, is_open):
#     return not is_open if n else is_open

# app.callback(Output("navbar-collapse", "is_open"),
#              Input("navbar-toggler", "n_clicks"),
#              State("navbar-collapse", "is_open"))(toggle_navbar_collapse)

# @app.callback(
#     Output("user-guide-modal", "is_open"),
#     Output("user-guide-content", "children"),
#     Input("open-guide-button", "n_clicks"),
#     Input("close-guide-button", "n_clicks"),
#     State("user-guide-modal", "is_open")
# )
# def toggle_user_guide(open_clicks, close_clicks, is_open):
#     if not (open_clicks or close_clicks):
#         raise PreventUpdate
#     if (open_clicks or close_clicks) and not is_open:
#         # Load guide content on open
#         try:
#             guide_md = Path("user_guide.md").read_text(encoding="utf-8")
#         except Exception:
#             guide_md = "User guide not available."
#         return True, dcc.Markdown(guide_md)
#     return False if is_open else is_open, no_update

# @app.callback(
#     Output("faq-modal", "is_open"),
#     Input("open-faq-button", "n_clicks"),
#     Input("close-faq-button", "n_clicks"),
#     State("faq-modal", "is_open")
# )
# def toggle_faq(open_clicks, close_clicks, is_open):
#     if not (open_clicks or close_clicks):
#         raise PreventUpdate
#     return (not is_open) if (open_clicks or close_clicks) else is_open

# def create_empty_response():
#     """Produce empty outputs (geojson, 5 figs, viewport) for no-data cases."""
#     empty_geojson = {'type': 'FeatureCollection', 'features': []}
#     empty_fig = {}
#     default_bounds = [[41, -141], [83, -52]]  # Canada bounding box
#     default_viewport = {'bounds': default_bounds, 'transition': "flyToBounds"}
#     return (empty_geojson, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, default_viewport)
