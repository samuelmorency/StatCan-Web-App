"""Layout model for Infozone application."""

import dash_bootstrap_components as dbc
from dash import html, dcc
import brand_colours as bc
from config.settings import UI_CONFIG
from components import (
    create_navbar, create_map_component, create_chart_card,
    create_scrollable_chart_card, create_filters_section
)

def create_app_layout(data, stem_bhase_options, year_options, prov_options, 
                     isced_options, credential_options, institution_options, cma_options):
    """
    Creates the main application layout with tabs, charts, and filters.
    
    Args:
        data: Dataset for the pivot table
        stem_bhase_options: Options for the STEM/BHASE filter
        year_options: Options for the year filter
        prov_options: Options for the province filter
        isced_options: Options for the ISCED filter
        credential_options: Options for the credential filter
        institution_options: Options for the institution filter
        cma_options: Options for the CMA filter
        
    Returns:
        dash.html.Div: The complete application layout
    """
    # Create navbar
    logo = create_navbar()
    
    # Create filters section
    filters_button, filters_section = create_filters_section(
        stem_bhase_options, year_options, prov_options, 
        isced_options, credential_options, institution_options, cma_options
    )
    
    # Create map card
    map_card = dbc.Card([
        dbc.CardHeader("Number of Graduates by CMA/CA", style={
            "font-family": 'Open Sans',
            "font-weight": "600",
            "font-size": "16px"
        }),
        dbc.CardBody([
            dbc.Spinner(
                create_map_component(),
                color="primary",
                type="border",
            ),
            dcc.Store(id='selected-feature', data=None),
        ])
    ], className="mb-2 mt-4")
    
    # Create chart cards
    isced_card = create_chart_card("graph-isced", "Number of Graduates by ISCED Level of Education")
    province_card = create_chart_card("graph-province", "Number of Graduates by Province")
    cma_card = create_scrollable_chart_card("graph-cma", "Number of Graduates by CMA/CA")
    credential_card = create_chart_card("graph-credential", "Number of Graduates by Credential Type")
    institution_card = create_scrollable_chart_card("graph-institution", "Number of Graduates by Institution")
    
    # Create visualization content
    visualization_content = html.Div([
        # Main content row with filters and visualizations
        dbc.Row([
            dbc.Col([
                # Filters button and section
                html.Div([filters_button, dbc.Row(filters_section)], className="sticky-top")
            ], width="auto"),
            dbc.Col([
                map_card,
                dbc.Row([
                    dbc.Col([isced_card], width=6),
                    dbc.Col([province_card], width=6)
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col([cma_card], width=6),
                    dbc.Col([credential_card], width=6)
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col([institution_card], width=10)
                ], justify="center", className="mb-4"),
            ])
        ], style={'background-color': '#F1F1F1'})
    ])
    
    # Create table content with pivot table
    from components.data_explorer import create_data_explorer
    table_content = create_data_explorer(data)
    
    # Create the app layout with tabs
    app_layout = html.Div([
        logo,
        dbc.Container([
            dbc.Tabs([
                dbc.Tab(
                    visualization_content, 
                    label="Interactive Map and Charts", 
                    tab_id="tab-visualization", 
                    active_label_style=UI_CONFIG["active_tab_style"], 
                    label_style=UI_CONFIG["tab_style"]
                ),
                dbc.Tab(
                    table_content, 
                    label="Data Explorer", 
                    tab_id="tab-data", 
                    active_label_style=UI_CONFIG["active_tab_style"], 
                    label_style=UI_CONFIG["tab_style"]
                ),
            ], id="tabs", active_tab="tab-visualization"),
        ], fluid=True)
    ], className="bg-dark", style={"font-family": UI_CONFIG["font_family"], "font-weight": UI_CONFIG["font_weight"]})
    
    return app_layout