"""Sidebar component for filters and selections."""

import dash_bootstrap_components as dbc
from dash import html, dcc
import dash_mantine_components as dmc
import brand_colours as bc

# Define common styles
LABEL_STYLE = {
    "font-family": 'Open Sans',
    "font-weight": "600",
    "margin-bottom": "5px",
    "margin-top": "10px"
}

BUTTON_FORMAT = {
    "margin-top": "15px",
    "border": "none",
    "padding": "10px 20px",
    "border-radius": "5px",
    "font-family": 'Open Sans',
    "font-weight": "600"
}

CHECKLIST_FORMAT = {
    "inputStyle": {"margin-right": "5px", "margin-left": "20px"},
    "style": {"margin-bottom": "15px"}
}

MULTI_DROPDOWN_FORMAT = {
    "value": [],
    "searchable": True,
    "placeholder": "All",
    "style": {
        "marginBottom": "15px",
    },
    "styles": {
        "input": {"fontFamily": "Open Sans", "fontWeight": 600},
        "item": {"fontFamily": "Open Sans", "fontWeight": 600},
    }
}

def filter_args(id, options, format):
    """Create args for filter components"""
    return {
        "id": id,
        "options": options,
        "value": [option['value'] for option in options],
        **format
    }

def mantine_filter_args(id, options, format):
    """Convert options from dash format to mantine format"""
    data = [{"value": option['value'], "label": option['label']} for option in options]
    return {
        "id": id,
        "data": data,
        "value": [option['value'] for option in options],
        **{k: v for k, v in format.items() if k != 'options'}  # Remove 'options' if present
    }

def button_args(id, background_color, color, format):
    """Create args for button components"""
    return {
        "id": id,
        "n_clicks": 0,
        "style": {
            "background-color": background_color,
            "color": color,
            **format
        }
    }

def create_filters_section(stem_bhase_options, year_options, prov_options, 
                          isced_options, credential_options, institution_options,
                          cma_options):
    """
    Creates a collapsible filters section with all filter components.
    
    Args:
        stem_bhase_options (list): Options for STEM/BHASE filter
        year_options (list): Options for Year filter
        prov_options (list): Options for Province filter
        isced_options (list): Options for ISCED filter
        credential_options (list): Options for Credential filter
        institution_options (list): Options for Institution filter
        cma_options (list): Options for CMA filter
        
    Returns:
        tuple: A tuple containing (filters_button, filters_section)
    """
    # Create filter args
    stem_bhase_args = filter_args("stem-bhase-filter", stem_bhase_options, CHECKLIST_FORMAT)
    year_args = filter_args("year-filter", year_options, CHECKLIST_FORMAT)
    
    # Use mantine_filter_args for Mantine components
    prov_args = mantine_filter_args("prov-filter", prov_options, MULTI_DROPDOWN_FORMAT)
    isced_args = mantine_filter_args("isced-filter", isced_options, MULTI_DROPDOWN_FORMAT)
    credential_args = mantine_filter_args("credential-filter", credential_options, MULTI_DROPDOWN_FORMAT)
    institution_args = mantine_filter_args("institution-filter", institution_options, MULTI_DROPDOWN_FORMAT)
    cma_args = mantine_filter_args("cma-filter", cma_options, MULTI_DROPDOWN_FORMAT)
    
    # Create button for showing/hiding filters
    filters_button = dbc.Button(
        "Show/hide filters",
        id="horizontal-collapse-button",
        className="mb-3 mt-3",
        color="dark",
        n_clicks=0,
    )
    
    # Create the filters content
    filters_section = dbc.Collapse(
        dbc.Card([
            dbc.CardHeader("Filters", 
                style={
                    "font-family": 'Open Sans',
                    "font-weight": "600",
                    "font-size": "16px"
                }),
            dbc.CardBody([
                html.Label("STEM/BHASE:", style=LABEL_STYLE),
                dbc.Checklist(**stem_bhase_args, inline=False, input_checked_style={
                    "backgroundColor": bc.MAIN_RED,
                    "borderColor": bc.MAIN_RED,
                }),
                html.Label("Academic Year:", style=LABEL_STYLE),
                dbc.Checklist(**year_args, inline=False, input_checked_style={
                    "backgroundColor": bc.MAIN_RED,
                    "borderColor": bc.MAIN_RED,
                }),
                html.Label("Province:", style=LABEL_STYLE),
                dmc.MultiSelect(**prov_args),
                html.Label("Census Metropolitan Area/Census Agglomeration:", style=LABEL_STYLE),
                dmc.MultiSelect(**cma_args),
                html.Label("ISCED Level:", style=LABEL_STYLE),
                dmc.MultiSelect(**isced_args),
                html.Label("Credential Type:", style=LABEL_STYLE),
                dmc.MultiSelect(**credential_args),
                html.Label("Institution:", style=LABEL_STYLE),
                dmc.MultiSelect(**institution_args),
                dbc.Button('Reset Filters', id='reset-filters', color="secondary", className="me-1"),
                dbc.Button('Clear Selection', id='clear-selection', color="danger", className="me-1"),
                # Hidden stores for selected values
                dcc.Store(id='selected-isced', data=None),
                dcc.Store(id='selected-province', data=None),
                dcc.Store(id='selected-feature', data=None),
                dcc.Store(id='selected-cma', data=None),
                dcc.Store(id='selected-credential', data=None),
                dcc.Store(id='selected-institution', data=None),
            ])
        ], className="mb-4 h-100"),
        id="horizontal-collapse",
        is_open=True,
        dimension="width",
    )
    
    return filters_button, filters_section