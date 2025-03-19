import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import brand_colours as bc
from dash_extensions.javascript import assign
import dash_leaflet as dl
from dash_pivottable import PivotTable
from faq import faq

IIC_LOGO = "assets/logo.png"

CANADA_BOUNDS = [
    [36.676556, -141.001735],  # Southwest corner
    [68.110626, -52.620422]    # Northeast corner
]


TAB_STYLE = {
    "backgroundColor": "#cccccc", 
    "borderColor": "#F1F1F1", 
    "color": "black",
    "font-family": 'Open Sans',
    "font-weight": "600"
}

ACTIVE_TAB_STYLE = {
    "backgroundColor": "#F1F1F1", 
    "borderColor": "#F1F1F1", 
    "color": "black",
    "font-family": 'Open Sans',
    "font-weight": "600"
}

APP_STYLE = {
    "font-family": 'Open Sans',
    "font-weight": "600"
}

TEXT_STYLE = {
    "font-family": 'Open Sans',
    "font-weight": "600"
}

CARD_HEADER_STYLE = {
    "font-family": 'Open Sans',
    "font-weight": "600",
    "font-size": "16px"
}

LABEL_STYLE = {
    "font-family": 'Open Sans',
    "font-weight": "600",
    "margin-bottom": "3px",
    "margin-top": "8px",
    "font-size": "16px"
}

checklist_format = {
    "inputStyle": {"margin-right": "4px", "margin-left": "16px"},  # Reduced margins
    "style": {"margin-bottom": "12px", "font-size": "14px"}        # Added font size and reduced margin
}


multi_dropdown_format = {
    "value": [],
    "multi": True,
    "placeholder": "All",
    "searchable": True,
    "style": {
        "margin-bottom": "12px",
        "font-family": 'Open Sans',
        "font-weight": "600",
        "font-size": "12px"
    }
}

button_format = {
    "margin-top": "15px",
    "border": "none",
    "padding": "10px 20px",
    "border-radius": "5px",
    "font-family": 'Open Sans',
    "font-weight": "600"
}

tile_layer = dl.TileLayer(
    url='https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png',
    attribution='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>, &copy; <a href="https://carto.com/attribution">CARTO</a>'
)

geo_json = dl.GeoJSON(
    id='cma-geojson',
    data=None,
    style=assign("""
    function(feature) {
        return feature.properties.style;
    }
    """),
    zoomToBounds=True,
    hoverStyle=dict(
        weight=2, color='black', dashArray='',
        fillOpacity=0.7
    ),
    onEachFeature=assign("""
    function(feature, layer) {
        if (feature.properties && feature.properties.tooltip) {
            layer.bindTooltip(feature.properties.tooltip);
        }
    }
    """),
    options=dict(interactive=True),
    eventHandlers=dict(
        click=assign("""
        function(e, ctx) {
            e.originalEvent._stopped = true;
            const clickData = {
                feature: e.sourceTarget.feature.properties.DGUID,
                points: [{
                    featureId: e.sourceTarget.feature.properties.DGUID
                }]
            };
            ctx.setProps({ 
                clickData: clickData,
                clickedFeature: null  // Reset the clicked feature
            });
        }
        """)
    ),
)

map_args = dict(
    id='map',
    center=[56, -96],
    zoom=4,
    children=[tile_layer, geo_json],
    style={'width': '100%', 'height': '600px'},
    maxBounds=CANADA_BOUNDS,          # Add bounds restriction
    maxBoundsViscosity=1.0,           # Makes the bounds "sticky"
    minZoom=4,                        # Set minimum zoom level
    #maxZoom=10,                       # Set maximum zoom level
)

# make a reuseable navitem for the different examples
nav_item = dbc.NavItem(dbc.NavLink("Link", href="#"))

# make a reuseable dropdown for the different examples
dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Entry 1"),
        dbc.DropdownMenuItem("Entry 2"),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem("Entry 3"),
    ],
    nav=True,
    in_navbar=True,
    label="Menu",
)

def create_user_guide_modal():
    """Create modal dialog for user guide"""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("User Guide"), close_button=True),
            dbc.ModalBody(
                html.Div(
                    [
                        html.Div(
                            html.Div(id="user-guide-content"),
                            style={
                                "maxHeight": "70vh",
                                "overflowY": "auto",
                                "padding": "10px"
                            }
                        )
                    ]
                )
            ),
            dbc.ModalFooter(
                dbc.Button(
                    "Close",
                    id="close-guide-button",
                    className="ms-auto",
                    n_clicks=0,
                    style = {
                        "background-color": bc.MAIN_RED,
                        "color": "white",
                        **button_format
                    }
                )
            ),
        ],
        id="user-guide-modal",
        size="lg",
        is_open=False,
    )

faq_div_contents = []

#loop through each key value pair in the faq dictionary
for key, value in faq.items():
    faq_div_contents.extend([
        html.H3(key),
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    dcc.Markdown(value[subkey]),
                    title=html.H5(subkey)
                ) for subkey in value
            ],
            start_collapsed=True, className="mb-4"
        )
    ])



#faq_accordion = dbc.Accordion()

def create_faq_modal():
    """Create modal dialog for FAQ"""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Frequently Asked Questions"), close_button=True),
            dbc.ModalBody(
                html.Div(
                    [
                        html.Div(
                            html.Div(faq_div_contents, id="faq-content"),
                            style={
                                "maxHeight": "70vh",
                                "overflowY": "auto",
                                "padding": "10px"
                            }
                        )
                    ]
                )
            ),
            dbc.ModalFooter(
                dbc.Button(
                    "Close",
                    id="close-faq-button",
                    className="ms-auto",
                    n_clicks=0,
                    style = {
                        "background-color": bc.MAIN_RED,
                        "color": "white",
                        **button_format
                    }
                )
            ),
        ],
        id="faq-modal",
        size="lg",
        is_open=False,
    )

logo = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=IIC_LOGO, height="45px")),
                        dbc.Col(dbc.NavbarBrand("Canadian STEM/BHASE Graduates Dashboard", className="ms-2", style=TEXT_STYLE)),
                    ],
                    align="center",
                    className="g-0",
                ),
                #href="https://invcanadazone.sharepoint.com/sites/infozone",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                dbc.Nav(
                    [
                        html.Div(
                            "The Canadian Centre for Education Statistics (CCES) at Statistics Canada by CSBP. July 2024.",
                            className="me-3 text-light d-flex align-items-center",
                            style={"font-size": "12px", "font-style": "italic"}
                        ),
                        dbc.Button(
                            "User Guide",
                            id="open-guide-button",
                            className="me-2",
                            n_clicks=0,
                            style = {
                                "background-color": bc.MAIN_RED,
                                "color": "white",
                                **button_format
                            }
                        ),
                        dbc.Button(
                            "FAQ",
                            id="open-faq-button",
                            className="me-2",
                            n_clicks=0,
                            style = {
                                "background-color": bc.MAIN_RED,
                                "color": "white",
                                **button_format
                            }
                        )
                    ],
                    className="ms-auto",
                    navbar=True,
                ),
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ], fluid=True,
    ),
    color="dark",
    dark=True,
    className="mb-0",
)



def filter_args(id, options, format):
    return {
        "id": id,
        "options": options,
        "value": [option['value'] for option in options],
        **format
    }

def button_args(id, background_color, color, format):
    return {
        "id": id,
        "n_clicks": 0,
        "style": {
            "background-color": background_color,
            "color": color,
            **format
        }
    }

def initialize_pivot_table(data):
    """Initialize Pivot Table with the dataset"""
    return PivotTable(
        id='pivot-table',
        data=data.reset_index().to_dict('records'),
        cols=['year'],
        rows=['Province_Territory', 'CMA_CA'],
        vals=['Value'],
        aggregatorName='Integer Sum',
        rendererName='Table',
        colOrder='key_a_to_z',
        rowOrder='key_a_to_z',
        menuLimit=2000,
        unusedOrientationCutoff=10000,
        #className='pvtUi'
    )

def create_layout(data, stem_bhase_options_full, year_options_full, prov_options_full, isced_options_full, credential_options_full, institution_options_full, cma_options_full):
    # ...existing code...
    stem_bhase_args = filter_args("stem-bhase-filter", stem_bhase_options_full, checklist_format)
    year_args = filter_args("year-filter", year_options_full, checklist_format)
    prov_args = filter_args("prov-filter", prov_options_full, multi_dropdown_format)
    isced_args = filter_args("isced-filter", isced_options_full, multi_dropdown_format)
    credential_args = filter_args("credential-filter", credential_options_full, multi_dropdown_format)
    institution_args = filter_args("institution-filter", institution_options_full, multi_dropdown_format)
    cma_args = filter_args("cma-filter", cma_options_full, multi_dropdown_format)
    reset_filters_args = button_args("reset-filters", bc.MAIN_RED, "white", button_format)
    clear_selection_args = button_args("clear-selection", "danger", bc.IIC_BLACK, button_format)
    download_button_args = button_args('download-button', bc.MAIN_RED, "white", button_format)
    
    filters_button = dbc.Button(
        "Show/hide filters",
        id="horizontal-collapse-button",
        className="mb-3 mt-3",
        color="dark",
        n_clicks=0,
    )
    
    filters_section = dbc.Collapse(
        dbc.Card([
            dbc.CardHeader("Filters", style=CARD_HEADER_STYLE),
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
                dcc.Dropdown(**prov_args),
                html.Label("Census Metropolitan Area/Census Agglomeration:", style=LABEL_STYLE),
                dcc.Dropdown(**cma_args),
                html.Label("ISCED Level:", style=LABEL_STYLE),
                dcc.Dropdown(**isced_args),
                html.Label("Credential Type:", style=LABEL_STYLE),
                dcc.Dropdown(**credential_args),
                html.Label("Institution:", style=LABEL_STYLE),
                dcc.Dropdown(**institution_args),
                #dbc.Button('Reset Filters', **reset_filters_args),
                dbc.Button('Reset Filters', id='reset-filters', color="secondary", className="me-1"),
                #dbc.Button('Clear Selection', **clear_selection_args, outline=True),
                dbc.Button('Clear Selection', id='clear-selection', color="danger", className="me-1"),
                dcc.Store(id={'type': 'store', 'item': 'isced'}, data=None),
                dcc.Store(id={'type': 'store', 'item': 'province'}, data=None),
                dcc.Store(id={'type': 'store', 'item': 'cma'}, data=None),
                dcc.Store(id={'type': 'store', 'item': 'credential'}, data=None),
                dcc.Store(id={'type': 'store', 'item': 'institution'}, data=None),
            ])
        ], className="mb-4 h-100"),
        id="horizontal-collapse",
        is_open=True,
        dimension="width",
    )
    
    map_card = dbc.Card([
        dbc.CardHeader("Number of Graduates by CMA/CA", style=CARD_HEADER_STYLE),
        dbc.CardBody([
            dbc.Spinner(
                dl.Map(**map_args),
                color="primary",
                type="border",
                ),
            dcc.Store(id='selected-feature', data=None),
        ])
    ], className="mb-2 mt-4")
    
    isced_card = dbc.Card([
        dbc.CardHeader("Number of Graduates by ISCED Level of Education", style=CARD_HEADER_STYLE),
        dbc.CardBody([
            dbc.Spinner(
                dcc.Graph(id={'type': 'graph', 'item': 'isced'}, config={'displaylogo': False}),
                color="primary",
                type="border",
            ),
        ])
    ], className="mb-2 mt-2")
    
    province_card = dbc.Card([
        dbc.CardHeader("Number of Graduates by Province", style=CARD_HEADER_STYLE),
        dbc.CardBody([
            dbc.Spinner(
                dcc.Graph(id={'type': 'graph', 'item': 'province'}, config={'displaylogo': False}),
                color="primary",
                type="border",
            ),
        ])
    ], className="mb-2 mt-2")
    
    cma_card = dbc.Card([
        dbc.CardHeader("Number of Graduates by CMA/CA", style=CARD_HEADER_STYLE),
        dbc.CardBody([
            dbc.Spinner(
                html.Div([
                    html.Div(
                        dcc.Graph(id={'type': 'graph', 'item': 'cma'}, config={'displaylogo': False}), className='scroll'
                    )
                ]),
                color="primary",
                type="border",
            ),
        ])
    ], className="mb-2 mt-2")

    credential_card = dbc.Card([
        dbc.CardHeader("Number of Graduates by Credential Type", style=CARD_HEADER_STYLE),
        dbc.CardBody([
            dbc.Spinner(
                dcc.Graph(id={'type': 'graph', 'item': 'credential'}, config={'displaylogo': False}),
                color="primary",
                type="border",
            ),
        ])
    ], className="mb-2 mt-2")

    institution_card = dbc.Card([
        dbc.CardHeader("Number of Graduates by Institution", style=CARD_HEADER_STYLE),
        dbc.CardBody([
            dbc.Spinner(
                html.Div([
                    html.Div(
                        dcc.Graph(id={'type': 'graph', 'item': 'institution'}, config={'displaylogo': False}), className='scroll'
                    )
                ]),
                color="primary",
                type="border",
            ),
        ])
    ], className="mb-2 mt-2")

    visualization_content = html.Div([
        
        # Main content row with filters and visualizations
        dbc.Row([
            dbc.Col([
                # Filters button
                html.Div([filters_button, dbc.Row(filters_section)])], width="auto"),
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

    table_content = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(dbc.Button("Download as Displayed", id='download-button', style={'background-color': bc.MAIN_RED, 'borderColor': bc.MAIN_RED})),
                dbc.CardBody([          
                    
                    dcc.Download(id="download-data"),
                    dbc.Col([
                        initialize_pivot_table(data)  # Pass the dataset to pivot table
                    ])
                ], className="m-1")], className="mb-4 mt-4 mx-8")
            ], style={'background-color': '#F1F1F1'})
        ])
        

    # Create the app layout with tabs
    app_layout = html.Div([
        logo,
        create_user_guide_modal(),
        create_faq_modal(),  # Add FAQ modal
        dbc.Container([
            dbc.Tabs([
                dbc.Tab(visualization_content, label="Interactive Map and Charts", tab_id="tab-visualization", active_label_style=ACTIVE_TAB_STYLE, label_style=TAB_STYLE),
                dbc.Tab(table_content, label="Data Explorer", tab_id="tab-data", active_label_style=ACTIVE_TAB_STYLE, label_style=TAB_STYLE),
            ], id="tabs", active_tab="tab-visualization"),
            ], fluid=True)
        ], className="bg-dark", style=APP_STYLE)

    return app_layout


