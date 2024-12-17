import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import brand_colours as bc
from dash_extensions.javascript import assign
import dash_leaflet as dl

checklist_format = {
    "inputStyle": {"margin-right": "5px", "margin-left": "20px"},
    "style": {"margin-bottom": "15px"}
}

def args(id, options, format):
    return {
        "id": id,
        "options": options,
        "value": [option['value'] for option in options],
        **format
    }

def create_layout(stem_bhase_options_full, year_options_full, prov_options_full, isced_options_full, credential_options_full, institution_options_full):
    stem_bhase_args = args("stem-bhase-filter", stem_bhase_options_full, checklist_format)
    year_args = args("year-filter", year_options_full, checklist_format)
    # Create the app layout
    app_layout = dbc.Container([
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H1("Canadian STEM/BHASE Graduates Dashboard", className="text-primary mb-4"),
                        html.H4("Interactive visualization of graduate statistics across Canada", className="text-muted")
                    ])
                ], className="mb-4 border-0"),
            width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Filters"),
                    dbc.CardBody([
                        html.Label("STEM/BHASE:"),
                        dcc.Checklist(**stem_bhase_args),
                        html.Label("Academic Year:"),
                        dcc.Checklist(**year_args),
                        html.Label("Province:"),
                        dcc.Dropdown(
                            id='prov-filter',
                            options=prov_options_full,
                            value=[],
                            multi=True,
                            placeholder="All Provinces",
                            searchable=True,
                            style={"margin-bottom": "15px"}
                        ),
                        html.Label("ISCED Level:"),
                        dcc.Dropdown(
                            id='isced-filter',
                            options=isced_options_full,
                            value=[],
                            multi=True,
                            placeholder="All Levels",
                            searchable=True,
                            style={"margin-bottom": "15px"}
                        ),
                        html.Label("Credential Type:"),
                        dcc.Dropdown(
                            id='credential-filter',
                            options=credential_options_full,
                            value=[],
                            multi=True,
                            placeholder="All Credential Types",
                            searchable=True,
                            style={"margin-bottom": "15px"}
                        ),
                        html.Label("Institution:"),
                        dcc.Dropdown(
                            id='institution-filter',
                            options=institution_options_full,
                            value=[],
                            multi=True,
                            placeholder="All Institutions",
                            searchable=True,
                            style={"margin-bottom": "15px"}
                        ),
                        html.Button('Reset Filters', 
                            id='reset-filters', 
                            n_clicks=0, 
                            style={
                                "margin-top": "15px",
                                "background-color": bc.MAIN_RED,
                                "color": "white",
                                "border": "none",
                                "padding": "10px 20px",
                                "border-radius": "5px",
                                "margin-right": "10px"
                            }
                        ),
                        html.Button('Clear Selection', 
                            id='clear-selection', 
                            n_clicks=0, 
                            style={
                                "margin-top": "15px",
                                "background-color": bc.LIGHT_GREY,
                                "color": bc.IIC_BLACK,
                                "border": "none",
                                "padding": "10px 20px",
                                "border-radius": "5px"
                            }
                        ),
                        # Add dcc.Store components to store selected data for cross-filtering
                        dcc.Store(id='selected-isced', data=None),
                        dcc.Store(id='selected-province', data=None),
                        dcc.Store(id='selected-cma', data=None),
                    ])
                    
                ], className="mb-4 h-100"),
            ], width=3),

            dbc.Col([
                # Remove Spinner from map, keep direct map component
                dbc.Card([
                    dbc.CardHeader("Graduates by CMA/CA"),
                    dbc.CardBody([
                        dl.Map(
                            id='map',
                            center=[56, -96],
                            zoom=4,
                            children=[
                                dl.TileLayer(
                                    url='https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png',
                                    attribution='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>, &copy; <a href="https://carto.com/attribution">CARTO</a>'
                                ),
                                dl.GeoJSON(
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
                                ),
                            ],
                            style={'width': '100%', 'height': '600px'},
                        ),
                    ])
                ], className="mb-4"),

                # Keep existing spinners for charts
                # Arrange the two graphs side by side with chart type selection
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("ISCED Level Distribution"),
                            dbc.CardBody([
                                dbc.Spinner(
                                    dcc.Graph(id='graph-isced'),
                                    color="primary",
                                    type="border",
                                ),
                            ])
                        ])
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Provincial Distribution"),
                            dbc.CardBody([
                                dbc.Spinner(
                                    dcc.Graph(id='graph-province'),
                                    color="primary",
                                    type="border",
                                ),
                            ])
                        ])
                    ], width=6)
                ], className="mb-4"),

                # Add the scrollable table at the bottom
                html.H3("Number of Graduates by CMA/CA"),
                dash_table.DataTable(
                    id='table-cma',
                    columns=[],  # Placeholder for table columns
                    data=[],  # Placeholder for table data
                    style_table={'height': '400px', 'overflowY': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'color': bc.IIC_BLACK,
                        'backgroundColor': 'white'
                    },
                    style_header={
                        'backgroundColor': bc.LIGHT_BLUE,
                        'color': bc.IIC_BLACK,
                        'fontWeight': 'bold'
                    },
                    page_action='none',  # Disable pagination
                    sort_action='native',  # Enable sorting
                    filter_action='native',  # Enable filtering
                ),
            ], width=9)
        ])
    ], fluid=True)

    return app_layout


