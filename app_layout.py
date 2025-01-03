import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import brand_colours as bc
from dash_extensions.javascript import assign
import dash_leaflet as dl
from dash_ag_grid import AgGrid

checklist_format = {
    "inputStyle": {"margin-right": "5px", "margin-left": "20px"},
    "style": {"margin-bottom": "15px"}
}

multi_dropdown_format = {
    "value": [],
    "multi": True,
    "placeholder": "All",
    "searchable": True,
    "style": {"margin-bottom": "15px"}
}

button_format = {
    "margin-top": "15px",
    "border": "none",
    "padding": "10px 20px",
    "border-radius": "5px",
    #"margin-right": "10px"
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

def initialize_ag_grid(data):
    """Initialize AG Grid with the full dataset"""
    column_defs = [
        {
            "field": "STEM/BHASE",
            "headerName": "Category",
            "enableRowGroup": True,
            "filter": 'agSetColumnFilter',
        },
        {
            "field": "year",
            "headerName": "Year",
            "enableRowGroup": True,
            "filter": 'agSetColumnFilter',
        },
        {
            "field": "Province_Territory",
            "headerName": "Province/Territory",
            "enableRowGroup": True,
            "filter": 'agSetColumnFilter',
        },
        {
            "field": "CMA_CA",
            "headerName": "CMA/CA",
            "enableRowGroup": True,
            "filter": 'agSetColumnFilter',
        },
        {
            "headerName": "Education Level",
            "valueGetter": {
                "function": """
                const isced = params.data.ISCED_level_of_education;
                const cred = params.data.Credential_Type;
                return `${isced} (${cred})`;
                """
            },
            "enableRowGroup": True,
            "filter": 'agSetColumnFilter',
        },
        {
            "field": "Institution",
            "headerName": "Institution",
            "enableRowGroup": True,
            "filter": 'agSetColumnFilter',
        },
        {
            "field": "value",
            "headerName": "Graduates",
            "type": "numericColumn",
            "enableValue": True,
            "valueGetter": {
                "function": "Number(params.data.value).toLocaleString()"
            },
            "aggFunc": "sum",
            "filter": "agNumberColumnFilter",
        },
        {
            "headerName": "% of Total",
            "type": "numericColumn",
            "valueGetter": {
                "function": """
                const total = params.api.getModel().rootNode.childrenAfterFilter
                    .reduce((sum, node) => sum + Number(node.data.value), 0);
                return ((params.data.value / total) * 100).toFixed(1) + '%';
                """
            },
            "filter": "agNumberColumnFilter",
        }
    ]
    
    return AgGrid(
        id='table-cma',
        columnDefs=column_defs,
        rowData=data.reset_index().to_dict('records'),
        defaultColDef={
            "resizable": True,
            "sortable": True,
            "filter": True,
            "minWidth": 125,
            "filterParams": {
                "buttons": ["reset", "apply"],
                "debounceMs": 500
            },
            "enableValue": True,
            "enablePivot": True,
            "enableRowGroup": True
        },
        dashGridOptions={
            "groupDisplayType": "groupRows",
            "rowSelection": "multiple",
            "pagination": True,
            "paginationPageSize": 100,
            "cacheBlockSize": 100,
            "rowBuffer": 50,
            "maxBlocksInCache": 10,
            "rowModelType": "clientSide",
            "enableCellTextSelection": True,
            "ensureDomOrder": False,
            "suppressAnimationFrame": True
        },
        columnSize="sizeToFit",
        enableEnterpriseModules=True,
        className="ag-theme-alpine",
        style={"height": "600px", "width": "100%"}
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
    clear_selection_args = button_args("clear-selection", bc.LIGHT_GREY, bc.IIC_BLACK, button_format)
    download_button_args = button_args('download-button', bc.MAIN_RED, "white", button_format)
    
    visualization_content = html.Div([
        # Filters button
        dbc.Row(
            dbc.Col(
                dbc.Button(
                    "Show/hide filters",
                    id="horizontal-collapse-button",
                    className="mb-3",
                    color="primary",
                    n_clicks=0,
                ), width="auto"
            )
        ),
        # Main content row with filters and visualizations
        dbc.Row([
            dbc.Col([
                html.Div(
                    dbc.Collapse(
                        dbc.Card([
                            dbc.CardHeader("Filters"),
                            dbc.CardBody([
                                html.Label("STEM/BHASE:"),
                                dcc.Checklist(**stem_bhase_args),
                                html.Label("Academic Year:"),
                                dcc.Checklist(**year_args),
                                html.Label("Province:"),
                                dcc.Dropdown(**prov_args),
                                html.Label("Census Metropolitan Area/Census Agglomeration:"),
                                dcc.Dropdown(**cma_args),
                                html.Label("ISCED Level:"),
                                dcc.Dropdown(**isced_args),
                                html.Label("Credential Type:"),
                                dcc.Dropdown(**credential_args),
                                html.Label("Institution:"),
                                dcc.Dropdown(**institution_args),
                                html.Button('Reset Filters', **reset_filters_args),
                                html.Button('Clear Selection', **clear_selection_args),
                                dcc.Store(id='selected-isced', data=None),
                                dcc.Store(id='selected-province', data=None),
                                dcc.Store(id='selected-cma', data=None),
                            ])
                        ], className="mb-4 h-100"),
                        id="horizontal-collapse",
                        is_open=True,
                        dimension="width",
                    ), className="sticky-top"
                )
            ], width="auto"),
            dbc.Col([
                # Map card
                dbc.Card([
                    dbc.CardHeader("Graduates by CMA/CA"),
                    dbc.CardBody([
                        dl.Map(**map_args),
                    ])
                ], className="mb-4"),
                # Charts row
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
            ], width=9)
        ])
    ])

    table_content = dbc.Card([
        dbc.CardHeader(
            dbc.Row([
                dbc.Col(html.H3("Number of Graduates by CMA/CA"), width=9),
                dbc.Col(
                    html.Button("Download Table", **download_button_args),
                    width=3
                ),
            ]),
            className="d-flex align-items-center"
        ),
        dbc.CardBody([
            dcc.Download(id="download-data"),
            initialize_ag_grid(data)  # Pass the full dataset to AG Grid
        ])
    ], className="mb-4")

    # Create the app layout with tabs
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
        dbc.Tabs([
            dbc.Tab(visualization_content, label="Visualization", tab_id="tab-visualization"),
            dbc.Tab(table_content, label="Data Table", tab_id="tab-data"),
        ], id="tabs", active_tab="tab-visualization"),
    ], fluid=True)

    return app_layout


