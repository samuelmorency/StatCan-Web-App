from dash import html, dcc, dash_table
import dash_leaflet as dl
import dash_bootstrap_components as dbc
from dash_extensions.javascript import assign

# Constants for styling
FILTER_STYLES = {
    'dropdown': {"margin-bottom": "15px"},
    'button': {"margin-top": "15px"},
    'map': {'width': '100%', 'height': '600px'}
}

# Component factory functions
def create_dropdown(id, options, placeholder):
    return dcc.Dropdown(
        id=id,
        options=options,
        value=[],
        multi=True,
        placeholder=placeholder,
        searchable=True,
        style=FILTER_STYLES['dropdown']
    )

def create_filters(stem_bhase_options_full, year_options_full, prov_options_full, isced_options_full, credential_options_full, institution_options_full):
    stem_bhase_filter = [
        #html.H5("Filters"),
        html.Label("STEM/BHASE:"),
        dcc.Checklist(
            id='stem-bhase-filter',
            options=stem_bhase_options_full,
            value=[option['value'] for option in stem_bhase_options_full],
            #inputStyle={"margin-right": "5px", "margin-left": "20px"},
            #style={"margin-bottom": "15px"}
        )
    ]

    year_filter = [
        html.Label("Academic Year:"),
        dcc.Slider(
            id='year-filter',
            #options=year_options_full,
            value=[option['value'] for option in year_options_full],
            #inputStyle={"margin-right": "5px", "margin-left": "20px"},
            #style={"margin-bottom": "15px"}
        )
    ]

    prov_filter = [
        html.Label("Province:"),
        dcc.Dropdown(
            id='prov-filter',
            options=prov_options_full,
            value=[],
            multi=True,
            placeholder="All Provinces",
            searchable=True,
            #style={"margin-bottom": "15px"}
        ),
    ]
    
    isced_filter = [
        html.Label("ISCED Level:"),
        dcc.Dropdown(
            id='isced-filter',
            options=isced_options_full,
            value=[],
            multi=True,
            placeholder="All Levels",
            searchable=True,
            #style={"margin-bottom": "15px"}
        ),
    ]
    
    credential_filter = [
        html.Label("Credential Type:"),
        dcc.Dropdown(
            id='credential-filter',
            options=credential_options_full,
            value=[],
            multi=True,
            placeholder="All Credential Types",
            searchable=True,
            #style={"margin-bottom": "15px"}
        ),
    ]
    
    institution_filter = [
        html.Label("Institution:"),
        create_dropdown('institution-filter', institution_options_full, "All Institutions")
    ]

    filters = [
        dbc.Col(stem_bhase_filter),
        dbc.Col(year_filter),
        dbc.Col(prov_filter),
        dbc.Col(isced_filter),
        dbc.Col(credential_filter),
        dbc.Col(institution_filter),
        dbc.Col([
            html.Button('Reset Filters', id='reset-filters', n_clicks=0, style={"margin-top": "15px"}),
            html.Button('Clear Selection', id='clear-selection', n_clicks=0, style={"margin-top": "15px"}),
        ]),
        # Add dcc.Store components to store selected data for cross-filtering
        dcc.Store(id='selected-isced', data=None),
        dcc.Store(id='selected-province', data=None),
        dcc.Store(id='selected-cma', data=None)
    ]
    return filters

def create_map():
    mapvisual = html.Div(
        dl.Map(
            id='map',
            center=[56, -96],
            zoom=4,
            children=[
                dl.TileLayer(),
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
        )
    )
    return mapvisual

def create_charts():
    charts = dbc.Col([
        dbc.Row([
            html.Label("Chart Type:"),
            dcc.RadioItems(
                id='chart-type-isced',
                options=[
                    {'label': 'Bar', 'value': 'bar'},
                    {'label': 'Line', 'value': 'line'},
                    {'label': 'Pie', 'value': 'pie'}
                ],
                value='bar',
                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
            ),
            dcc.Graph(id='graph-isced'),  # Graph for ISCED level of education
        ], style={'height': '50%'}),
        dbc.Row([
            html.Label("Chart Type:"),
            dcc.RadioItems(
                id='chart-type-province',
                options=[
                    {'label': 'Bar', 'value': 'bar'},
                    {'label': 'Line', 'value': 'line'},
                    {'label': 'Pie', 'value': 'pie'}
                ],
                value='bar',
                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
            ),
            dcc.Graph(id='graph-province'),  # Graph for provinces
        ], style={'height': '50%'})
    ], width=4)
    return charts

def create_layout(stem_bhase_options_full, year_options_full, prov_options_full, isced_options_full, credential_options_full, institution_options_full):
    filters = create_filters(stem_bhase_options_full, year_options_full, prov_options_full, isced_options_full, credential_options_full, institution_options_full)
    layout = dbc.Container([
        #html.H1("Interactive Choropleth Map of STEM/BHASE Graduates in Canada", className="my-4"),
        html.Div([
            dbc.Row(filters, style={"background-color": "#f8f9fa", 'height': "10%"}),
            dbc.Row([
                # Arrange the two graphs side by side with chart type selection
                create_charts(),

                dbc.Col([
                    html.Div([create_map(),], style={"height": "100%"}),
                ])
            ], style={"height": "60%"}),
            dbc.Row([
                # Add the scrollable table at the bottom
                html.H3("Number of Graduates by CMA/CA"),
                dash_table.DataTable(
                    id='table-cma',
                    columns=[],  # Placeholder for table columns
                    data=[],  # Placeholder for table data
                    style_table={'height': '400px', 'overflowY': 'auto'},
                    style_cell={'textAlign': 'left'},
                    page_action='none',  # Disable pagination
                    sort_action='native',  # Enable sorting
                    filter_action='native',  # Enable filtering
                ),
            ], style={"height": "30%"})
        ])
    ], fluid=True)
    return layout