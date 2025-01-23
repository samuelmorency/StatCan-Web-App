import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import brand_colours as bc
from dash_extensions.javascript import assign
import dash_leaflet as dl
from dash_pivottable import PivotTable

IIC_LOGO = "assets/logo.png"

TAB_STYLE = dict(backgroundColor="#e6e6e6", borderColor="#F1F1F1", color="black")

ACTIVE_TAB_STYLE = dict(backgroundColor="#F1F1F1", borderColor="#F1F1F1", color="black")


#'#F1F1F1'



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

logo = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=IIC_LOGO, height="45px")),
                        dbc.Col(dbc.NavbarBrand("Canadian STEM/BHASE Graduates Dashboard", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="https://invcanadazone.sharepoint.com/sites/infozone",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                dbc.Nav(
                    [nav_item, dropdown],
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
    className="mb-1",
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
        vals=['value'],
        aggregatorName='Integer Sum',
        rendererName='Table',
        colOrder='key_a_to_z',
        rowOrder='key_a_to_z',
        menuLimit=2000,
        unusedOrientationCutoff=10000
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
            dbc.CardHeader("Filters"),
            dbc.CardBody([
                html.Label("STEM/BHASE:"),
                dbc.Checklist(**stem_bhase_args, inline=False, input_checked_style={
                    "backgroundColor": bc.MAIN_RED,
                    "borderColor": bc.MAIN_RED,
                }),
                html.Label("Academic Year:"),
                dbc.Checklist(**year_args, inline=False, input_checked_style={
                    "backgroundColor": bc.MAIN_RED,
                    "borderColor": bc.MAIN_RED,
                }),
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
                #dbc.Button('Reset Filters', **reset_filters_args),
                dbc.Button('Reset Filters', id='reset-filters', color="secondary", className="me-1"),
                #dbc.Button('Clear Selection', **clear_selection_args, outline=True),
                dbc.Button('Clear Selection', id='clear-selection', color="danger", className="me-1"),
                dcc.Store(id='selected-isced', data=None),
                dcc.Store(id='selected-province', data=None),
                dcc.Store(id='selected-cma', data=None),
            ])
        ], className="mb-4 h-100"),
        id="horizontal-collapse",
        is_open=True,
        dimension="width",
    )
    
    map_card = dbc.Card([
        dbc.CardHeader("Graduates by CMA/CA"),
        dbc.CardBody([
            dbc.Spinner(
                dl.Map(**map_args),
                color="primary",
                type="border",
                ),
        ])
    ], className="mb-4 mt-4")
    
    isced_card = dbc.Card([
        dbc.CardHeader("ISCED Level Distribution"),
        dbc.CardBody([
            dbc.Spinner(
                dcc.Graph(id='graph-isced'),
                color="primary",
                type="border",
            ),
        ])
    ], className="mb-4 mt-4")
    
    province_card = dbc.Card([
        dbc.CardHeader("Provincial Distribution"),
        dbc.CardBody([
            dbc.Spinner(
                dcc.Graph(id='graph-province'),
                color="primary",
                type="border",
            ),
        ])
    ], className="mb-4 mt-4")
    
    visualization_content = html.Div([
        
        # Main content row with filters and visualizations
        dbc.Row([
            dbc.Col([
                # Filters button
                html.Div([filters_button, dbc.Row(filters_section)], className="sticky-top")], width="auto"),
            dbc.Col([
                map_card,
                dbc.Row([
                    dbc.Col([isced_card], width=6),
                    dbc.Col([province_card], width=6)
                ], className="mb-4"),
            ])
        ], style={'background-color': '#F1F1F1'})
    ])

    table_content = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H3("Pivot Table"), className="m-1"),
                dbc.CardBody([          
                    dbc.Button("Download as Displayed", id='download-button', color="secondary"),
                    dcc.Download(id="download-data"),
                    dbc.Col([
                        initialize_pivot_table(data)  # Pass the dataset to pivot table
                    ])
                ], className="m-1")], className="m-2")
            ], style={'background-color': '#F1F1F1'})
        ])
        

    # Create the app layout with tabs
    app_layout = html.Div([
        logo,
        dbc.Container([
            dbc.Tabs([
                dbc.Tab(visualization_content, label="Interactive Map and Charts", tab_id="tab-visualization", active_label_style=ACTIVE_TAB_STYLE, label_style=TAB_STYLE),
                dbc.Tab(table_content, label="Data Explorer", tab_id="tab-data", active_label_style=ACTIVE_TAB_STYLE, label_style=TAB_STYLE),
            ], id="tabs", active_tab="tab-visualization"),
            ], fluid=True)
        ], className="bg-dark")

    return app_layout


