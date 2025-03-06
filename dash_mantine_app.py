import dash_mantine_components as dmc
import dash
from dash import Dash, Input, Output, callback, html
from dash.exceptions import PreventUpdate
from brand_colours import DMC_COLOURS

from datetime import date

dash._dash_renderer._set_react_version("18.2.0")

app = Dash(__name__, external_stylesheets=dmc.styles.ALL)

theme = {
    "primaryColor": "MainRed",
    "fontFamily": "'Open Sans', sans-serif",
    "colors": DMC_COLOURS,
    "components": {
        "Button": {"defaultProps": {"fw": 400}},
        "Alert": {"styles": {"title": {"fontWeight": 500}}},
        "AvatarGroup": {"styles": {"truncated": {"fontWeight": 500}}},
        "Badge": {"styles": {"root": {"fontWeight": 500}}},
        "Progress": {"styles": {"label": {"fontWeight": 500}}},
        "RingProgress": {"styles": {"label": {"fontWeight": 500}}},
        "CodeHighlightTabs": {"styles": {"file": {"padding": 12}}},
        "Table": {
            "defaultProps": {
                "highlightOnHover": True,
                "withTableBorder": True,
                "verticalSpacing": "sm",
                "horizontalSpacing": "md",
            }
        },
    },
}

app.layout = dmc.MantineProvider(
     forceColorScheme="light",
     theme=theme,
     children=[
         dmc.DatePickerInput(
            id="date-picker",
            label="Start Date",
            description="You can also provide a description",
            minDate=date(2020, 8, 5),
            value=None,
            w=200
        ),
        dmc.Space(h=10),
        dmc.Text(id="selected-date"),
     ],
 )

@callback(Output("selected-date", "children"), Input("date-picker", "value"))
def update_output(d):
    prefix = "You have selected: "
    if d:
        return prefix + d
    else:
        raise PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=True)