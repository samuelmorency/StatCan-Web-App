import dash_mantine_components as dmc
from dash import Dash, _dash_renderer
_dash_renderer._set_react_version("18.2.0")

app = Dash(external_stylesheets=dmc.styles.ALL)

theme = {
    "primaryColor": "#C01823",
    "fontFamily": "'Open Sans', sans-serif",
    "colors": {
         # add your colors
        "deepBlue": ["#ffeaec", "#fdd4d7", "#f4a7ac", "#ed767e", "#e74e57", "#e4353e", "#e32632", "#ca1925", "#b5121f", "#9e0319"], # 10 colors
        # or replace default theme color
        "blue": ["#E9EDFC", "#C1CCF6", "#99ABF0" "..."],   # 10 colors
    },
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
         # content
     ],
 )

if __name__ == "__main__":
    app.run(debug=True)