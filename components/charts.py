"""Chart components for visualization."""

import dash_bootstrap_components as dbc
from dash import dcc, html
import brand_colours as bc

def create_chart_card(chart_id, title, spinner=True):
    """
    Creates a card containing a chart with optional loading spinner.
    
    Args:
        chart_id (str): The ID for the chart component
        title (str): Card title to display
        spinner (bool): Whether to wrap the chart in a loading spinner
        
    Returns:
        dbc.Card: Card component with chart
    """
    # Define card header style
    card_header_style = {
        "font-family": 'Open Sans',
        "font-weight": "600",
        "font-size": "16px"
    }
    
    # Create the chart content
    chart_content = dcc.Graph(
        id=chart_id, 
        config={'displaylogo': False}
    )
    
    # Optionally wrap in spinner
    if spinner:
        chart_content = dbc.Spinner(
            chart_content,
            color="primary",
            type="border",
        )
    
    # Create and return the card
    return dbc.Card([
        dbc.CardHeader(title, style=card_header_style),
        dbc.CardBody([chart_content])
    ], className="mb-2 mt-2")

def create_scrollable_chart_card(chart_id, title):
    """
    Creates a card containing a chart with scrollable content for large datasets.
    
    Args:
        chart_id (str): The ID for the chart component
        title (str): Card title to display
        
    Returns:
        dbc.Card: Card component with scrollable chart
    """
    # Define card header style
    card_header_style = {
        "font-family": 'Open Sans',
        "font-weight": "600",
        "font-size": "16px"
    }
    
    # Create and return the scrollable card
    return dbc.Card([
        dbc.CardHeader(title, style=card_header_style),
        dbc.CardBody([
            dbc.Spinner(
                html.Div([
                    html.Div(
                        dcc.Graph(id=chart_id, config={'displaylogo': False}), 
                        className='scroll'
                    )
                ]),
                color="primary",
                type="border",
            ),
        ])
    ], className="mb-2 mt-2")