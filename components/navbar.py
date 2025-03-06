"""Navbar component for application header."""

import dash_bootstrap_components as dbc
from dash import html
from config.settings import APP_CONFIG

def create_navbar(logo_path="assets/logo.png"):
    """
    Creates a navigation bar with logo and application title.
    
    Args:
        logo_path (str): Path to the logo image
        
    Returns:
        dbc.Navbar: Configured navbar component
    """
    # Define text style
    text_style = {
        "font-family": 'Open Sans',
        "font-weight": "600"
    }
    
    # Create nav items for dropdown menu
    nav_item = dbc.NavItem(dbc.NavLink("Link", href="#"))
    
    # Create dropdown menu
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
    
    # Create and return the navbar
    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src=logo_path, height="45px")),
                            dbc.Col(dbc.NavbarBrand(APP_CONFIG["title"], className="ms-2", style=text_style)),
                        ],
                        align="center",
                        className="g-0",
                    ),
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
        className="mb-0",
    )