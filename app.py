from dash import Dash, html
from flask import Flask
import dash_bootstrap_components as dbc

flask_server = Flask(__name__)

app = Dash(__name__, server=flask_server, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = [html.Div(children='Hello World')]

server = app.server
