from dash import Dash, html
from flask import Flask
import dash_bootstrap_components as dbc

flask_server = Flask(__name__)

app = Dash(__name__, server=flask_server, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = [html.Div(children='Hello World')]

# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()