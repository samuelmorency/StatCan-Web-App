"""Data explorer component for interactive data analysis."""

import dash_bootstrap_components as dbc
from dash import html, dcc
from dash_pivottable import PivotTable
import brand_colours as bc

def initialize_pivot_table(data, id="pivot-table"):
    """
    Initialize a Pivot Table with the dataset.
    
    Args:
        data (pandas.DataFrame): Data to display in the pivot table
        id (str): The component ID to use
        
    Returns:
        dash_pivottable.PivotTable: Configured pivot table component
    """
    return PivotTable(
        id=id,
        data=data.reset_index().to_dict('records'),
        cols=['year'],
        rows=['Province_Territory', 'CMA_CA'],
        vals=['value'],
        aggregatorName='Integer Sum',
        rendererName='Table',
        colOrder='key_a_to_z',
        rowOrder='key_a_to_z',
        menuLimit=2000,
        unusedOrientationCutoff=10000,
    )

def create_data_explorer(data):
    """
    Create a data explorer section with pivot table and download button.
    
    Args:
        data (pandas.DataFrame): Data to display in the pivot table
        
    Returns:
        dbc.Row: Row component containing the data explorer
    """
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    dbc.Button(
                        "Download as Displayed", 
                        id='download-button', 
                        style={'background-color': 'black', 'borderColor': 'black'}
                    )
                ),
                dbc.CardBody([          
                    dcc.Download(id="download-data"),
                    dbc.Col([
                        initialize_pivot_table(data)  # Pass the dataset to pivot table
                    ])
                ], className="m-1")
            ], className="mb-4 mt-4 mx-8")
        ], style={'background-color': '#F1F1F1'})
    ])