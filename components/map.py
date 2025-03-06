"""Map component for visualization."""

import dash_leaflet as dl
from dash_extensions.javascript import assign
import brand_colours as bc
from config.settings import MAP_CONFIG

def create_map_component(id="map"):
    """
    Creates a map component for geographic visualization.
    
    Args:
        id (str): The component ID to use
        
    Returns:
        dash_leaflet.Map: Configured map component
    """
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
        id=id,
        center=MAP_CONFIG["default_center"],
        zoom=MAP_CONFIG["default_zoom"],
        children=[tile_layer, geo_json],
        style={'width': '100%', 'height': '600px'},
        maxBounds=MAP_CONFIG["bounds"],
        maxBoundsViscosity=1.0,
        minZoom=4,
    )

    return dl.Map(**map_args)