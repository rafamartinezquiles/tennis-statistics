from typing import Optional, Tuple, Dict, Any

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium import plugins


def create_map() -> Dict[str, Any]:
    st.subheader("Region of interest")
    st.caption("Use the drawing tools to define the area for which imagery will be retrieved.")

    base_map = folium.Map(location=[0, 0], zoom_start=2, tiles=None)

    folium.TileLayer(
        tiles="http://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Google Satellite",
        max_zoom=20,
        subdomains=["mt0", "mt1", "mt2", "mt3"],
    ).add_to(base_map)

    plugins.Draw(export=True).add_to(base_map)
    folium.LayerControl().add_to(base_map)

    map_state = st_folium(base_map, width=750, height=500)
    return map_state


def extract_bbox(map_state: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """
    Returns (min_lon, min_lat, max_lon, max_lat) if a polygon was drawn.
    """
    if not map_state:
        return None

    last_drawing = map_state.get("last_active_drawing")
    if not last_drawing or "geometry" not in last_drawing:
        return None

    geometry = last_drawing["geometry"]
    # GeoJSON polygon: "coordinates" -> [ [ [lon, lat], ... ] ]
    try:
        coords = geometry["coordinates"][0]
        lats = [pt[1] for pt in coords]
        lons = [pt[0] for pt in coords]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        return (min_lon, min_lat, max_lon, max_lat)
    except Exception:
        return None
