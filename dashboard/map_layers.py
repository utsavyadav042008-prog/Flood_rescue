"""Assemble the Folium map from prepared layers (no file reading here).

All heavy work (reading GeoJSON, resampling rasters) happened earlier in
``dashboard.data.prepare_layers``; this module only turns the result into
Folium objects, so it is fast and re-runs on every checkbox change.

Design choices
--------------
* Lines and polygons use ``folium.GeoJson`` (one object per layer, fast).
* Points (settlements, hospitals, cut-off) are added one by one as markers
  with hand-built popups: there are only tens to hundreds of them, and it
  avoids relying on how a library styles GeoJSON points.
* Colour is never the only cue: roads are lines, buildings are filled
  shapes, settlements are small grey dots, cut-off settlements are large red
  dots with a white ring, destinations are plus-sign icons.
"""

from __future__ import annotations

import folium

from dashboard import data
from dashboard.data import LINE, POINT_CIRCLE, POINT_ICON, POLYGON, RASTER_MASK, RASTER_PROB

#: (radius px, outline colour, outline width) per point layer.
_CIRCLE_STYLE = {"settlements": (4, "#ffffff", 1), "cutoff": (10, "#ffffff", 3)}
_LINE_WEIGHT = {"osm_roads": 1, "affected_roads": 4, "affected_bridges": 6}


def _geojson_layer(spec: data.LayerSpec, features: list) -> folium.GeoJson:
    collection = {"type": "FeatureCollection", "features": features}
    color = spec.color
    if spec.kind == LINE:
        weight = _LINE_WEIGHT.get(spec.key, 3)
        style = lambda _feature, c=color, w=weight: {"color": c, "weight": w, "opacity": 0.9}
    else:                                              # polygon
        style = lambda _feature, c=color: {"color": c, "weight": 1, "fillColor": c, "fillOpacity": 0.6}
    popup = None
    if spec.popup:
        popup = folium.GeoJsonPopup(fields=[k for k, _ in spec.popup],
                                    aliases=[label for _, label in spec.popup],
                                    labels=True, localize=False)
    return folium.GeoJson(collection, name=spec.label, style_function=style, popup=popup)


def _add_points(group, spec: data.LayerSpec, features: list) -> None:
    for feature in features:
        point = data.representative_point(feature.get("geometry") or {})
        if point is None:
            continue
        lon, lat = point
        props = feature.get("properties") or {}
        popup = folium.Popup(data.popup_html(spec, props), max_width=340)
        tooltip = props.get("name") or None
        if spec.kind == POINT_ICON:
            folium.Marker([lat, lon], popup=popup, tooltip=tooltip,
                          icon=folium.Icon(color="green", icon="plus")).add_to(group)
        else:
            radius, outline, weight = _CIRCLE_STYLE.get(spec.key, (5, "#ffffff", 1))
            folium.CircleMarker([lat, lon], radius=radius, color=outline, weight=weight, fill=True,
                                fill_color=spec.color, fill_opacity=0.95, popup=popup,
                                tooltip=tooltip).add_to(group)


def build_map(prepared: dict, bounds: list, enabled: list) -> folium.Map:
    """Build the map.  ``bounds`` is ``[[south, west], [north, east]]``."""
    (south, west), (north, east) = bounds
    fmap = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=11,
                      tiles="OpenStreetMap", control_scale=True, prefer_canvas=True)
    # Keyless terrain basemap (CartoDB tiles now need an API key).  show=False keeps
    # OpenStreetMap as the visible default; the layer control switches between them.
    folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", name="Terrain (OpenTopoMap)",
                     attr="Map data © OpenStreetMap contributors, SRTM | style © OpenTopoMap (CC-BY-SA)",
                     max_zoom=17, show=False).add_to(fmap)
    # Background picture only (never an analysis input): helps people recognise the valley.
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     name="Satellite (Esri)", attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
                     max_zoom=19, show=False).add_to(fmap)
    try:                                               # optional plugin (absent in the test fakes)
        from folium.plugins import Fullscreen
        Fullscreen(position="topleft").add_to(fmap)
    except Exception:
        pass

    for spec in data.LAYERS:                           # registry order = draw order, bottom to top
        layer = prepared.get(spec.key)
        if layer is None:
            continue
        group = folium.FeatureGroup(name=spec.label, show=True)
        if spec.kind in (RASTER_MASK, RASTER_PROB):
            if layer.png is None or layer.bounds is None:
                continue
            w, s, e, n = layer.bounds
            # A file PATH (not a data URI): folium reads and embeds the file itself.
            folium.raster_layers.ImageOverlay(image=str(layer.png), bounds=[[s, w], [n, e]],
                                              opacity=1.0, interactive=False, name=spec.label).add_to(group)
        elif not layer.features:
            continue                                   # an empty GeoJsonPopup layer would raise
        elif spec.kind in (LINE, POLYGON):
            _geojson_layer(spec, layer.features).add_to(group)
        elif spec.kind in (POINT_CIRCLE, POINT_ICON):
            _add_points(group, spec, layer.features)
        group.add_to(fmap)

    fmap.fit_bounds(bounds)
    fmap.get_root().html.add_child(folium.Element(data.legend_html(data.legend_items(enabled))))
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap
