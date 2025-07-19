import json
import numpy as np
import pandas as pd
import geopandas as gpd
import pydeck as pdk
from pyproj import CRS
from pydeck.data_utils import compute_view
from matplotlib import cm, colors as mcolors

# --- HELPERS (to_rgba, _create_arc_layer are unchanged) ---

def to_rgba(val, alpha=1.0):
    """Converts a color to an RGBA list [0-255], applying a specific alpha."""
    if isinstance(val, (tuple, list, np.ndarray)):
        arr = np.asarray(val, float)
        if arr.max() <= 1.0: arr *= 255
        if arr.size == 3: arr = np.append(arr, alpha * 255)
        else: arr[3] = alpha * 255
        return arr.astype(int).tolist()
    r, g, b, a_float = mcolors.to_rgba(val)
    final_alpha = a_float if alpha is None else alpha
    return [int(255 * r), int(255 * g), int(255 * b), int(255 * final_alpha)]

def _create_geojson_layer(gdf, **kwargs):
    """
    Internal helper to create a GeoJsonLayer.
    Passing the GeoDataFrame directly is the key fix.
    """
    # THE FIX: Pass the GeoDataFrame directly, not its GeoJSON representation.
    return pdk.Layer("GeoJsonLayer", data=gdf, **kwargs)

def _create_arc_layer(od_df, points_gdf, origin_col, dest_col, **kwargs):
    """Internal helper to create an ArcLayer, ensuring data integrity."""
    od_df = od_df.copy()
    points_gdf = points_gdf.copy()
    if not CRS(points_gdf.crs).equals(CRS(4326)):
        points_gdf = points_gdf.to_crs(4326)
    coords = points_gdf.geometry.apply(lambda p: [p.x, p.y])
    od_df["source"] = od_df[origin_col].map(coords)
    od_df["target"] = od_df[dest_col].map(coords)
    od_df.dropna(subset=["source", "target"], inplace=True)
    return pdk.Layer("ArcLayer", data=od_df, get_source_position="source", get_target_position="target", **kwargs)

# --- HIGH-LEVEL PLOTTING FUNCTION (Unchanged from last version) ---

def plot_od_map(
    od_df,
    region_gdf=None,
    points_gdf=None,
    background_gdf=None,
    origin_col="origin",
    dest_col="destination",
    weight_col="weight",
    id_col=None,
    w_min=0.05,
    edge_alpha=0.4,
    edge_color="red",
    edge_cmap=None,
    fill_cmap="Reds"
):
    """Generates an origin-destination map using PyDeck with optional layers."""
    od_df = od_df[od_df[weight_col] >= w_min].copy()
    if od_df.empty:
        raise ValueError("No flows with weight >= w_min found.")

    layers = []
    
    if points_gdf is None and region_gdf is not None:
        points_gdf = region_gdf.copy()
        points_gdf['geometry'] = points_gdf.geometry.centroid
    if points_gdf is None:
         raise ValueError("Cannot draw arcs without points_gdf or region_gdf to derive it.")

    # 1. Background Layer
    if background_gdf is not None:
        bg = background_gdf.copy()
        if not CRS(bg.crs).equals(CRS(4326)): bg = bg.to_crs(4326)
        if "color" not in bg.columns: bg["color"] = "#0e0e0e"
        bg["fill_color"] = bg["color"].apply(lambda c: to_rgba(c, alpha=1.0))
        layers.append(_create_geojson_layer(
            bg, stroked=True, filled=True, get_fill_color="fill_color",
            get_line_color=[255, 255, 255, 200], line_width_min_pixels=2,
            parameters={"depthTest": False}
        ))

    # 2. Region Layer
    if region_gdf is not None:
        regs = region_gdf.set_index(id_col) if id_col else region_gdf.copy()
        if not CRS(regs.crs).equals(CRS(4326)): regs = regs.to_crs(4326)
        inflow = od_df.groupby(dest_col)[weight_col].sum()
        max_in = inflow.max() or 1
        cmap = cm.get_cmap(fill_cmap) if isinstance(fill_cmap, str) else fill_cmap
        norm = mcolors.Normalize(0, max_in)
        touched = np.hstack([od_df[origin_col].unique(), od_df[dest_col].unique()])
        regs = regs.loc[regs.index.isin(touched)]
        regs["fill_color"] = inflow.reindex(regs.index, fill_value=0)\
                                 .apply(lambda v: to_rgba(cmap(norm(v)), alpha=0.6))
        layers.append(_create_geojson_layer(
            regs, pickable=True, auto_highlight=True, stroked=True, filled=True,
            get_fill_color="fill_color", get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=2, parameters={"depthTest": False}
        ))

    # 3. Arc Layer
    max_w = od_df[weight_col].max()
    od_df["width"] = 2 + np.sqrt(od_df[weight_col] / max_w) * 18
    if edge_cmap:
        cmap = cm.get_cmap(edge_cmap) if isinstance(edge_cmap, str) else edge_cmap
        norm = mcolors.Normalize(od_df[weight_col].min(), max_w)
        od_df["color"] = od_df[weight_col].apply(lambda v: to_rgba(cmap(norm(v)), alpha=edge_alpha))
    else:
        od_df["color"] = [to_rgba(edge_color, alpha=edge_alpha)] * len(od_df)
    layers.append(_create_arc_layer(
        od_df, points_gdf, origin_col, dest_col, get_width="width",
        get_source_color="color", get_target_color="color",
        pickable=True, auto_highlight=True
    ))

    bounds_provider = region_gdf if region_gdf is not None else points_gdf
    bounds = bounds_provider.to_crs(4326).total_bounds
    view = compute_view([[bounds[0], bounds[1]], [bounds[2], bounds[3]]])
    view.pitch = 30
    
    return pdk.Deck(layers=layers, initial_view_state=view)