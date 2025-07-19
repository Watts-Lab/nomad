import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.affinity import scale
import pandas as pd
import numpy as np
import nomad.io.base as loader
import h3
import pdb
pd.plotting.register_matplotlib_converters()

def h3_cell_to_polygon(cell):
    """Return a shapely Polygon for the given H3 cell (lon/lat)."""
    # h3.cell_to_boundary returns [(lat, lng), ...]
    coords = h3.cell_to_boundary(cell)
    lats, lons = zip(*coords)
    return Polygon(zip(lons, lats))

def adjust_zoom(x, y, ax, buffer=0.5):
    if buffer is not None:
        x0, x1 = x.quantile(0.01), x.quantile(0.99)
        y0, y1 = y.quantile(0.01), y.quantile(0.99)
        pad_x = (x1 - x0) * buffer / 2
        pad_y = (y1 - y0) * buffer / 2
        ax.set_xlim(x0 - pad_x, x1 + pad_x)
        ax.set_ylim(y0 - pad_y, y1 + pad_y)

def plot_pings(pings_df, ax, current_idx=None, point_color='black', cmap=None,
               radius=None, s=6, marker='o', alpha=1, circle_color=None, circle_alpha=None, traj_cols=None, **kwargs):
    """
    Plot pings as true‐radius circles (projected CRS only) and point centers.
    If `radius` is None or using lon/lat, falls back to a simple scatter.
    If `radius` is provided (float or column), draws buffered circles.
    If `cluster` exists and CRS is projected, colors circles by cluster
    (noise/cluster==-1 transparent); otherwise uses `point_color`.

    Parameters
    ----------
    pings_df : pandas.DataFrame
        Trajectory points, with spatial columns (x/y or lon/lat) and optional 'cluster'.
    ax : matplotlib.axes.Axes
        Axis to draw on.
    current_idx : int, optional
        Index of ping to highlight (not used here).
    radius : float or str, default None
        If float, all circles use this constant radius (CRS units).
        If str, treated as a column name or canonical key: first looks in
        pings_df[radius], then in traj_cols[radius].
        If None, no circles are drawn and only a scatter is shown.
    point_color : color, default 'black'
        Marker color for points (and circles if no cluster).
    cmap : str or Colormap, default 'twilight'
        Colormap for mapping cluster IDs to circle colors.
    traj_cols : dict, optional
        Mapping of canonical keys to actual column names.
    s : float or array-like, default 6
        Marker size for scatter points.
    marker : str or MarkerStyle, default 'o'
        Marker style for scatter points.
    alpha : float, default 1
        Opacity for point markers (and for circles if circle_alpha is None).
    circle_alpha : float, optional
        Opacity for circle fills; if None, uses the same value as `alpha`.
    """
    # determine columns for plotting
    coord_key1, coord_key2, use_lon_lat = loader._fallback_spatial_cols(
        pings_df.columns, traj_cols, kwargs
    )
    traj_cols = loader._parse_traj_cols(
        pings_df.columns, traj_cols, kwargs, warn=False
    )
    xcol, ycol = traj_cols[coord_key1], traj_cols[coord_key2]

    # default circle alpha to point alpha
    if circle_alpha is None:
        circle_alpha = alpha

    # if no radius or geographic coords: simple scatter
    if radius is None or use_lon_lat:
        ax.scatter( pings_df[xcol], pings_df[ycol], s=s, color=point_color, marker=marker, alpha=alpha)
        return

    # otherwise draw circles + scatter
    # resolve radius: scalar or per‐point
    if isinstance(radius, str):
        if radius in pings_df.columns:
            r = pings_df[radius]
        else:
            r = pings_df[traj_cols.get(radius, radius)]
    else:
        r = radius

    # build GeoDataFrame in projected CRS
    gdf_pings = gpd.GeoDataFrame(
        pings_df,
        geometry=[Point(x, y) for x, y in zip(pings_df[xcol], pings_df[ycol])],
        crs="EPSG:3857"
    )

    # choose circle colors
    gdf = gdf_pings.copy()
    if circle_color:
        colors= circle_color
    elif 'cluster' in pings_df and cmap:
        gdf = gdf_pings.loc[gdf['cluster']!=-1]
        n = gdf['cluster'].nunique() or 1
        colors = gdf['cluster'].map(
            lambda c: plt.get_cmap(cmap)((c + 1) / (n + 1))
        )
    else:
        colors = "gray"

    # buffer geometries
    if np.isscalar(r):
        geoms = [pt.buffer(r) for pt in gdf.geometry]
    else:
        geoms = [pt.buffer(rv) for pt, rv in zip(gdf.geometry, r)]

    # plot circles
    gdf.geometry = geoms
    gdf.plot(ax=ax, color=colors, alpha=circle_alpha, linewidth=0.5, edgecolor=point_color)

    # plot point centers
    gdf_pings.plot(ax=ax, color=point_color, markersize=s, marker=marker, linewidth=0, alpha=alpha)

def plot_stops(stops, ax, cmap='Reds', traj_cols=None, crs=None, stagger=True, edge_only=False, **kwargs):
    try:
        coord_x, coord_y, use_lonlat = loader._fallback_spatial_cols(stops.columns, traj_cols, kwargs)
        traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs, warn=False)
    except Exception:
        coord_x = coord_y = None
        use_lonlat = False
        traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs, warn=False)

    # simplified color logic
    cmap_obj = plt.get_cmap(cmap)
    clusters = stops['cluster'] if 'cluster' in stops else range(len(stops))
    n = len(stops)
    colors = [cmap_obj((int(c) + 1) / (n + 1)) for c in clusters]

    # 1) diameter circles
    if coord_x and coord_y and not use_lonlat and 'diameter' in stops:
        geom = [
            Point(x, y).buffer(d / 2)
            for x, y, d in zip(
                stops[traj_cols[coord_x]],
                stops[traj_cols[coord_y]],
                stops['diameter']
            )
        ]
        gdf = gpd.GeoDataFrame(stops, geometry=geom, crs=crs or "EPSG:3857")
        if edge_only:
            gdf.plot(ax=ax, facecolor="none", edgecolor=colors, linewidth=4, alpha=0.8)
        else:
            gdf.plot(ax=ax, facecolor=colors, edgecolor='k', linewidth=0.75, alpha=0.8)
        return

    # 2) H3 hexagons
    loc = traj_cols['location_id']
    if loc in stops.columns and stops[loc].apply(h3.is_valid_cell).all():
        geom = stops[loc].map(h3_cell_to_polygon)
        gdf = gpd.GeoDataFrame(stops, geometry=geom, crs="EPSG:4326")
        if crs:
            gdf = gdf.to_crs(crs)
        # ----- stagger concentric hexes if duplicates -----
        if stagger:
            dup_mask = gdf.geometry.duplicated(keep=False)
            if dup_mask.any():
                wkt_series = gdf.geometry.apply(lambda g: g.wkt)
                for wkt_val, idx in wkt_series[dup_mask].groupby(wkt_series):
                    idx_mask = (wkt_series == wkt_val)
                    k = idx_mask.sum()
                    factors = np.linspace(1.0, 0.7, k)          # shrink 1.0 → 0.7
                    gdf.loc[idx_mask, 'geometry'] = [
                        scale(poly, xfact=f, yfact=f, origin=poly.centroid)
                        for poly, f in zip(gdf.loc[idx_mask, 'geometry'], factors)
                    ]
        if edge_only:
            gdf.plot(ax=ax, facecolor="none", edgecolor=colors, linewidth=4, alpha=0.8)
        else:
            gdf.plot(ax=ax, facecolor=colors, edgecolor='k', linewidth=0.75, alpha=0.8)
        return

    # 3) fallback scatter
    if coord_x and coord_y:
        ax.scatter(
            stops[traj_cols[coord_x]],
            stops[traj_cols[coord_y]],
            c=colors, s=60,
            edgecolor='k', linewidth=0.5, alpha=0.75
        )

def plot_time_barcode(ts_series, ax, current_idx=None, set_xlim=True):
    """
    Plot a barcode of timestamps on ax. Optionally highlight current_idx in red.
    If set_xlim is True, auto-sets x-axis to padded timestamp range.
    """
    ts_dt = pd.to_datetime(ts_series, unit='s')
    if set_xlim:
        pad = pd.Timedelta(minutes=20)
        ax.set_xlim(ts_dt.min() - pad, ts_dt.max() + pad)
    ax.vlines(ts_dt, 0.2, 0.8, colors='black', lw=0.5)
    if current_idx is not None:
        ax.vlines(ts_dt.iloc[current_idx], 0, 1, colors='red', lw=1.5)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    locator = mdates.AutoDateLocator(minticks=2, maxticks=4)
    formatter = mdates.DateFormatter('%I %p')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='x', labelsize=10)

def plot_stops_barcode(stops, ax, cmap='Reds', stop_color=None, set_xlim=True, traj_cols=None, **kwargs):
    """
    Plot colored stop intervals as bars on ax using temporal columns and colors by cluster with cmap.
    If set_xlim is True, auto-sets x-axis to padded range.
    """
    t_key, use_datetime = loader._fallback_time_cols_dt(stops.columns, traj_cols, kwargs)
    traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs)
    start = stops[traj_cols[t_key]] if use_datetime else pd.to_datetime(stops[traj_cols[t_key]], unit='s')

    end_col_present = loader._has_end_cols(stops.columns, traj_cols)
    duration_col_present = loader._has_duration_cols(stops.columns, traj_cols)
    
    end_t_key = 'end_datetime' if use_datetime else 'end_timestamp'
    if not (end_col_present or duration_col_present):
        raise ValueError("Missing required (end or duration) temporal columns for true_visits dataframe.")
    elif not end_col_present:
        end = stops[traj_cols[t_key]] + pd.to_timedelta(stops[traj_cols['duration']] * 60, unit='s')
    else:
        end = stops[traj_cols[end_t_key]] if use_datetime else pd.to_datetime(stops[traj_cols[end_t_key]], unit='s')
        
    clusters = np.arange(len(stops)) if 'cluster' not in stops else stops['cluster']
    n = len(stops)
    if stop_color:
        colors = [stop_color for c in clusters]
    elif cmap:
        colors = [plt.get_cmap(cmap)((c+1)/(n+1)) for c in clusters]
    else:
        raise ValueError("Specify either a color map (cmap) or a solid color (stop_color).")
        
    for s, e, color in zip(start, end, colors):
        ax.fill_betweenx([0, 1], s, e, color=color, alpha=0.75)
    if set_xlim:
        pad = pd.Timedelta(minutes=20)
        ax.set_xlim(start.min() - pad, end.max() + pad)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_yticklabels([])
    locator = mdates.AutoDateLocator(minticks=2, maxticks=4)
    ax.xaxis.set_major_locator(locator)
    formatter = mdates.DateFormatter('%I %p')
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='x', labelsize=10)
