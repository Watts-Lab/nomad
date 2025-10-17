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
pd.plotting.register_matplotlib_converters()

def h3_cell_to_polygon(cell):
    """Return shapely Polygon for H3 cell (lon/lat)."""
    coords = h3.cell_to_boundary(cell)
    lats, lons = zip(*coords)
    return Polygon(zip(lons, lats))

def clip_spatial_outliers(data, lower_quantile=0.01, upper_quantile=0.99, traj_cols=None, **kwargs):
    """
    Remove spatial outliers using quantile-based clipping.
    
    Parameters
    ----------
    data : pandas.DataFrame or GeoDataFrame
        Data with spatial columns.
    lower_quantile : float, default 0.01
        Lower quantile threshold (0-1).
    upper_quantile : float, default 0.99
        Upper quantile threshold (0-1).
    traj_cols : dict, optional
        Column mappings.
    **kwargs : dict
        Additional column mappings (e.g., latitude='lat').
    
    Returns
    -------
    DataFrame or GeoDataFrame
        Data with spatial outliers removed.
    """
    if isinstance(data, gpd.GeoDataFrame):
        x = data.geometry.x
        y = data.geometry.y
    else:
        coord_key1, coord_key2, _ = loader._fallback_spatial_cols(data.columns, traj_cols, kwargs)
        traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs, warn=False)
        x = data[traj_cols[coord_key1]]
        y = data[traj_cols[coord_key2]]
    
    x_min, x_max = x.quantile(lower_quantile), x.quantile(upper_quantile)
    y_min, y_max = y.quantile(lower_quantile), y.quantile(upper_quantile)
    
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    return data[mask]

def _plot_base_geometry(ax, base_geometry, gdf_crs, base_geom_color='#2c353c', base_geom_background='#0e0e0e'):
    """Helper to plot base geometry layer."""
    if base_geometry.crs is not None and gdf_crs is not None:
        if base_geometry.crs != gdf_crs:
            raise ValueError(f"CRS mismatch: base_geometry={base_geometry.crs}, data={gdf_crs}")
    
    ax.set_facecolor(base_geom_background)
    
    if isinstance(base_geom_color, str) and base_geom_color in base_geometry.columns:
        base_geometry.plot(ax=ax, column=base_geom_color, edgecolor='white', 
                          linewidth=0.5, zorder=0, autolim=False)
    else:
        base_geometry.plot(ax=ax, color=base_geom_color, edgecolor='white', 
                          linewidth=0.5, zorder=0, autolim=False)

def _map_cluster_colors(gdf, cmap):
    """Map cluster IDs to colors, filtering noise."""
    gdf = gdf.loc[gdf['cluster'] != -1].copy()
    n = gdf['cluster'].nunique() or 1
    cmap_obj = plt.get_cmap(cmap)
    return gdf, gdf['cluster'].map(lambda c: cmap_obj((int(c) + 1) / (n + 1)))

def plot_pings(data, ax,
               color='black',
               cmap=None,
               s=6,
               marker='o',
               alpha=1,
               base_geometry=None,
               base_geom_color='#2c353c',
               base_geom_background='#0e0e0e',
               data_crs=None,
               traj_cols=None,
               **kwargs):
    """
    Plot trajectory points.

    Parameters
    ----------
    data : pandas.DataFrame or GeoDataFrame
        Data with spatial columns (x/y or lon/lat).
    ax : matplotlib.axes.Axes
        Axis to draw on.
    color : color or str, default 'black'
        Point color. Can be color string or column name. If column name and values
        are integers (e.g., clusters), cmap is required to map to colors.
    cmap : str or Colormap, optional
        Colormap when color is a column with integer values.
    s : float or array-like, default 6
        Marker size in points².
    marker : str, default 'o'
        Marker style.
    alpha : float, default 1
        Opacity.
    base_geometry : GeoDataFrame, optional
        Base layer to plot underneath.
    base_geom_color : str or column name, default '#2c353c'
        Color for base geometry.
    base_geom_background : color, default '#0e0e0e'
        Axes background color.
    data_crs : str or CRS, optional
        CRS of the data.
    traj_cols : dict, optional
        Column name mappings.
    **kwargs : dict
        Additional column mappings.
    
    Returns
    -------
    matplotlib collection
    """
    if isinstance(data, gpd.GeoDataFrame):
        gdf = data
        if data_crs is not None and gdf.crs is None:
            gdf = gdf.set_crs(data_crs)
    else:
        coord_key1, coord_key2, use_lon_lat = loader._fallback_spatial_cols(data.columns, traj_cols, kwargs)
        traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs, warn=False)
        xcol, ycol = traj_cols[coord_key1], traj_cols[coord_key2]
        if data_crs is None and use_lon_lat:
            data_crs = "EPSG:4326"
        gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data[xcol], data[ycol]), crs=data_crs)
    
    if base_geometry is not None:
        _plot_base_geometry(ax, base_geometry, gdf.crs, base_geom_color, base_geom_background)
    
    # Hide axes elements but keep facecolor
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    
    # Color logic: if column name with integer values (clusters), use cmap
    if isinstance(color, str) and color in data.columns:
        if cmap:
            return gdf.plot(ax=ax, column=color, cmap=cmap, markersize=s,
                           marker=marker, alpha=alpha, zorder=2).collections[-1]
        else:
            # Column contains actual colors
            return gdf.plot(ax=ax, color=data[color], markersize=s,
                           marker=marker, alpha=alpha, zorder=2).collections[-1]
    
    return gdf.plot(ax=ax, color=color, markersize=s,
                   marker=marker, alpha=alpha, zorder=2).collections[-1]


def plot_circles(data, ax,
                 radius=None,
                 diameter=None,
                 color=None,
                 edgecolor=None,
                 cmap=None,
                 alpha=0.75,
                 linewidth=None,
                 base_geometry=None,
                 base_geom_color='#2c353c',
                 base_geom_background='#0e0e0e',
                 data_crs=None,
                 traj_cols=None,
                 **kwargs):
    """
    Plot buffered circles. Automatically projects geographic coordinates to Mercator.
    
    Parameters
    ----------
    data : pandas.DataFrame or GeoDataFrame
        Data with spatial columns.
    ax : matplotlib.axes.Axes
        Axis to draw on.
    radius : float or str, optional
        Circle radius in meters (scalar or column name).
    diameter : float or str, optional
        Circle diameter in meters (scalar or column name).
    color : color, str, or None, default None
        Face color. None = edge-only. 'cluster' requires cmap.
    edgecolor : color, str, or None, default None
        Edge color. 'cluster' requires cmap.
    cmap : str or Colormap, optional
        Colormap when coloring by integer column (e.g., 'cluster').
    alpha : float, default 0.75
        Opacity.
    linewidth : float, optional
        Edge width (auto: 2 if edge-only, 0.5 otherwise).
    base_geometry : GeoDataFrame, optional
        Base layer underneath.
    base_geom_color : str or column, default '#2c353c'
        Base geometry color.
    base_geom_background : color, default '#0e0e0e'
        Axes background color.
    data_crs : str or CRS, optional
        Data CRS.
    traj_cols : dict, optional
        Column mappings.
    **kwargs : dict
        Additional column mappings.
    
    Returns
    -------
    matplotlib collection
    """
    if radius is None and diameter is None:
        raise ValueError("Must provide either radius or diameter")
    if radius is not None and diameter is not None:
        raise ValueError("Cannot provide both radius and diameter")
    
    if isinstance(data, gpd.GeoDataFrame):
        gdf = data if data_crs is None else data.set_crs(data_crs) if data.crs is None else data
        use_lon_lat = gdf.crs and gdf.crs.is_geographic
    else:
        coord_key1, coord_key2, use_lon_lat = loader._fallback_spatial_cols(data.columns, traj_cols, kwargs)
        traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs, warn=False)
        if data_crs is None and use_lon_lat:
            data_crs = "EPSG:4326"
        gdf = gpd.GeoDataFrame(data, 
                              geometry=gpd.points_from_xy(data[traj_cols[coord_key1]], 
                                                         data[traj_cols[coord_key2]]),
                              crs=data_crs)
    
    # Check base_geometry CRS before any projection
    original_crs = gdf.crs
    if base_geometry is not None:
        _plot_base_geometry(ax, base_geometry, original_crs, base_geom_color, base_geom_background)
    
    # Project to Mercator if geographic coordinates (for accurate buffering in meters)
    if use_lon_lat:
        gdf = gdf.to_crs("EPSG:3857")
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    
    if diameter is not None:
        r = (data[diameter] if isinstance(diameter, str) and diameter in data.columns else diameter) / 2
    else:
        r = data[radius] if isinstance(radius, str) and radius in data.columns else radius
    gdf = gdf.copy()
    gdf['geometry'] = gdf.geometry.buffer(r) if np.isscalar(r) else [pt.buffer(rv) for pt, rv in zip(gdf.geometry, r)]
    
    # Convert back to original CRS if we projected
    if use_lon_lat and original_crs:
        gdf = gdf.to_crs(original_crs)
    
    # Color logic: 'cluster' special case requires cmap
    if color == 'cluster' and 'cluster' in data.columns:
        if not cmap:
            raise ValueError("cmap required when color='cluster'")
        gdf, face_colors = _map_cluster_colors(gdf, cmap)
        edge_colors, column = edgecolor or 'black', None
    elif edgecolor == 'cluster' and 'cluster' in data.columns:
        if not cmap:
            raise ValueError("cmap required when edgecolor='cluster'")
        gdf, edge_colors = _map_cluster_colors(gdf, cmap)
        face_colors, column = color, None
    elif isinstance(color, str) and color in data.columns:
        if cmap:
            face_colors, edge_colors, column = None, edgecolor or 'black', color
        else:
            face_colors, edge_colors, column = data[color], edgecolor or 'black', None
    elif isinstance(edgecolor, str) and edgecolor in data.columns:
        if cmap:
            face_colors, edge_colors, column = color, None, edgecolor
        else:
            face_colors, edge_colors, column = color, data[edgecolor], None
    else:
        face_colors, edge_colors, column = color, edgecolor or ('gray' if not color else 'black'), None
    
    lw = linewidth or (2 if face_colors is None else 0.5)
    plot_kwargs = {'ax': ax, 'alpha': alpha, 'zorder': 1, 'linewidth': lw}
    
    if column:
        plot_kwargs.update({'column': column, 'cmap': cmap})
    
    if face_colors is None:
        return gdf.plot(facecolor='none', edgecolor=edge_colors, **plot_kwargs).collections[-1]
    elif column:
        return gdf.plot(edgecolor=edge_colors, **plot_kwargs).collections[-1]
    else:
        return gdf.plot(color=face_colors, edgecolor=edge_colors, **plot_kwargs).collections[-1]

def plot_hexagons(data, ax,
                  h3_resolution=None,
                  color=None,
                  edgecolor=None,
                  cmap=None,
                  alpha=0.8,
                  linewidth=0.75,
                  stagger=True,
                  base_geometry=None,
                  base_geom_color='#2c353c',
                  base_geom_background='#0e0e0e',
                  data_crs=None,
                  traj_cols=None,
                  **kwargs):
    """
    Plot H3 hexagons.
    
    Parameters
    ----------
    data : pandas.DataFrame or GeoDataFrame
        Data with spatial columns or location_id/h3_cell column.
    ax : matplotlib.axes.Axes
        Axis to draw on.
    h3_resolution : int, optional
        H3 resolution. If None, expects location_id column with H3 cells.
    color : color, str, or None, default None
        Face color. 'cluster' requires cmap.
    edgecolor : color, str, or None, default None  
        Edge color. 'cluster' requires cmap.
    cmap : str or Colormap, optional
        Colormap for integer columns.
    alpha : float, default 0.8
        Opacity.
    linewidth : float, default 0.75
        Edge width.
    stagger : bool, default True
        Stagger duplicate hexagons concentrically.
    base_geometry : GeoDataFrame, optional
        Base layer underneath.
    base_geom_color : str or column, default '#2c353c'
        Base geometry color.
    base_geom_background : color, default '#0e0e0e'
        Axes background.
    data_crs : str or CRS, optional
        Target CRS (default EPSG:4326).
    traj_cols : dict, optional
        Column mappings.
    **kwargs : dict
        Additional column mappings.
    
    Returns
    -------
    matplotlib collection
    """
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs, warn=False) if not isinstance(data, gpd.GeoDataFrame) else traj_cols
    loc_col = traj_cols.get('location_id', 'location_id') if traj_cols else 'location_id'
    
    if h3_resolution is not None:
        coord_key1, coord_key2, use_lon_lat = loader._fallback_spatial_cols(data.columns, traj_cols, kwargs)
        if not use_lon_lat:
            raise ValueError("H3 requires geographic coordinates (lat/lon)")
        traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs, warn=False)
        lat_col, lon_col = traj_cols[coord_key2], traj_cols[coord_key1]
        df = data.copy()
        df['h3_cell'] = df.apply(lambda row: h3.latlng_to_cell(row[lat_col], row[lon_col], h3_resolution), axis=1)
        loc_col = 'h3_cell'
    elif loc_col not in data.columns:
        raise ValueError(f"Column '{loc_col}' not found and h3_resolution not provided")
    else:
        df = data
    
    if not df[loc_col].apply(h3.is_valid_cell).all():
        raise ValueError(f"Column '{loc_col}' contains invalid H3 cells")
    
    gdf = gpd.GeoDataFrame(df, geometry=df[loc_col].map(h3_cell_to_polygon), crs="EPSG:4326")
    if data_crs:
        gdf = gdf.to_crs(data_crs)
    
    if base_geometry is not None:
        _plot_base_geometry(ax, base_geometry, gdf.crs, base_geom_color, base_geom_background)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    
    if stagger:
        dup_mask = gdf.geometry.duplicated(keep=False)
        if dup_mask.any():
            wkt_series = gdf.geometry.apply(lambda g: g.wkt)
            for wkt_val in wkt_series[dup_mask].unique():
                idx_mask = wkt_series == wkt_val
                k = idx_mask.sum()
                factors = np.linspace(1.0, 0.7, k)
                gdf.loc[idx_mask, 'geometry'] = [
                    scale(poly, xfact=f, yfact=f, origin=poly.centroid)
                    for poly, f in zip(gdf.loc[idx_mask, 'geometry'], factors)
                ]
    
    # Color logic
    if color == 'cluster' and 'cluster' in df.columns:
        if not cmap:
            raise ValueError("cmap required when color='cluster'")
        gdf, face_colors = _map_cluster_colors(gdf, cmap)
        edge_colors, column = edgecolor or 'black', None
    elif edgecolor == 'cluster' and 'cluster' in df.columns:
        if not cmap:
            raise ValueError("cmap required when edgecolor='cluster'")
        gdf, edge_colors = _map_cluster_colors(gdf, cmap)
        face_colors, column = color, None
    elif isinstance(color, str) and color in df.columns:
        if cmap:
            face_colors, edge_colors, column = None, edgecolor or 'black', color
        else:
            face_colors, edge_colors, column = df[color], edgecolor or 'black', None
    elif isinstance(edgecolor, str) and edgecolor in df.columns:
        if cmap:
            face_colors, edge_colors, column = color, None, edgecolor
        else:
            face_colors, edge_colors, column = color, df[edgecolor], None
    else:
        face_colors, edge_colors, column = color, edgecolor or 'black', None
    
    plot_kwargs = {'ax': ax, 'alpha': alpha, 'zorder': 1, 'linewidth': linewidth}
    if column:
        plot_kwargs.update({'column': column, 'cmap': cmap})
    
    if face_colors is None:
        return gdf.plot(facecolor='none', edgecolor=edge_colors, **plot_kwargs).collections[-1]
    elif column:
        return gdf.plot(edgecolor=edge_colors, **plot_kwargs).collections[-1]
    else:
        return gdf.plot(color=face_colors, edgecolor=edge_colors, **plot_kwargs).collections[-1]


def plot_stops(stops, ax,
               radius=None,
               diameter=None,
               cmap='Reds',
               edge_only=False,
               stagger=True,
               base_geometry=None,
               base_geom_color='#2c353c',
               base_geom_background='#0e0e0e',
               data_crs=None,
               traj_cols=None,
               **kwargs):
    """
    Plot stop table (circles, hexagons, or points).
    
    Parameters
    ----------
    stops : pandas.DataFrame or GeoDataFrame
        Stop data.
    ax : matplotlib.axes.Axes
        Axis to draw on.
    radius : float or str, optional
        Circle radius in meters (scalar or column name).
    diameter : float or str, optional
        Circle diameter in meters (scalar or column name).
    cmap : str or Colormap, default 'Reds'
        Colormap for cluster coloring.
    edge_only : bool, default False
        If True, hollow circles/hexagons (edge-only).
    stagger : bool, default True
        Stagger duplicate hexagons.
    base_geometry : GeoDataFrame, optional
        Base layer underneath.
    base_geom_color : str or column, default '#2c353c'
        Base geometry color.
    base_geom_background : color, default '#0e0e0e'
        Axes background.
    data_crs : str or CRS, optional
        Data CRS.
    traj_cols : dict, optional
        Column mappings.
    **kwargs : dict
        Additional column mappings.
    
    Returns
    -------
    matplotlib collection or None
    """
    try:
        coord_x, coord_y, use_lonlat = loader._fallback_spatial_cols(stops.columns, traj_cols, kwargs)
        traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs, warn=False)
    except Exception:
        coord_x = coord_y = use_lonlat = None
        traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs, warn=False)
    
    # Determine size parameter (prefer explicit args over columns)
    size_param = (radius is not None) or (diameter is not None) or ('diameter' in stops.columns)
    
    # Circles (any coordinates - will auto-project if geographic)
    if coord_x and coord_y and size_param:
        # Use explicit radius/diameter if provided, otherwise use diameter column
        if diameter is not None:
            return plot_circles(
                stops, ax,
                diameter=diameter,
                color=None if edge_only else 'cluster',
                edgecolor='cluster' if edge_only else 'black',
                cmap=cmap,
                alpha=0.8,
                linewidth=4 if edge_only else 0.75,
                base_geometry=base_geometry,
                base_geom_color=base_geom_color,
                base_geom_background=base_geom_background,
                data_crs=data_crs,
                traj_cols=traj_cols,
                **kwargs
            )
        else:  # radius provided or diameter column
            return plot_circles(
                stops, ax,
                radius=radius,
                diameter='diameter' if radius is None else None,
                color=None if edge_only else 'cluster',
                edgecolor='cluster' if edge_only else 'black',
                cmap=cmap,
                alpha=0.8,
                linewidth=4 if edge_only else 0.75,
                base_geometry=base_geometry,
                base_geom_color=base_geom_color,
                base_geom_background=base_geom_background,
                data_crs=data_crs,
                traj_cols=traj_cols,
                **kwargs
            )
    
    # H3 hexagons
    loc_col = traj_cols.get('location_id', 'location_id') if traj_cols else 'location_id'
    if loc_col in stops.columns and stops[loc_col].apply(h3.is_valid_cell).all():
        return plot_hexagons(
            stops, ax,
            h3_resolution=None,
            color=None if edge_only else 'cluster',
            edgecolor='cluster' if edge_only else 'black',
            cmap=cmap,
            alpha=0.8,
            linewidth=4 if edge_only else 0.75,
            stagger=stagger,
            base_geometry=base_geometry,
            base_geom_color=base_geom_color,
            base_geom_background=base_geom_background,
            data_crs=data_crs,
            traj_cols=traj_cols,
            **kwargs
        )
    
    # Fallback scatter
    if coord_x and coord_y:
        return plot_pings(
            stops, ax,
            color='cluster',
            cmap=cmap,
            s=60,
            alpha=0.75,
            base_geometry=base_geometry,
            base_geom_color=base_geom_color,
            base_geom_background=base_geom_background,
            data_crs=data_crs,
            traj_cols=traj_cols,
            **kwargs
        )

def plot_time_barcode(ts_series, ax, current_idx=None, color=None, set_xlim=True):
    """
    Plot a barcode of timestamps on ax. Optionally highlight current_idx in red.
    If set_xlim is True, auto-sets x-axis to padded timestamp range.
    """
    ts_dt = pd.to_datetime(ts_series, unit='s')
    if set_xlim:
        pad = pd.Timedelta(minutes=20)
        ax.set_xlim(ts_dt.min() - pad, ts_dt.max() + pad)
    vlines = ax.vlines(ts_dt, 0.2, 0.8, colors='black', lw=0.5)
    if current_idx is not None:
        ax.vlines(ts_dt.iloc[current_idx], 0, 1, colors='red', lw=1.5)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    
    # Choose appropriate locator and formatter based on time range
    time_range = (ts_dt.max() - ts_dt.min()).total_seconds()
    
    if time_range <= 3600 * 12:  # Up to 12 hours - hourly major ticks
        major_locator = mdates.HourLocator(interval=1)
        formatter = mdates.DateFormatter('%I %p')
        minor_locator = None
        
    elif time_range <= 3600 * 72:  # Up to 3 days - 6-hour major ticks
        major_locator = mdates.HourLocator(interval=6)
        formatter = mdates.DateFormatter('%I %p')
        minor_locator = mdates.HourLocator(interval=1)  # Hourly minor ticks
        
    else:  # Longer than 3 days - daily major ticks with weekday
        major_locator = mdates.DayLocator()
        formatter = mdates.DateFormatter('%a\n%I %p')  # Weekday letter + time
        minor_locator = mdates.HourLocator(interval=6)  # 6-hour minor ticks
    
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(formatter)
    
    if minor_locator is not None:
        ax.xaxis.set_minor_locator(minor_locator)
        ax.tick_params(axis='x', which='minor', length=3)
    
    ax.tick_params(axis='x', which='major', labelsize=10)
    return vlines

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
        # end = stops[traj_cols[t_key]] + pd.to_timedelta(stops[traj_cols['duration']] * 60, unit='s')
        start = pd.to_datetime(stops[traj_cols[t_key]], unit='s')
        end = start + pd.to_timedelta(stops[traj_cols['duration']] * 60, unit='s')
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
    
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    if set_xlim:
        pad = pd.Timedelta(minutes=20)
        ax.set_xlim(start.min() - pad, end.max() + pad)
        
        # Use same tick logic as plot_time_barcode for consistency
        time_range = (end.max() - start.min()).total_seconds()
        
        if time_range <= 3600 * 12:  # Up to 12 hours - hourly major ticks
            major_locator = mdates.HourLocator(interval=1)
            formatter = mdates.DateFormatter('%I %p')
            minor_locator = None
            
        elif time_range <= 3600 * 72:  # Up to 3 days - 6-hour major ticks
            major_locator = mdates.HourLocator(interval=6)
            formatter = mdates.DateFormatter('%I %p')
            minor_locator = mdates.HourLocator(interval=1)  # Hourly minor ticks
            
        else:  # Longer than 3 days - daily major ticks with weekday
            major_locator = mdates.DayLocator()
            formatter = mdates.DateFormatter('%a\n%I %p')  # Weekday letter + time
            minor_locator = mdates.HourLocator(interval=6)  # 6-hour minor ticks
        
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(formatter)
        
        if minor_locator is not None:
            ax.xaxis.set_minor_locator(minor_locator)
            ax.tick_params(axis='x', which='minor', length=3)
        
        ax.tick_params(axis='x', which='major', labelsize=10)
