import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from nomad.io.base import _parse_traj_cols

pd.plotting.register_matplotlib_converters()

def adjust_zoom(x, y, ax, buffer=0.5):
    if buffer is not None:
        x0, x1 = x.quantile(0.01), x.quantile(0.99)
        y0, y1 = y.quantile(0.01), y.quantile(0.99)
        pad_x = (x1 - x0) * buffer / 2
        pad_y = (y1 - y0) * buffer / 2
        ax.set_xlim(x0 - pad_x, x1 + pad_x)
        ax.set_ylim(y0 - pad_y, y1 + pad_y)

def plot_pings(pings_df, ax, radius=1, point_color='black', cmap='twilight', traj_cols=None, **kwargs):
    """
    Plot pings as true-radius circles and centers. 
    If 'cluster' exists, colors by cluster (noise/cluster==-1 transparent).
    If no 'cluster', just plots points.
    """
    traj_cols = _parse_traj_cols(pings_df.columns, traj_cols, kwargs)
    x, y = traj_cols['x'], traj_cols['y']
    if 'cluster' not in pings_df:
        ax.scatter(pings_df[x], pings_df[y], s=6, color=point_color, alpha=1)
        return

    gdf = gpd.GeoDataFrame(
        pings_df.copy(),
        geometry=[Point(x0, y0) for x0, y0 in zip(pings_df[x], pings_df[y])],
        crs="EPSG:3857"
    )
    valid = gdf['cluster'] != -1
    n = gdf.loc[valid, 'cluster'].nunique() or 1
    cmap_obj = plt.get_cmap(cmap)
    def color_func(c):
        if c == -1:
            return (0,0,0,0)
        return cmap_obj((c+1)/(n+1))
    colors = gdf['cluster'].map(color_func)
    gdf.geometry.buffer(radius).plot(ax=ax, color=colors, alpha=0.8, linewidth=0)
    gdf.plot(ax=ax, color=point_color, markersize=3, linewidth=0)

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

def plot_stops_barcode(stops, ax, cmap='Reds', set_xlim=True, traj_cols=None, **kwargs):
    """
    Plot colored stop intervals as bars on ax using 'unix_ts', 'end_timestamp', and colors by cluster with cmap.
    If set_xlim is True, auto-sets x-axis to padded range.
    """
    traj_cols = _parse_traj_cols(stops.columns, traj_cols, kwargs)
    unix_ts = traj_cols['timestamp'] if 'timestamp' in traj_cols else 'unix_ts'
    end_ts = traj_cols.get('end_timestamp', 'end_timestamp')
    cluster = traj_cols.get('cluster', 'cluster')

    start = pd.to_datetime(stops[unix_ts], unit='s')
    end = pd.to_datetime(stops[end_ts], unit='s')
    if cluster in stops:
        clusters = stops[cluster]
        n = clusters.nunique() or 1
        cmap_obj = plt.get_cmap(cmap)
        colors = [cmap_obj((c+1)/(n+1)) for c in clusters]
    else:
        colors = ['#aaa'] * len(stops)

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
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%I %p'))
    ax.tick_params(axis='x', labelsize=10)