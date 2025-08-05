import numpy as np
import pandas as pd
from nomad.constants import SEC_PER_UNIT
from nomad.stop_detection import utils
import nomad.io.base as loader

def _centroid(coords, metric='euclidean', weight=None):
    """
    Calculate the centroid (arithmetic mean) of a set of coordinates.
    For 'haversine', computes the spherical centroid (lat, lon in degrees).
    If weight is given, computes the weighted centroid.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of coordinates, each row a point.
    metric : str, optional
        'euclidean' or 'haversine'. For 'haversine', coords should be in degrees.
    weight : array-like, optional
        Non-negative weights for each coordinate (length = len(coords)).

    Returns
    -------
    numpy.ndarray
        The centroid as a single point (same shape as input).
    """
    if len(coords) < 2:
        return coords[0]
    coords = np.asarray(coords)
    if weight is not None:
        weight = np.asarray(weight)
        weight = weight / weight.sum()
    else:
        weight = np.full(len(coords), 1.0 / len(coords))

    if metric == 'haversine':
        rad = np.radians(coords)
        lat, lon = rad[:, 0], rad[:, 1]
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        x_mean = np.sum(x * weight)
        y_mean = np.sum(y * weight)
        z_mean = np.sum(z * weight)
        hyp = np.hypot(x_mean, y_mean)
        lat_c = np.arctan2(z_mean, hyp)
        lon_c = np.arctan2(y_mean, x_mean)
        return np.degrees([lat_c, lon_c])
    else:
        return np.sum(coords * weight[:, None], axis=0)

def rog(stops, agg_freq='D', weighted=True, traj_cols=None, **kwargs):
    """
    Compute radius of gyration per bucket (and per user, if present).

    Parameters
    ----------
    stops : pd.DataFrame
    agg_freq : str
        Pandas offset alias for time‐bucketing (e.g. 'D','W','M').
    weighted : bool
        If True, weight by duration; else unweighted.
    traj_cols : dict, optional
        Mapping for x/y (or lon/lat), timestamp/datetime, duration, user_id.

    Returns
    -------
    pd.DataFrame
        Columns = [bucket, user_id? , rog].
    """
    # 1) column mapping + check
    t_key, coord_x, coord_y, use_datetime, use_lon_lat = utils._fallback_st_cols(
        stops.columns, traj_cols, kwargs
    )
    traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs, warn=False)
    dur_key = traj_cols['duration']
    if dur_key not in stops.columns:
        raise ValueError("Missing required 'duration' column")

    df = stops.copy()

    # 2) time buckets
    if use_datetime:
        df['period'] = df[traj_cols[t_key]].dt.to_period(agg_freq).dt.to_timestamp()
    else:
        step = SEC_PER_UNIT[agg_freq.lower()]
        df['period'] = (df[traj_cols[t_key]] // step) * step

    # 3) grouping keys
    keys = ['period']
    uid_key = traj_cols['user_id']
    if uid_key in df.columns:
        keys.append(uid_key)

    # 4) compute per‐group centroid
    metric = 'haversine' if use_lon_lat else 'euclidean'
    def _group_centroid(g):
        # for haversine we need [lat, lon] order
        if metric == 'haversine':
            pts = g[[traj_cols['latitude'], traj_cols['longitude']]].to_numpy()
        else:
            pts = g[[coord_x, coord_y]].to_numpy()
        w = g[dur_key].to_numpy() if weighted else None
        return _centroid(pts, metric=metric, weight=w)

    cent = df.groupby(keys).apply(_group_centroid)
    # unpack into DataFrame
    cent_df = pd.DataFrame(
        cent.tolist(),
        index=cent.index,
        columns=(['lat_c','lon_c'] if metric=='haversine' else [coord_x+'_c',coord_y+'_c'])
    )

    # 5) join back and compute squared distances
    df = df.join(cent_df, on=keys)
    if metric == 'haversine':
        # convert to radians
        def _sq_hav(r):
            c1 = np.radians([r['lat_c'], r['lon_c']])
            p1 = np.radians([r[traj_cols['latitude']], r[traj_cols['longitude']]])
            d = utils._haversine_distance(p1, c1)
            return d*d
        df['d2'] = df.apply(_sq_hav, axis=1)
    else:
        dx = df[coord_x] - df[coord_x+'_c']
        dy = df[coord_y] - df[coord_y+'_c']
        df['d2'] = dx*dx + dy*dy

    # 6) aggregate into rog
    if weighted:
        rog = df.groupby(keys).apply(
            lambda g: np.sqrt((g['d2'] * g[dur_key]).sum() / g[dur_key].sum())
        )
    else:
        rog = np.sqrt(df.groupby(keys)['d2'].mean())

    return rog.reset_index(name='rog')