import numpy as np
import pandas as pd
from nomad.constants import SEC_PER_UNIT
from nomad.stop_detection import utils
import nomad.io.base as loader
import warnings
import pdb

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

def rog(stops, agg_freq='d', weighted=True, traj_cols=None, time_weights=None, exploded=True, **kwargs):
    """
    if weighted AND time_weights, then
        1. weights = duration * time_weights
    if (weighted) AND (time_weights is None and not found in colunmns), then
        2. weights = duration
    if (weighted) AND (time_weights is None and found in columns), then
        3. weights = duration * time_weights
    if NOT weighted, then
        4. weights = 1
            
    Compute radius of gyration per bucket (and per user, if present).

    Parameters
    ----------
    stops : pd.DataFrame
    agg_freq : str
        Pandas offset alias for time‐bucketing (e.g. 'd','w','m').
    weighted : bool
        If True, weight by duration; else unweighted.
    traj_cols : dict, optional
        Mapping for x/y (or lon/lat), timestamp/datetime, duration, user_id.
    time_weights : pd.Series, optional
        If None or 1 and weighted is True, weights = duration. Otherwise, stops have weights = time_weights * duration.
    weight_freq : str, optional
        'D' for daily, 'H' for hourly weights. Default is 'D'.

    Returns
    -------
    pd.DataFrame
        Columns = [bucket, user_id? , rog].
    """
    stops = stops.copy()

    # Restrict agg_freq to days and weeks only
    allowed_freqs = ['d', 'w', 'D', 'W']  # Allow both cases
    if agg_freq not in allowed_freqs:
        raise ValueError(f"agg_freq must be one of {allowed_freqs} (got '{agg_freq}')")

    # Add time_weights column if provided and matches index
    if time_weights is not None:
        if isinstance(time_weights, pd.Series) and (len(time_weights) == len(stops)):
            stops['time_weights'] = time_weights
        else:
            raise ValueError("time_weights must be a pd.Series with the same length and index as stops.")

    if exploded:
        stops = utils.explode_stops(stops, agg_freq=agg_freq, **kwargs)
        warnings.warn(
            f"Some stops straddle multiple {agg_freq.upper()}s. They will be exploded into separate rows.",
            UserWarning
        )
    
    # 1) column mapping + check
    t_key, coord_x, coord_y, use_datetime, use_lon_lat = utils._fallback_st_cols(stops.columns, traj_cols, kwargs)
    traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs, warn=False)
    dur_key = traj_cols['duration']
    if dur_key not in stops.columns:
        raise ValueError("Missing required 'duration' column")

    stops = stops.copy()
    
    # 2) time buckets
    if use_datetime:
        temp_dt = stops[traj_cols[t_key]]
    else:
        temp_dt = pd.to_datetime(stops[traj_cols[t_key]], unit='s') 
    if agg_freq == "W":
        agg_freq = "W-MON"

    stops['period'] = temp_dt.dt.to_period(agg_freq).dt.start_time

    # 3) grouping keys
    keys = ['period']
    uid_key = traj_cols['user_id']
    if uid_key in stops.columns:
        keys.append(uid_key)

    # 4) compute per‐group centroid
    metric = 'haversine' if use_lon_lat else 'euclidean'

    def _group_centroid(g):
        if metric == 'haversine':
            pts = g[[traj_cols['latitude'], traj_cols['longitude']]].to_numpy()
        else:
            pts = g[[coord_x, coord_y]].to_numpy()
        # Weight logic
        if weighted:
            if time_weights is not None:
                w = g[dur_key].to_numpy() * time_weights.loc[g.index].to_numpy()
            elif 'time_weights' in g.columns:
                w = g[dur_key].to_numpy() * g['time_weights'].to_numpy()
            else:
                w = g[dur_key].to_numpy()
        else:
            w = None
        return _centroid(pts, metric=metric, weight=w)

    cent = stops.groupby(keys).apply(_group_centroid, include_groups=False)
    
    cent_df = pd.DataFrame(
        cent.tolist(),
        index=cent.index,
        columns=(['lat_c','lon_c'] if metric=='haversine' else [coord_x+'_c',coord_y+'_c'])
    )

    # 5) join back and compute squared distances
    stops = stops.join(cent_df, on=keys)
    
    if metric == 'haversine':
        def _sq_hav(r):
            c1 = np.radians([r['lat_c'], r['lon_c']])
            p1 = np.radians([r[traj_cols['latitude']], r[traj_cols['longitude']]])
            d = utils._haversine_distance(p1, c1)
            return d*d
        stops['d2'] = stops.apply(_sq_hav, axis=1)
    else:
        dx = stops[coord_x] - stops[coord_x+'_c']
        dy = stops[coord_y] - stops[coord_y+'_c']
        stops['d2'] = dx*dx + dy*dy

    # 6) aggregate into rog
    def _group_rog(g):
        if weighted:
            if time_weights is not None:
                weights = g[dur_key] * time_weights.loc[g.index]
            elif 'time_weights' in g.columns:
                weights = g[dur_key] * g['time_weights']
            else:
                weights = g[dur_key]
            return np.sqrt((g['d2'] * weights).sum() / weights.sum())
        else:
            return np.sqrt(g['d2'].mean())

    rog = stops.groupby(keys).apply(_group_rog, include_groups=False)
    return rog.reset_index(name='rog')