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

    cent = stops.groupby(keys).apply(_group_centroid)
    
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

    rog = stops.groupby(keys).apply(_group_rog)
    return rog.reset_index(name='rog')

def self_containment(stops, threshold, agg_freq='d', weighted=True, home_activity_type='Home',
                     activity_type_col='activity_type', traj_cols=None, time_weights=None,
                     exploded=True, **kwargs):
    """
    Compute self-containment (proportion of non-home time spent within threshold distance from home).

    Self-containment describes the propensity of individuals to stay close to home. It is calculated
    as the time-weighted proportion of non-home activities that are within a threshold distance from home.

    Parameters
    ----------
    stops : pd.DataFrame
        Stop data with spatial coordinates, duration, and activity type.
    threshold : float
        Distance threshold in the same units as coordinates (meters for projected, degrees for lat/lon).
        Activities within this distance from home are considered "contained".
    agg_freq : str
        Pandas offset alias for time-bucketing (e.g. 'd','w','m').
    weighted : bool
        If True, weight by duration; else unweighted (count activities).
    home_activity_type : str
        Value in activity_type_col that identifies home locations. Default is 'Home'.
    activity_type_col : str
        Column name containing activity type. Default is 'activity_type'.
    traj_cols : dict, optional
        Mapping for x/y (or lon/lat), timestamp/datetime, duration, user_id.
    time_weights : pd.Series, optional
        Additional time weights to multiply with duration (if weighted=True).
    exploded : bool
        If True, explode stops that straddle multiple time periods. Default is True.
    **kwargs
        Additional arguments passed to explode_stops.

    Returns
    -------
    pd.DataFrame
        Columns = [period, user_id?, self_containment].
        self_containment is the proportion [0, 1] of non-home time spent within threshold from home.
    """
    stops = stops.copy()

    # Restrict agg_freq to days and weeks only
    allowed_freqs = ['d', 'w', 'D', 'W']
    if agg_freq not in allowed_freqs:
        raise ValueError(f"agg_freq must be one of {allowed_freqs} (got '{agg_freq}')")

    # Check for required columns
    if activity_type_col not in stops.columns:
        raise ValueError(f"Missing required '{activity_type_col}' column")

    # Add time_weights column if provided
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

    # 1) Column mapping + check
    t_key, coord_x, coord_y, use_datetime, use_lon_lat = utils._fallback_st_cols(stops.columns, traj_cols, kwargs)
    traj_cols = loader._parse_traj_cols(stops.columns, traj_cols, kwargs, warn=False)
    dur_key = traj_cols['duration']
    if dur_key not in stops.columns:
        raise ValueError("Missing required 'duration' column")

    # 2) Time buckets
    if use_datetime:
        temp_dt = stops[traj_cols[t_key]]
    else:
        temp_dt = pd.to_datetime(stops[traj_cols[t_key]], unit='s')
    if agg_freq == "W":
        agg_freq = "W-MON"

    stops['period'] = temp_dt.dt.to_period(agg_freq).dt.start_time

    # 3) Grouping keys
    keys = ['period']
    uid_key = traj_cols['user_id']
    if uid_key in stops.columns:
        keys.append(uid_key)

    # 4) Calculate distance from home for each group
    metric = 'haversine' if use_lon_lat else 'euclidean'

    # Initialize distance column
    stops['dist_from_home'] = np.nan

    # Calculate for each group
    for group_keys_tuple, group_df in stops.groupby(keys):
        home_stops = group_df[group_df[activity_type_col] == home_activity_type]

        if len(home_stops) == 0:
            # No home location in this group
            continue

        # Use the first home location as reference
        if metric == 'haversine':
            home_coords = home_stops[[traj_cols['latitude'], traj_cols['longitude']]].iloc[0].values
            home_coords_rad = np.radians(home_coords)

            # Calculate distance for each point in group
            for idx, row in group_df.iterrows():
                point_coords = np.radians([row[traj_cols['latitude']], row[traj_cols['longitude']]])
                stops.loc[idx, 'dist_from_home'] = utils._haversine_distance(point_coords, home_coords_rad)
        else:
            home_coords = home_stops[[coord_x, coord_y]].iloc[0].values
            dx = group_df[coord_x] - home_coords[0]
            dy = group_df[coord_y] - home_coords[1]
            distances = np.sqrt(dx*dx + dy*dy)
            stops.loc[group_df.index, 'dist_from_home'] = distances.values

    # 5) Calculate self-containment per group
    def _group_self_containment(g):
        """Calculate self-containment for a group (time period + user)."""
        # Filter for non-home activities
        non_home = g[g[activity_type_col] != home_activity_type]

        if len(non_home) == 0:
            return np.nan  # No non-home activities

        # Check if all distances are NaN (no home location found)
        if non_home['dist_from_home'].isna().all():
            return np.nan

        # Check which are within threshold
        within_threshold = non_home['dist_from_home'] <= threshold

        if weighted:
            # Calculate weights
            if time_weights is not None:
                weights = non_home[dur_key] * time_weights.loc[non_home.index]
            elif 'time_weights' in non_home.columns:
                weights = non_home[dur_key] * non_home['time_weights']
            else:
                weights = non_home[dur_key]

            # Time-weighted proportion
            total_weight = weights.sum()
            if total_weight == 0:
                return np.nan
            within_weight = (weights * within_threshold).sum()
            return within_weight / total_weight
        else:
            # Unweighted proportion (count of activities)
            return within_threshold.sum() / len(non_home)

    result = stops.groupby(keys).apply(_group_self_containment)
    return result.reset_index(name='self_containment')