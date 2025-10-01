import pandas as pd
from scipy.spatial.distance import pdist, cdist
import numpy as np
import datetime as dt
import itertools
import os
import nomad.io.base as loader
import nomad.constants as constants
import h3
#import dtoolkit.geoaccessor
import warnings
import pdb
from datetime import datetime, time, timedelta
from nomad.filters import to_timestamp



def _diameter(coords, metric='euclidean'):
    """
    Calculate the diameter of a set of coordinates, defined as the maximum pairwise distance.
    
    Parameters
    ----------
    coords : numpy.ndarray
        Array of coordinates, where each row represents a point in space.
    metric : str, optional
        Distance metric to use. Supported metrics include 'euclidean' (default) 
        and 'haversine'. If 'haversine' is used, coordinates should be in degrees.
    
    Returns
    -------
    float
        The diameter of the coordinate set, i.e., the maximum pairwise distance.
        Returns 0 if there are fewer than two coordinates.
    """
    if len(coords) < 2:
        return 0
   
    if metric == 'haversine':
        coords = np.radians(coords)
        pairwise_dists = pdist(coords,
                               metric=lambda u, v: _haversine_distance(u, v))
        return np.max(pairwise_dists)
    else:
        return np.max(pdist(coords, metric=metric))

def _medoid(coords, metric='euclidean'):
    """
    Calculate the medoid of a set of coordinates, defined as the point with the minimal 
    sum of distances to all other points.
    
    Parameters
    ----------
    coords : numpy.ndarray
        Array of coordinates, where each row represents a point in space.
    metric : str, optional
        Distance metric to use. Supported metrics include 'euclidean' (default) 
        and 'haversine'. If 'haversine' is used, coordinates should be in degrees.
    
    Returns
    -------
    numpy.ndarray
        The medoid of the coordinate set, represented as a single point. If there
        is only one point, returns that point.
    """
    if len(coords) < 2:
        return coords[0]
    
    if metric == 'haversine':
        distances = _pairwise_haversine(np.radians(coords))
    else:
        distances = cdist(coords, coords, metric=metric)

    sum_distances = np.sum(distances, axis=1)
    medoid_index = np.argmin(sum_distances)
    return coords[medoid_index, :]

def _haversine_distance(coord1, coord2):
    """
    Compute the haversine distance between two points on Earth.

    Parameters:
        coord1: [lat1, lon1] in radians
        coord2: [lat2, lon2] in radians

    Returns:
        Distance in meters.
    """
    earth_radius_meters = 6371000  # Earth's radius in meters
    delta_lat = coord2[0] - coord1[0]
    delta_lon = coord2[1] - coord1[1]
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(coord1[0]) * np.cos(
        coord2[0]) * np.sin(delta_lon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius_meters * c  # Distance in meters


def _pairwise_haversine(coords):
    """
    Compute the pairwise Haversine distances between a set of coordinates.
    
    Parameters
    ----------
    coords : numpy.ndarray
        Array of coordinates, where each row represents a point in [latitude, longitude] 
        in radians.
    
    Returns
    -------
    numpy.ndarray
        A symmetric 2D array where the element at [i, j] represents the Haversine 
        distance between the i-th and j-th points.
    """
    n = len(coords)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = _haversine_distance(coords[i], coords[j])
            distances[j, i] = distances[i, j]
    return distances

    
def _update_diameter(c_j, coords_prev, D_prev, metric='euclidean'):
    """
    Update the diameter of a set of coordinates when a new point is added.
    
    Parameters
    ----------
    c_j : numpy.ndarray
        The new point being added to the coordinate set.
    coords_prev : numpy.ndarray
        Array of existing coordinates, where each row represents a point.
    D_prev : float
        The previous diameter of the coordinate set, before adding the new point.
    metric : str, optional
        Distance metric to use. Supported metrics are 'euclidean' (default) and 'haversine'.
        If 'haversine' is used, coordinates should be in degrees.
    
    Returns
    -------
    float
        The updated diameter of the coordinate set, considering the new point.
    """
    if metric == 'euclidean':
        X_prev = coords_prev[:, 0]
        Y_prev = coords_prev[:, 1]
        x_j, y_j = c_j[0], c_j[1]
        new_dists = np.sqrt((X_prev - x_j) ** 2 + (Y_prev - y_j) ** 2)

    elif metric == 'haversine':
        coords_prev = np.radians(coords_prev)
        c_j = np.radians(c_j)

        lat_j, lon_j = c_j[0], c_j[1]
        new_dists = np.array([
            _haversine_distance([lat_j, lon_j], [lat_i, lon_i])
            for lat_i, lon_i in coords_prev
        ])

    else:
        raise ValueError("metric must be 'euclidean' or 'haversine'")

    D_i_jp1 = np.max([D_prev, np.max(new_dists)])

    return D_i_jp1

def _fallback_time_cols(col_names, traj_cols, kwargs):
    '''
    Helper to decide whether to use datetime vs timestamp in cases of ambiguity
    '''
    traj_cols = loader._parse_traj_cols(col_names, traj_cols, kwargs, defaults={}, warn=False)
    # check for explicit datetime usage
    t_keys = ['timestamp', 'start_timestamp', 'datetime', 'start_datetime']
    if 'datetime' in kwargs or 'start_datetime' in kwargs: # prioritize datetime 
        t_keys = t_keys[-2:] + t_keys[:2]

    # load defaults and check for time columns
    traj_cols = loader._update_schema(constants.DEFAULT_SCHEMA, traj_cols)
    loader._has_time_cols(col_names, traj_cols) # error if no columns
    
    for t_key in t_keys:
        if traj_cols[t_key] in col_names:
            use_datetime = (t_key in ['datetime', 'start_datetime']) ## necessary?
            break
            
    return t_key, use_datetime

# should this be moved to io.utils?
def _fallback_st_cols(col_names, traj_cols, kwargs):
    '''
    Helper function to decide whether to use latitude and longitude or x,y,
    as well as datetime vs timestamp in cases of ambiguity
    '''
    traj_cols = loader._parse_traj_cols(col_names, traj_cols, kwargs, defaults={}, warn=False)
    
    # check for sufficient spatial coords
    loader._has_spatial_cols(col_names, traj_cols, exclusive=True) 

    use_lon_lat = ('latitude' in traj_cols and 'longitude' in traj_cols)
    if use_lon_lat:
        coord_key1, coord_key2 = 'longitude', 'latitude'
    else:
        coord_key1, coord_key2 = 'x', 'y'

    # check for explicit datetime usage
    t_keys = ['timestamp', 'start_timestamp', 'datetime', 'start_datetime']
    if 'datetime' in kwargs or 'start_datetime' in kwargs: # prioritize datetime 
        t_keys = t_keys[-2:] + t_keys[:2]

    # load defaults and check for time columns
    traj_cols = loader._update_schema(constants.DEFAULT_SCHEMA, traj_cols)
    loader._has_time_cols(col_names, traj_cols)

    for t_key in t_keys:
        if traj_cols[t_key] in col_names:
            use_datetime = (t_key in ['datetime', 'start_datetime']) ## necessary?
            break
            
    return t_key, coord_key1, coord_key2, use_datetime, use_lon_lat

def summarize_stop(
    grouped_data,
    method='medoid',
    complete_output=False,
    keep_col_names=True,
    passthrough_cols=None,
    traj_cols=None,
    **kwargs
):
    """
    Summarize a single coordinate-based stop cluster.

    Parameters
    ----------
    grouped_data : pd.DataFrame
        One cluster’s worth of pings, sorted by time.
    method : str
        Medoid method (currently only 'medoid' supported).
    complete_output : bool
        If True, include diameter, n_pings, end_time, and max_gap.
    keep_col_names : bool
        If False, rename output coord columns to DEFAULT_SCHEMA (['x','y']).
        If True, keep user’s original names (e.g. 'lon','lat').
    passthrough_cols : list[str], optional
        Any extra columns (e.g. 'user_id') to copy through.
    traj_cols : dict, optional
        Overrides for logical column mapping (x, y, timestamp, etc).
    **kwargs
        Additional mapping args (e.g. datetime='datetime_col').

    Returns
    -------
    pd.Series
        A one‐row summary with:
          - start_[timestamp|datetime]
          - [x|lon], [y|lat]
          - duration (always)
          - end_[timestamp|datetime], diameter, n_pings, max_gap (if complete_output)
          - any passthrough_cols
    """
    if passthrough_cols is None:
        passthrough_cols = []

    # 1) pick time & spatial keys (may raise if missing)
    t_key, coord_x, coord_y, use_datetime, use_lon_lat = _fallback_st_cols(
        grouped_data.columns, traj_cols, kwargs
    )

    # 2) parse full mapping
    traj_cols = loader._parse_traj_cols(grouped_data.columns, traj_cols, kwargs)

    # 3) decide output column names
    start_key = 'start_datetime' if use_datetime else 'start_timestamp'
    end_key   = 'end_datetime'   if use_datetime else 'end_timestamp'

    # 4) medoid + time bounds

    coords   = grouped_data[[traj_cols[coord_x], traj_cols[coord_y]]].to_numpy()
    medoid   = _medoid(coords, metric='haversine' if use_lon_lat else 'euclidean')
    start_ts = grouped_data[traj_cols[t_key]].iloc[0]
    end_ts   = grouped_data[traj_cols[t_key]].iloc[-1]

    if not keep_col_names:
        traj_cols[coord_x] = constants.DEFAULT_SCHEMA[coord_x]
        traj_cols[coord_y] = constants.DEFAULT_SCHEMA[coord_y]
        # leave start_key as constants.DEFAULT_SCHEMA[start_key]
    else:
        # write over the default start_key so we output the user’s original t_key name
        traj_cols[start_key] = traj_cols[t_key]
        
    out = {
        traj_cols[coord_x]:     medoid[0],
        traj_cols[coord_y]:     medoid[1],
        traj_cols[start_key]:   start_ts,
        'duration':        (end_ts - start_ts).total_seconds()//60 if use_datetime else (end_ts - start_ts)//60
    }

    # 5) complete stats
    if complete_output:
        out[end_key]       = end_ts
        out['diameter']    = _diameter(coords, metric='haversine' if use_lon_lat else 'euclidean')
        out['n_pings']     = len(grouped_data)
        diffs = grouped_data[traj_cols[t_key]].diff().dropna()
        out['max_gap'] = (
            int(diffs.max().total_seconds() // 60)
            if use_datetime else
            int(diffs.max() // 60) if len(diffs) else 0
        )

    # 6) passthrough
    for c in passthrough_cols:
        if c in grouped_data.columns:
            out[c] = grouped_data[c].iloc[0]

    return pd.Series(out, dtype='object')

def summarize_stop_grid(
    grouped_data,
    complete_output=False,
    keep_col_names=True,
    passthrough_cols=None,
    traj_cols=None,
    **kwargs
):
    """
    Summarize index/grid‐based stop clusters (location_id only).

    Parameters
    ----------
    grouped_data : pd.DataFrame or gpd.GeoDataFrame
        All pings sharing the same location_id.
    complete_output : bool
        If True, include n_pings, duration, and max_gap.
    keep_col_names : bool
        If False, output start_/end_ keys are DEFAULT_SCHEMA ones;
        if True, they use the user’s time‐column name.
    passthrough_cols : list[str], optional
        Additional columns (e.g. 'user_id') to carry through.
    traj_cols : dict, optional
        Column‐name overrides.

    Returns
    -------
    pd.Series
        One‐row summary with:
          - start_[timestamp|datetime],
          - end_[timestamp|datetime] (if complete_output),
          - duration, n_pings, max_gap (if complete_output),
          - location_id, geometry (if present), plus passthrough_cols.
    """
    if passthrough_cols is None:
        passthrough_cols = []

    # 1) pick time key
    t_key, use_datetime = _fallback_time_cols(grouped_data.columns, traj_cols, kwargs)

    # 2) parse full mapping
    cols = loader._parse_traj_cols(grouped_data.columns, traj_cols, kwargs)

    # 3) decide output key names
    start_key = 'start_datetime' if use_datetime else 'start_timestamp'
    end_key   = 'end_datetime'   if use_datetime else 'end_timestamp'

    if keep_col_names:
        cols[start_key] = cols[t_key]
        cols[end_key]   = cols.get(end_key, end_key)
    else:
        cols[start_key] = start_key
        cols[end_key]   = end_key

    # 4) time bounds
    times      = grouped_data[cols[t_key]]
    start_ts   = times.iloc[0]
    end_ts     = times.iloc[-1]

    out = {
        cols[start_key]: start_ts,
    }

    if complete_output:
        out[cols[end_key]] = end_ts
        out['n_pings']    = len(grouped_data)
        diffs = times.diff().dropna()
        max_gap = (
            int(diffs.max().total_seconds()//60)
            if use_datetime else
            int(diffs.max()//60) if len(diffs) else 0
        )
        out['max_gap']    = max_gap
        out['duration'] = (
            (end_ts - start_ts).total_seconds()//60
            if use_datetime else
            (end_ts - start_ts)//60
        )


    # 5) always pass through location_id & geometry & any others
    to_pass = set(passthrough_cols)
    loc_col = cols['location_id']
    to_pass.add(loc_col)
    if 'geometry' in grouped_data.columns:
        to_pass.add('geometry')

    for c in to_pass:
        if c in grouped_data.columns:
            out[c] = grouped_data[c].iloc[0]

    return pd.Series(out, dtype='object')

def pad_short_stops(stop_data, pad=5, dur_min=None, traj_cols = None, **kwargs):
    """
    Helper that pads stops shorter or equal than dur_min minutes 
    extending the duration by pad minutes, but avoiding overlap. 
    stop_data must be sorted chronologically and not overlap.
    """
    stop_data = stop_data.copy()
    if not dur_min:
        dur_min = pad
    pad = max(pad, dur_min, 1) # we never shorten a stop
    
    t_key, use_datetime = _fallback_time_cols(stop_data.columns, traj_cols, kwargs)
    end_t_key = 'end_datetime' if use_datetime else 'end_timestamp'

    # Load default col names
    traj_cols = loader._parse_traj_cols(stop_data.columns, traj_cols, kwargs)    
    # check is diary table
    end_col_present = loader._has_end_cols(stop_data.columns, traj_cols)
    duration_col_present = loader._has_duration_cols(stop_data.columns, traj_cols)
    if not (end_col_present or duration_col_present):
        raise ValueError("Missing required (end or duration) temporal columns for stop_table dataframe.")

    # to compute time differences
    ts = to_timestamp(stop_data[traj_cols[t_key]]) if use_datetime else stop_data[traj_cols[t_key]]
    next_ts = ts.shift(-1, fill_value=ts.iloc[-1]+3*pad*60)

    if not duration_col_present:
        if use_datetime:
            stop_data[traj_cols['duration']] = int((stop_data[traj_cols[end_t_key]] - stop_data[traj_cols[t_key]]).total_seconds())//60
        else:
            stop_data[traj_cols['duration']] = (stop_data[traj_cols[end_t_key]] - stop_data[traj_cols[t_key]])//60
    
    new_duration = np.minimum((next_ts - ts)//(2*60), pad)
    mask = (stop_data[traj_cols['duration']] <= dur_min)
    stop_data[traj_cols['duration']] = np.maximum(new_duration*mask, stop_data[traj_cols['duration']])

    # fix end column
    if end_col_present:
        if use_datetime:
            stop_data[traj_cols[end_t_key]] = np.maximum(stop_data[traj_cols[end_t_key]],
                                                         stop_data[traj_cols[t_key]] + pd.to_timedelta(stop_data[traj_cols['duration']]*60, unit='s'))
        else:
            stop_data[traj_cols[end_t_key]] = np.maximum(stop_data[traj_cols[end_t_key]],
                                                         stop_data[traj_cols[t_key]] + stop_data[traj_cols['duration']]*60)   
    return stop_data

def explode_stops(stops, agg_freq="d", start_col="start_datetime", end_col="end_datetime", use_datetime=True):
    """
    Explode each stop into one row per time bucket (day or week) based on agg_freq.

    Parameters
    ----------
    stops : pd.DataFrame
    agg_freq : str
        "d" for daily, "w" for weekly.
    start_col, end_col : str
        Column names for start and end times.
    use_datetime : bool
        If False, converts start/end columns from unix seconds to datetime.

    Returns
    -------
    pd.DataFrame
        Exploded table with updated start/end/duration per bucket.
    """
    stops = stops.copy()
    stops.drop(columns=["n_pings", "diameter", "max_gap", "identifier"], errors="ignore", inplace=True)

    # Convert to datetime if needed
    if not use_datetime:
        stops[start_col] = pd.to_datetime(stops[start_col], unit='s')
        stops[end_col] = pd.to_datetime(stops[end_col], unit='s')

    if agg_freq.lower() == "d":
        # Daily buckets
        stops["_bucket_start"] = stops.apply(
            lambda r: [
                pd.Timestamp(datetime.combine(d, time(0)), tz=r[start_col].tzinfo)
                for d in pd.date_range(r[start_col].date(), r[end_col].date(), freq="D")
                if pd.Timestamp(datetime.combine(d, time(0)), tz=r[start_col].tzinfo) < r[end_col]
            ],
            axis=1,
        )
        stops = stops.explode("_bucket_start", ignore_index=True)
        stops["_bucket_end"] = stops["_bucket_start"] + timedelta(days=1)
    elif agg_freq.lower() == "w":
        # Weekly buckets (start on Monday)
        stops["_bucket_start"] = stops.apply(
            lambda r: [
                pd.Timestamp(d, tz=r[start_col].tzinfo)
                for d in pd.date_range(
                    r[start_col].date() - timedelta(days=r[start_col].weekday()),
                    r[end_col].date(),
                    freq="W-MON"
                )
                if pd.Timestamp(d, tz=r[start_col].tzinfo) < r[end_col]
            ],
            axis=1,
        )
        stops = stops.explode("_bucket_start", ignore_index=True)
        stops["_bucket_end"] = stops["_bucket_start"] + timedelta(days=7)
    else:
        raise ValueError("agg_freq must be 'd' (day) or 'w' (week)")

    # Clip the stop to the bucket interval
    stops[start_col] = stops[[start_col, "_bucket_start"]].max(axis=1)
    stops[end_col] = stops[[end_col, "_bucket_end"]].min(axis=1)
    stops["duration"] = (
        (stops[end_col] - stops[start_col]).dt.total_seconds() // 60
    ).astype(int)

    return stops[stops["duration"] > 0].drop(columns=["_bucket_start", "_bucket_end"])


def clip_stops_datetime(stops_table, start_datetime, end_datetime, start_col="start_datetime", end_col="end_datetime", duration_col="duration"):
    """
    Clip each stop to a specific datetime range.
    
    This function takes a stop table and clips each stop to the specified datetime range,
    recomputing durations and dropping stops that don't intersect the range.
    
    Parameters
    ----------
    stops_table : pandas.DataFrame
        Stop table with at least start_datetime, end_datetime, and duration columns
    start_datetime : str or pandas.Timestamp
        Start of the datetime range to clip to
    end_datetime : str or pandas.Timestamp  
        End of the datetime range to clip to
    start_col : str, default "start_datetime"
        Column name for start times
    end_col : str, default "end_datetime"
        Column name for end times
    duration_col : str, default "duration"
        Column name for duration in minutes
        
    Returns
    -------
    pandas.DataFrame
        Clipped stop table with updated start/end times and durations.
        Only includes stops that intersect the specified datetime range.
    """
    df = stops_table.copy()
    
    # Convert start and end times to datetime
    start = pd.to_datetime(df[start_col])
    duration = df[duration_col]
    
    # Ensure timezone-aware clipping bounds
    tz = start.dt.tz
    if tz is not None:
        start_dt = pd.Timestamp(start_datetime, tz=tz)
        end_dt = pd.Timestamp(end_datetime, tz=tz)
    else:
        start_dt = pd.Timestamp(start_datetime)
        end_dt = pd.Timestamp(end_datetime)
    
    # Calculate end times
    end = start + pd.to_timedelta(duration, unit='m')
    
    # Clip to datetime range
    start_clipped = start.clip(lower=start_dt, upper=end_dt)
    end_clipped = end.clip(lower=start_dt, upper=end_dt)
    
    # Recompute durations
    duration_clipped = ((end_clipped - start_clipped).dt.total_seconds() // 60).astype(int)
    
    # Update the dataframe
    df[start_col] = start_clipped
    df[end_col] = end_clipped
    df[duration_col] = duration_clipped
    
    # Return only stops that have positive duration after clipping
    return df[df[duration_col] > 0]