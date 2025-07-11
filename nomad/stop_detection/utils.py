import pandas as pd
from scipy.spatial.distance import pdist, cdist
import numpy as np
import datetime as dt
from datetime import timedelta
import itertools
import os
import nomad.io.base as loader
import nomad.constants as constants
from nomad.filters import to_timestamp
import pdb

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

def summarize_stop(grouped_data, method='medoid', complete_output = False, keep_col_names = True, passthrough_cols= [], traj_cols=None, **kwargs):
    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = _fallback_st_cols(grouped_data.columns, traj_cols, kwargs)
    
    # Load default col names
    traj_cols = loader._parse_traj_cols(grouped_data.columns, traj_cols, kwargs)

    metric = 'haversine' if use_lon_lat else 'euclidean'    
    start_t_key = 'start_datetime' if use_datetime else 'start_timestamp'
    end_t_key = 'end_datetime' if use_datetime else 'end_timestamp'
    
    if not keep_col_names:
       traj_cols[coord_key1] = constants.DEFAULT_SCHEMA[coord_key1]
       traj_cols[coord_key2] = constants.DEFAULT_SCHEMA[coord_key2]
       # traj_cols[start_t_key] holds default or user provided value
    else:
        # use same time column key for start time
        # e.g. start time col in stops will be 'unix_timestamp' instead of default 'start_timestamp'
        traj_cols[start_t_key] = traj_cols[t_key]
    
    # Compute stop statistics
    coords = grouped_data[[traj_cols[coord_key1], traj_cols[coord_key2]]].to_numpy()
    medoid = _medoid(coords, metric=metric)
    end_time = grouped_data[traj_cols[t_key]].iloc[-1]

    stop_attr = {} # the pandas series for the output
    stop_attr[coord_key1] = medoid[0]
    stop_attr[coord_key2] = medoid[1]
    stop_attr[traj_cols[start_t_key]]  = grouped_data[traj_cols[t_key]].iloc[0]

    if complete_output:
        if traj_cols['ha'] in grouped_data.columns:
            stop_attr[traj_cols['ha']] = grouped_data[traj_cols['ha']].mean()
        stop_attr['diameter'] = _diameter(coords, metric=metric)
        stop_attr['n_pings'] = len(grouped_data)
        stop_attr[traj_cols[end_t_key]] = end_time
        
        time_diffs = grouped_data[traj_cols[t_key]].diff().dropna()
        max_gap = time_diffs.max() if len(time_diffs) > 0 else 0

    if use_datetime:
        stop_attr['duration'] = int((end_time - stop_attr[traj_cols[start_t_key]]).total_seconds())//60
        if complete_output:
            stop_attr['max_gap'] = int(max_gap.total_seconds())//60
    else:
        stop_attr['duration'] = (end_time - stop_attr[traj_cols[start_t_key]])//60
        if complete_output:
            stop_attr['max_gap'] = max_gap//60

    # passthrough columns: e.g. location_id
    for col in passthrough_cols:
        if col in grouped_data.columns:
            stop_attr[col] = grouped_data[col].iloc[0]

    return pd.Series(stop_attr)

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
