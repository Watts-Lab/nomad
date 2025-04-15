import pandas as pd
from scipy.spatial.distance import pdist, cdist
import numpy as np
import math
import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import sys
import os
import pdb
import geopandas as gpd
import nomad.io.base as loader
import nomad.constants as constants
from nomad.stop_detection import utils
from nomad.filters import to_timestamp

##########################################
########        Lachesis          ########
##########################################

def lachesis(traj, dur_min, dt_max, delta_roam, traj_cols=None, complete_output=False, keep_col_names = True, **kwargs):
    """
    Detects stops in trajectory data by analyzing spatial and temporal patterns.

    Parameters
    ----------
    traj : pd.DataFrame
        Input trajectory data containing columns for spatial coordinates and timestamps.
    dur_min : int
        Minimum duration (in minutes) for a valid stop.
    dt_max : int
        Maximum allowed time difference (in minutes) between consecutive pings within a stop. dt_max should be greater than dur_min
    delta_roam : float
        Maximum roaming distance for a stop.
    traj_cols : dict, optional
        A dictionary defining column mappings for 'x', 'y', 'longitude', 'latitude', 'timestamp', or 'datetime'.
        Defaults to None.
    complete_output : bool, optional
        If True, returns a detailed output with additional stop metrics; otherwise, provides a concise output.
        Defaults to False.
    keep_col_names : bool, optional
        If True, the output keeps the same names for coordinate and temporal columns, otherwise the output table
        uses default column names from constants.DEFAULT_SCHEMA
        Defaults to True.
    **kwargs :
        Additional parameters like 'latitude', 'longitude', or 'datetime' column names.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing detected stops with columns:
        - 'start_time', 'duration', 'x', 'y' (concise output).
        - Additional columns if `complete_output` is True: 'end_time', 'diameter', 'n_pings'.
    """
    if not isinstance(traj, (pd.DataFrame, gpd.GeoDataFrame)):
         raise TypeError("Input 'traj' must be a pandas DataFrame or GeoDataFrame.")
    if traj.empty:
        return pd.DataFrame()

    # To DO: implement safe handling of multiple
    # user data (should raise or cluster separately with groupby)
        
    traj_cols = loader._parse_traj_cols(traj.columns, traj_cols, kwargs)
    loader._has_spatial_cols(traj.columns, traj_cols)
    loader._has_time_cols(traj.columns, traj_cols)

    # Determine projection and time format preferences
    use_latlon = False
    if (traj_cols['x'] in traj.columns and traj_cols['y'] in traj.columns):
        metric = 'euclidean'
        spatial_cols_in = [traj_cols['x'], traj_cols['y']]
        coord_key1, coord_key2 = 'x', 'y'
    else:
        use_latlon = True

    # Check explicit user kwargs preference FIRST
    if 'latitude' in kwargs and 'longitude' in kwargs:
         # Explicit lat/lon preference, *if available*
         if traj_cols['latitude'] in traj.columns and traj_cols['longitude'] in traj.columns:
             use_latlon = True

    # Set final parameters based on use_latlon flag
    if use_latlon:
        metric = 'haversine'
        spatial_cols_in = [traj_cols['longitude'], traj_cols['latitude']] # Lon, Lat for util
        coord_key1, coord_key2 = 'longitude', 'latitude'

    coords = traj[spatial_cols_in].to_numpy(dtype='float64')

    use_datetime = False
    if traj_cols['timestamp'] in traj.columns:
        time_col_in = traj_cols['timestamp']
        time_key = 'timestamp'
    elif traj_cols['start_timestamp'] in traj.columns:
        time_col_in = traj_cols['start_timestamp']
        time_key = 'start_timestamp'
    else:
        use_datetime = True
        
    #check user preferences
    if 'datetime' in kwargs or 'start_datetime' in kwargs:
        # Explicit datetime preference if available
        if traj_cols['datetime'] or traj_cols['start_datetime'] in traj.columns:
            use_datetime = True

    # set final parameters based on use_datetime flag
    if use_datetime and traj_cols['datetime'] in traj.columns:
        time_col_in = traj_cols['datetime']
        time_key = 'datetime'
    elif use_datetime and traj_cols['start_datetime'] in traj.columns:
        time_col_in = traj_cols['start_datetime']
        time_key = 'start_datetime'
    
    # Parse if necessary
    if use_datetime:
        # Use to_timestamp to convert datetime series to UNIX timestamps (seconds)
        time_series = to_timestamp(traj[time_col_in])
    else:
        first_val = traj[time_col_in].iloc[0]
        s_val = str(first_val)
        if len(s_val) > 10:
            if len(s_val) == 13:
                warnings.warn(f"The '{time_col_in}' column appears to be in milliseconds. Converting to seconds.")
                time_series = traj[time_col_in].astype('int64') // 10**3
            elif len(s_val) == 19:
                warnings.warn(f"The '{time_col_in}' column appears to be in nanoseconds. Converting to seconds.")
                time_series = traj[time_col_in].astype('int64') // 10**9
            else:
                time_series = traj[time_col_in]
        else:
            time_series = traj[time_col_in]
  
    stops = []
    i = 0
    n = len(traj)
    while i < n - 1:
        t_i = time_series.iloc[i]
        j_star = next((j for j in range(i, n) if (time_series.iloc[j] - t_i) >= dur_min * 60), -1)
        if j_star == -1:
            break

        d_start = utils._diameter(coords[i:j_star + 1], metric=metric)
        time_diffs = np.diff(time_series.iloc[i:j_star + 1].values)
        if (time_diffs >= dt_max * 60).any() or d_start > delta_roam:
            i += 1
            continue

        j_final = j_star
        for j in range(j_star, n):
            d_update = utils._update_diameter(coords[j], coords[i:j], d_start, metric=metric)
            cc_diff = time_series.iloc[j] - time_series.iloc[j - 1]
            if d_update > delta_roam or cc_diff > dt_max * 60:
                j_final = j - 1
                break
            d_start = d_update
        else:
            j_final = n - 1

        duration = (time_series.iloc[j_final] - time_series.iloc[i]) // 60
        if duration >= dur_min:
            if use_datetime:
                start_val = traj[time_col_in].iloc[i]
                end_val = traj[time_col_in].iloc[j_final]
            else:
                start_val = time_series.iloc[i]
                end_val = time_series.iloc[j_final]

            medoid = utils._medoid(coords[i:j_final + 1])
            if complete_output:
                row = [
                    start_val,
                    end_val,
                    duration,
                    medoid[0],
                    medoid[1],
                    d_start,
                    j_final - i + 1
                ]
            else:
                row = [
                    start_val,
                    duration,
                    medoid[0],
                    medoid[1]
                ]
            stops.append(row)

        i = j_final + 1

    if keep_col_names:
        coord_cols = [traj_cols[coord_key1], traj_cols[coord_key2]]
        start_col = time_col_in
    else:
        coord_cols = [constants.DEFAULT_SCHEMA[coord_key1], constants.DEFAULT_SCHEMA[coord_key2]]
        start_col = constants.DEFAULT_SCHEMA['start_datetime'] if use_datetime else constants.DEFAULT_SCHEMA['start_timestamp']

    end_col = constants.DEFAULT_SCHEMA['end_datetime'] if use_datetime else constants.DEFAULT_SCHEMA['end_timestamp']

    if complete_output:
        columns = [start_col, end_col, 'duration'] + coord_cols + ['diameter', 'n_pings']
    else:
        columns = [start_col, 'duration'] + coord_cols

    return pd.DataFrame(stops, columns=columns)

def _lachesis_labels(traj, dur_min, dt_max, delta_roam, traj_cols=None, **kwargs):
    """
    Assigns a label to every point in the trajectory based on the stop it belongs to.

    Parameters
    ----------
    traj : pd.DataFrame
        Original trajectory data containing spatial and temporal columns.
    stops : pd.DataFrame
        Detected stops from the `lachesis` function. Should include 'start_time' and 'end_time'.
    traj_cols : dict, optional
        A dictionary defining column mappings for 'timestamp' or 'datetime'.
        Defaults to None.
    **kwargs :
        Additional parameters like 'datetime' column names.

    Returns
    -------
    pd.DataFrame
        A copy of `traj` with an additional 'stop_label' column, where:
        - Labels are integers corresponding to stop indices.
        - Points not in any stop are assigned a label of -1.
    """
    stops = lachesis(traj,
                     dur_min,
                     dt_max,
                     delta_roam,
                     traj_cols = traj_cols,
                     complete_output = True,
                     keep_col_names = False,
                     **kwargs)
    
    traj_cols = loader._parse_traj_cols(traj.columns, traj_cols, kwargs)
    loader._has_spatial_cols(traj.columns, traj_cols)
    loader._has_time_cols(traj.columns, traj_cols)

    use_datetime = False
    if traj_cols['timestamp'] in traj.columns:
        time_col_in = traj_cols['timestamp']
    elif traj_cols['start_timestamp'] in traj.columns:
        time_col_in = traj_cols['start_timestamp']
    else:
        use_datetime = True
        
    #check user preferences
    if 'datetime' in kwargs or 'start_datetime' in kwargs:
        # Explicit datetime preference if available
        if traj_cols['datetime'] or traj_cols['start_datetime'] in traj.columns:
            use_datetime = True

    # set final parameters based on use_datetime flag
    if use_datetime and traj_cols['datetime'] in traj.columns:
        time_col_in = traj_cols['datetime']
    elif use_datetime and traj_cols['start_datetime'] in traj.columns:
        time_col_in = traj_cols['start_datetime']
    
    start_col = constants.DEFAULT_SCHEMA['start_datetime'] if use_datetime else constants.DEFAULT_SCHEMA['start_timestamp']
    end_col = constants.DEFAULT_SCHEMA['end_datetime'] if use_datetime else constants.DEFAULT_SCHEMA['end_timestamp']

    # Initialize the stop_label column with default value -1
    stop_labels = pd.Series(-1, index=traj[time_col_in])
    
    # Iterate through detected stops and assign labels
    for stop_idx, stop in stops.iterrows():
        stop_start = stop[start_col]
        stop_end = stop[end_col]
        mask = (traj[time_col_in] >= stop_start) & (traj[time_col_in] <= stop_end)
        stop_labels.loc[traj[time_col_in][mask]] = stop_idx
    
    return stop_labels