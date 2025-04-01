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
import nomad.io.base as loader
import nomad.constants as constants
from nomad.stop_detection import utils

##########################################
########        Lachesis          ########
##########################################

def lachesis(traj, dur_min, dt_max, delta_roam, traj_cols=None, complete_output=False, **kwargs):
    """
    Detects stops in trajectory data by analyzing spatial and temporal patterns.

    Parameters
    ----------
    traj : pd.DataFrame
        Input trajectory data containing columns for spatial coordinates and timestamps.
    dur_min : float
        Minimum duration (in minutes) for a valid stop.
    dt_max : float
        Maximum allowed time difference (in minutes) between consecutive pings within a stop. dt_max should be greater than dur_min
    delta_roam : float
        Maximum roaming distance for a stop.
    traj_cols : dict, optional
        A dictionary defining column mappings for 'x', 'y', 'longitude', 'latitude', 'timestamp', or 'datetime'.
        Defaults to None.
    complete_output : bool, optional
        If True, returns a detailed output with additional stop metrics; otherwise, provides a concise output.
        Defaults to False.
    **kwargs :
        Additional parameters like 'latitude', 'longitude', or 'datetime' column names.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing detected stops with columns:
        - 'start_time', 'duration', 'x', 'y' (concise output).
        - Additional columns if `complete_output` is True: 'end_time', 'diameter', 'n_pings'.
    """
    # Check if user wants long and lat
    long_lat = 'latitude' in kwargs and 'longitude' in kwargs and kwargs[
        'latitude'] in traj.columns and kwargs['longitude'] in traj.columns

    # Check if user wants datetime
    datetime = 'datetime' in kwargs and kwargs['datetime'] in traj.columns

    # Set initial schema
    if not traj_cols:
        traj_cols = {}
    
    traj_cols = loader._update_schema(traj_cols, kwargs)
    traj_cols = loader._update_schema(constants.DEFAULT_SCHEMA, traj_cols)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(traj.columns, traj_cols)
    loader._has_time_cols(traj.columns, traj_cols)

    # Setting x and y as defaults if not specified by user in either traj_cols or kwargs
    if traj_cols['x'] in traj.columns and traj_cols['y'] in traj.columns and not long_lat:
        long_lat = False
        coords = traj[[traj_cols['x'], traj_cols['y']]].to_numpy()
    else:
        long_lat = True
        coords = traj[[traj_cols['longitude'], traj_cols['latitude']]].to_numpy()

    # Setting timestamp as default if not specified by user in either traj_cols or kwargs
    if traj_cols['timestamp'] in traj.columns and not datetime:
        datetime = False
        timestamp_col = traj_cols['timestamp']

        first_timestamp = traj[timestamp_col].iloc[0]
        timestamp_length = len(str(first_timestamp))

        if timestamp_length > 10:
            if timestamp_length == 13:
                warnings.warn(
                    f"The '{traj[timestamp_col]}' column appears to be in milliseconds. "
                    "This may lead to inconsistencies."
                )
                traj[timestamp_col] = traj[timestamp_col].values.view('int64') // 10 ** 3
            elif timestamp_length == 19:
                warnings.warn(
                    f"The '{timestamp_col_name}' column appears to be in nanoseconds. "
                    "This may lead to inconsistencies."
                )
                traj[timestamp_col] = traj[timestamp_col].values.view('int64') // 10 ** 9
    else:
        datetime = True
        datetime_col = traj_cols['datetime']
    
    stops = np.empty((0, 6))

    i = 0

    while i < len(traj) - 1:
        time_i = traj[timestamp_col].iat[i] if not datetime else traj[datetime_col].iat[i]

        if not datetime:
            j_star = next((j for j in range(i, len(traj)) if (
                        traj[timestamp_col].iat[j] - time_i) >= dur_min * 60), -1)
        else:
            j_star = next((j for j in range(i, len(traj)) if
                           (traj[datetime_col].iat[j] - time_i) >= timedelta(
                               minutes=dur_min)), -1)

        d_start = utils._diameter(coords[i:j_star + 1],
                           metric='haversine' if long_lat else 'euclidean')

        cond_exhausted_traj = j_star == -1
        cond_diam_OT = d_start > delta_roam

        if not datetime:
            cond_cc_diff_OT = (traj[timestamp_col].loc[i:j_star + 1]
                               .diff().dropna() >= dt_max * 60).any()
        else:
            cond_cc_diff_OT = (traj[datetime_col].loc[i:j_star + 1]
                               .diff().dropna().dt.total_seconds() >= dt_max * 60).any()

        if cond_exhausted_traj or cond_diam_OT or cond_cc_diff_OT:
            i += 1
        else:
            cond_found_jstar = False
            for j in range(j_star, len(traj) - 1):
                d_update = utils._update_diameter(coords[j], coords[i:j], d_start,
                                           metric='haversine' if long_lat else 'euclidean')

                if datetime:
                    cc_diff = (traj[datetime_col].iat[j] -
                               traj[datetime_col].iat[j - 1]).total_seconds()
                else:
                    cc_diff = traj[timestamp_col].iat[j] - \
                              traj[timestamp_col].iat[j - 1]

                cond_j = (d_update > delta_roam) or (cc_diff > dt_max * 60)

                if cond_j:
                    j_star = j - 1
                    cond_found_jstar = True
                    break
                else:
                    d_start = d_update

            if not cond_found_jstar:
                j_star = len(traj) - 1
            
            start, end = (traj[datetime_col].iat[i], traj[datetime_col].iat[j_star]) if datetime else (traj[timestamp_col].iat[i], traj[timestamp_col].iat[j_star])
            
            if datetime:
                duration = (traj[datetime_col].iat[j_star] - traj[datetime_col].iat[i]).total_seconds()
                cc_diffs = traj[datetime_col].loc[i:j_star].diff().dt.total_seconds()
            else:
                duration = traj[timestamp_col].iat[j_star] - traj[timestamp_col].iat[i]
                cc_diffs = traj[timestamp_col].loc[i:j_star].diff()
        
            if (duration >= dur_min * 60 and
                d_start <= delta_roam and
                (cc_diffs.dropna() <= dt_max * 60).all()):
                stop_medoid = utils._medoid(coords[i:j_star + 1])
                n_pings = j_star - i + 1
                stop = np.array([[start, end, stop_medoid[0], stop_medoid[1], d_start, n_pings]])
                stops = np.concatenate((stops, stop), axis=0)

            i = j_star + 1

    if long_lat:
        col_names = ['start_time', 'end_time', traj_cols['longitude'], traj_cols['latitude'], 'diameter', 'n_pings']
    else:
        col_names = ['start_time', 'end_time', traj_cols['x'], traj_cols['y'], 'diameter', 'n_pings']
    
    stops = pd.DataFrame(stops, columns=col_names)

    # compute stop duration
    if datetime:
        stops['duration'] = (stops['end_time'] - stops['start_time']).dt.total_seconds() / 60.0
    else:
        stops['duration'] = (stops['end_time'] - stops['start_time']) / 60.0

    if complete_output:
        return stops
    else:
        stops.drop(columns=['end_time', 'diameter', 'n_pings'], inplace=True)
        if long_lat:
            return stops[['start_time', 'duration', traj_cols['longitude'], traj_cols['latitude']]]
        else:
            return stops[['start_time', 'duration', traj_cols['x'], traj_cols['y']]]

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
    stops = lachesis(traj, dur_min, dt_max, delta_roam, traj_cols = traj_cols, complete_output = True, **kwargs)

    traj_cols = loader._update_schema(traj_cols, kwargs)
    traj_cols = loader._update_schema(constants.DEFAULT_SCHEMA, traj_cols)

    # Check if user wants long and lat
    long_lat = 'latitude' in kwargs and 'longitude' in kwargs and kwargs[
        'latitude'] in traj.columns and kwargs['longitude'] in traj.columns

    # Check if user wants datetime
    datetime = 'datetime' in kwargs and kwargs['datetime'] in traj.columns
    
    # Determine the timestamp column to use
    if traj_cols.get('timestamp') in traj.columns and not datetime:
        datetime = False
        timestamp_col = traj_cols['timestamp']
    elif traj_cols.get('datetime') in traj.columns and datetime:
        datetime = True
        datetime_col = traj_cols['datetime']

    # Initialize the stop_label column with default value -1
    output = traj.copy()
    output['cluster'] = -1

    # Iterate through detected stops and assign labels
    for stop_idx, stop in stops.iterrows():
        stop_start = stop['start_time']
        stop_end = stop['end_time']
        if datetime:
            output.loc[(output[datetime_col] >= stop_start) & (output[datetime_col] <= stop_end), 'cluster'] = stop_idx
        else:
            output.loc[(output[timestamp_col] >= stop_start) & (output[timestamp_col] <= stop_end), 'cluster'] = stop_idx

    # Keep only the time and cluster columns
    if datetime:
        output = output[['cluster']]
        output.index = list(traj[datetime_col])
    else:
        output = output[['cluster']]
        output.index = list(traj[timestamp_col])
    
    return output