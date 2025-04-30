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
########         DBSCAN           ########
##########################################
def _find_neighbors(data, time_thresh, dist_thresh, long_lat, datetime, traj_cols):
    """
    Compute neighbors within specified time and distance thresholds for a trajectory dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Trajectory data containing spatial and temporal information.
    traj_cols : dict
        Dictionary mapping column names for trajectory attributes.
    time_thresh : int
        Time threshold in minutes for considering neighboring points.
    dist_thresh : float
        Distance threshold for considering neighboring points.
    long_lat : bool, optional
        Whether to use longitude/latitude coordinates.
    datetime : bool, optional
        Whether to process timestamps as datetime objects.
    
    Returns
    -------
    dict
        A dictionary where keys are timestamps, and values are sets of neighboring
        timestamps that satisfy both time and distance thresholds.
    """
    # getting coordinates based on whether they are geographic coordinates (lon, lat) or catesian (x,y)
    if long_lat:
        coords = np.radians(data[[traj_cols['latitude'], traj_cols['longitude']]].values)
    else:
        coords = data[[traj_cols['x'], traj_cols['y']]].values
    
    # getting times based on whether they are datetime values or timestamps, changed to seconds for calculations
    if datetime:
        times = pd.to_datetime(data[traj_cols['datetime']])
        times = times.dt.tz_convert('UTC').dt.tz_localize(None)
        times = times.astype('int64') // 10**9
        times = times.values
        
    else:
        # if timestamps, we change the values to seconds
        first_timestamp = data[traj_cols['timestamp']].iloc[0]
        timestamp_length = len(str(first_timestamp))
    
        if timestamp_length > 10:
            if timestamp_length == 13:
                times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 3
            elif timestamp_length == 19:
                times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 9   
        else:
            times = data[traj_cols['timestamp']].values
      
    # Pairwise time differences
    time_diffs = np.abs(times[:, np.newaxis] - times)
    time_diffs = time_diffs.astype(int)
  
    # Filter by time threshold
    within_time_thresh = np.triu(time_diffs <= (time_thresh * 60), k=1)
    time_pairs = np.where(within_time_thresh)
  
    # Distance calculation
    if long_lat:
        distances = np.array([utils._haversine_distance(coords[i], coords[j]) for i, j in zip(*time_pairs)])
    else:
        distances_sq = (coords[time_pairs[0], 0] - coords[time_pairs[1], 0])**2 + (coords[time_pairs[0], 1] - coords[time_pairs[1], 1])**2
        distances = np.sqrt(distances_sq)

    # Filter by distance threshold
    neighbor_pairs = distances < dist_thresh
  
    # Building the neighbor dictionary
    neighbor_dict = defaultdict(set)
  
    for i, j in zip(time_pairs[0][neighbor_pairs], time_pairs[1][neighbor_pairs]):
        neighbor_dict[times[i]].add(times[j])
        neighbor_dict[times[j]].add(times[i])

    return neighbor_dict

def _extract_middle(data):
    """
    Extract the middle segment of a cluster within the provided data.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data containing a 'cluster' column.
    
    Returns
    -------
    tuple (i, j)
        Indices marking the start (`i`) and end (`j`) of the middle segment
        of the cluster. If the cluster does not reappear, returns indices marking
        the tail of the data.
    """
    current = data.iloc[0]['cluster']
    x = (data.cluster != current).values
    if len(np.where(x)[0]) == 0:  # There is no inbetween
        return (len(data), len(data))
    else:
        i = np.where(x)[0][
            0]  # First index where the cluster is not the value of the first entry's cluster
    if len(np.where(~x[i:])[
               0]) == 0:  # There is no current again (i.e., the first cluster does not reappear, so the middle is actually the tail)
        return (i, len(data))
    else:  # Current reappears
        j = i + np.where(~x[i:])[0][0]
    return (i, j)

def dbscan(data, time_thresh, dist_thresh, min_pts, long_lat, datetime, traj_cols, neighbor_dict=None):
    """
    Perform DBSCAN on a trajectory dataset with spatiotemporal constraints.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Trajectory data containing spatial and temporal information.
    time_thresh : int
        Time threshold in minutes for identifying neighbors.
    dist_thresh : float
        Distance threshold for identifying neighbors.
    min_pts : int
        Minimum number of points required to form a dense region (core point).
    long_lat : bool
        Whether to use longitude/latitude coordinates.
    datetime : bool
        Whether to process timestamps as datetime objects.
    traj_cols : dict
        Dictionary mapping column names for trajectory attributes.
    neighbor_dict : dict, optional
        Precomputed dictionary of neighbors. If not provided, it will be computed.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns:
        - 'cluster': Cluster labels assigned to each point. Noise points are labeled as -1.
        - 'core': Core point labels for each point. Non-core points are labeled as -3.
    """
    # getting the values for time, both the original and changed to seconds for calculations
    if datetime:
        valid_times = pd.to_datetime(data[traj_cols['datetime']])
        valid_times = valid_times.dt.tz_convert('UTC').dt.tz_localize(None)
        valid_times = valid_times.astype('int64') // 10**9
        valid_times = valid_times.values
    else:
        first_timestamp = data[traj_cols['timestamp']].iloc[0]
        timestamp_length = len(str(first_timestamp))
        
        if timestamp_length > 10:
            if timestamp_length == 13:
                valid_times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 3
                
            elif timestamp_length == 19:
                valid_times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 9
        else:
            valid_times = data[traj_cols['timestamp']].values
        
    if not neighbor_dict:
        neighbor_dict = _find_neighbors(data, time_thresh, dist_thresh, long_lat, datetime, traj_cols)
    else:
        neighbor_dict = defaultdict(set,
                                    {k: v.intersection(valid_times) for k, v in
                                     neighbor_dict.items() if
                                     k in valid_times})

    cluster_df = pd.Series(-2, index=valid_times, name='cluster')
    core_df = pd.Series(-3, index=valid_times, name='core')
    
    # Initialize cluster label
    cid = -1
    
    for i, cluster in cluster_df.items():
        if cluster < 0:
            if len(neighbor_dict[i]) < min_pts:
                # Mark as noise if below min_pts
                cluster_df[i] = -1
            else:
                cid += 1
                cluster_df[i] = cid  # Assign new cluster label
                core_df[i] = cid  # Assign new core label
                S = list(neighbor_dict[i])  # Initialize stack with neighbors
                while S:
                    j = S.pop()
                    if cluster_df[j] < 0:  # Process if not yet in a cluster
                        cluster_df[j] = cid
                        if len(neighbor_dict[j]) >= min_pts:
                            core_df[j] = cid  # Assign core label
                            for k in neighbor_dict[j]:
                                if cluster_df[k] < 0:
                                    S.append(k)  # Add new neighbors

    return pd.DataFrame({'cluster': cluster_df, 'core': core_df})


def _process_clusters(data, time_thresh, dist_thresh, min_pts, output, long_lat, datetime, traj_cols, 
                     cluster_df=None, neighbor_dict=None, min_duration=4):
    """
    Recursively process spatiotemporal clusters from trajectory data to identify and refine valid clusters.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Trajectory data containing spatial and temporal information.
    time_thresh : int
        Time threshold in minutes for identifying neighbors.
    dist_thresh : float
        Distance threshold for identifying neighbors.
    min_pts : int
        Minimum number of points required to form a dense region (core point).
    output : pandas.DataFrame
        Output DataFrame to store cluster and core labels for valid clusters.
    long_lat : bool
        Whether to use longitude/latitude coordinates.
    datetime : bool
        Whether to process timestamps as datetime objects.
    traj_cols : dict
        Dictionary mapping column names for trajectory attributes.
    cluster_df : pandas.DataFrame, optional
        DataFrame containing cluster and core labels from DBSCAN. If not provided,
        it will be computed.
    neighbor_dict : dict, optional
        Precomputed dictionary of neighbors. If not provided, it will be computed.
    min_duration : int, optional
        Minimum duration (in minutes) required for a cluster to be considered valid (default is 4).
    
    Returns
    -------
    bool
        True if at least one valid cluster is identified and processed, otherwise False.
    """
    if not neighbor_dict:
        neighbor_dict = _find_neighbors(data, time_thresh, dist_thresh, long_lat, datetime, traj_cols)
    if cluster_df is None:
        cluster_df = dbscan(data, time_thresh, dist_thresh, min_pts, long_lat, datetime, traj_cols, neighbor_dict=neighbor_dict)
    if len(cluster_df) < min_pts:
        return False

    cluster_df = cluster_df[cluster_df['cluster'] != -1]  # Remove noise pings

    # All pings are in the same cluster
    if len(cluster_df['cluster'].unique()) == 1:
        # We rerun dbscan because possibly these points no longer hold their own
        x = dbscan(data = data.loc[cluster_df.index], time_thresh = time_thresh, dist_thresh = dist_thresh,
                   min_pts = min_pts, long_lat = long_lat, datetime = datetime, traj_cols = traj_cols, neighbor_dict = neighbor_dict)
        
        y = x.loc[x['cluster'] != -1]
        z = x.loc[x['core'] != -1]

        if len(y) > 0:
            duration = int((y.index.max() - y.index.min()) // 60)

            if duration > min_duration:
                cid = max(output['cluster']) + 1 # Create new cluster id
                output.loc[y.index, 'cluster'] = cid
                output.loc[z.index, 'core'] = cid
            
            return True
        elif len(y) == 0: # The points in df, despite originally being part of a cluster, no longer hold their own
            return False

    # There are no clusters
    elif len(cluster_df['cluster'].unique()) == 0:
        return False
   
    # There is more than one cluster
    elif len(cluster_df['cluster'].unique()) > 1:
        i, j = _extract_middle(cluster_df)  # Indices of the "middle" of the cluster
        
        # Recursively processes clusters
        if _process_clusters(data, time_thresh, dist_thresh, min_pts, output, long_lat, datetime, traj_cols, cluster_df = cluster_df[i:j]):  # Valid cluster in the middle
            _process_clusters(data, time_thresh, dist_thresh, min_pts, output,
                             long_lat, datetime, traj_cols, cluster_df = cluster_df[:i])  # Process the initial stub
            _process_clusters(data, time_thresh, dist_thresh, min_pts, output,
                             long_lat, datetime, traj_cols, cluster_df = cluster_df[j:])  # Process the "tail"
            return True
        else:  # No valid cluster in the middle
            return _process_clusters(data, time_thresh, dist_thresh, min_pts, output, long_lat, datetime, traj_cols, pd.concat([cluster_df[:i], cluster_df[j:]]))


def temporal_dbscan(data, time_thresh, dist_thresh, min_pts, traj_cols=None, complete_output=False, **kwargs):
    # Check if user wants long and lat
    long_lat = 'latitude' in kwargs and 'longitude' in kwargs and kwargs['latitude'] in data.columns and kwargs['longitude'] in data.columns

    # Check if user wants datetime
    datetime = 'datetime' in kwargs and kwargs['datetime'] in data.columns

    # Set initial schema
    if not traj_cols:
        traj_cols = {}

    traj_cols = loader._update_schema(traj_cols, kwargs)
    traj_cols = loader._update_schema(constants.DEFAULT_SCHEMA, traj_cols)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    # Setting x and y as defaults if not specified by user in either traj_cols or kwargs
    if traj_cols['x'] in data.columns and traj_cols['y'] in data.columns and not long_lat:
        long_lat = False
    else:
        long_lat = True

    # Setting timestamp as default if not specified by user in either traj_cols or kwargs
    if traj_cols['timestamp'] in data.columns and not datetime:
        datetime = False
    else:
        datetime = True

    if datetime:
        time_col_name = traj_cols['datetime']
    else:
        first_timestamp = data[traj_cols['timestamp']].iloc[0]
        timestamp_length = len(str(first_timestamp))
        
        if timestamp_length > 10:
            if timestamp_length == 13:
                warnings.warn(
                    f"The '{data[traj_cols['timestamp']]}' column appears to be in milliseconds. "
                    "This may lead to inconsistencies."
                )
                time_col_name = traj_cols['timestamp']
            elif timestamp_length == 19:
                warnings.warn(
                    f"The '{timestamp_col_name}' column appears to be in nanoseconds. "
                    "This may lead to inconsistencies."
                )
                time_col_name = traj_cols['timestamp']
        else:
            time_col_name = traj_cols['timestamp']

    output = _temporal_dbscan_labels(data, time_thresh, dist_thresh, min_pts, traj_cols, **kwargs)
    
    output = output[output['cluster'] != -1]
    
    complete_data = pd.merge(output, data, left_index=True, right_on=time_col_name, how='inner')
    
    stop_table = complete_data.groupby('cluster').apply(lambda group: _stop_metrics(group, long_lat, datetime, traj_cols, complete_output), include_groups=False)
    
    return stop_table


def _temporal_dbscan_labels(data, time_thresh, dist_thresh, min_pts, traj_cols=None, **kwargs):
    # Check if user wants long and lat
    long_lat = 'latitude' in kwargs and 'longitude' in kwargs and kwargs['latitude'] in data.columns and kwargs['longitude'] in data.columns

    # Check if user wants datetime
    datetime = 'datetime' in kwargs and kwargs['datetime'] in data.columns

    # Set initial schema
    if not traj_cols:
        traj_cols = {}

    traj_cols = loader._update_schema(traj_cols, kwargs)
    traj_cols = loader._update_schema(constants.DEFAULT_SCHEMA, traj_cols)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    # Setting x and y as defaults if not specified by user in either traj_cols or kwargs
    if traj_cols['x'] in data.columns and traj_cols['y'] in data.columns and not long_lat:
        long_lat = False
    else:
        long_lat = True

    # Setting timestamp as default if not specified by user in either traj_cols or kwargs
    if traj_cols['timestamp'] in data.columns and not datetime:
        datetime = False
    else:
        datetime = True

    if datetime:
        time_col_name = traj_cols['datetime']

        valid_times = pd.to_datetime(data[traj_cols['datetime']])
        valid_times = valid_times.dt.tz_convert('UTC').dt.tz_localize(None)
        valid_times = valid_times.astype('int64') // 10**9
        valid_times = valid_times.values
    else:
        first_timestamp = data[traj_cols['timestamp']].iloc[0]
        timestamp_length = len(str(first_timestamp))
        
        if timestamp_length > 10:
            if timestamp_length == 13:
                warnings.warn(
                    f"The '{data[traj_cols['timestamp']]}' column appears to be in milliseconds. "
                    "This may lead to inconsistencies."
                )
                valid_times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 3
                time_col_name = traj_cols['timestamp']
            elif timestamp_length == 19:
                warnings.warn(
                    f"The '{timestamp_col_name}' column appears to be in nanoseconds. "
                    "This may lead to inconsistencies."
                )
                valid_times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 9
                time_col_name = traj_cols['timestamp']
        else:
            valid_times = data[traj_cols['timestamp']].values
            time_col_name = traj_cols['timestamp']

    data_temp = data.copy()
    data_temp.index = valid_times        

    output = pd.DataFrame({'cluster': -1, 'core': -1}, index=valid_times)

    _process_clusters(data_temp, time_thresh, dist_thresh, min_pts, output, long_lat, datetime, traj_cols, min_duration=4)

    output.index = list(data[time_col_name])

    return output

def _stop_metrics(grouped_data, long_lat, datetime, traj_cols, complete_output):
    # Coordinates array and distance metrics
    if long_lat:
        coords = grouped_data[[traj_cols['longitude'], traj_cols['latitude']]].to_numpy()
        stop_medoid = utils._medoid(coords, metric='haversine')
        diameter_m = utils._diameter(coords, metric='haversine')
    else:
        coords = grouped_data[[traj_cols['x'], traj_cols['y']]].to_numpy()
        stop_medoid = utils._medoid(coords, metric='euclidean')
        diameter_m = utils._diameter(coords, metric='euclidean')

    # Compute duration and start and end time of stop
    if datetime:
        start_time = grouped_data[traj_cols['datetime']].min()
        end_time = grouped_data[traj_cols['datetime']].max()
        duration = (end_time - start_time).total_seconds() / 60.0
    else:
        start_time = grouped_data[traj_cols['timestamp']].min()
        end_time = grouped_data[traj_cols['timestamp']].max()
        
        timestamp_length = len(str(start_time))

        if timestamp_length > 10:
            if timestamp_length == 13:
                duration = ((end_time // 10 ** 3) - (start_time // 10 ** 3)) / 60.0
            elif timestamp_length == 19:
                duration = ((end_time // 10 ** 9) - (start_time // 10 ** 9)) / 60.0
        else:
            duration = (end_time - start_time) / 60.0
                            
    # Number of pings in stop
    n_pings = len(grouped_data)
    
    # Compute max_gap between consecutive pings (in minutes)
    if datetime:
        times = pd.to_datetime(grouped_data[traj_cols['datetime']]).sort_values()
        time_diffs = times.diff().dropna()
        max_gap = int(time_diffs.max().total_seconds() / 60) if not time_diffs.empty else 0
    else:
        times = grouped_data[traj_cols['timestamp']].sort_values()
        timestamp_length = len(str(times.iloc[0]))
        
        if timestamp_length == 13:
            time_diffs = np.diff(times.values) // 1000
        elif timestamp_length == 19:  # nanoseconds
            time_diffs = np.diff(times.values) // 10**9
        else:
            time_diffs = np.diff(times.values)
        
        max_gap = int(np.max(time_diffs) / 60) if len(time_diffs) > 0 else 0

    # Prepare data for the Series
    if long_lat:
        if complete_output:
            stop_attr = {
                'start_time': start_time,
                'end_time': end_time,
                traj_cols['longitude']: stop_medoid[0],
                traj_cols['latitude']: stop_medoid[1],
                'diameter': diameter_m,
                'n_pings': n_pings,
                'duration': duration,
                'max_gap': max_gap
            }
        else:
            stop_attr = {
                'start_time': start_time,
                'duration': duration,
                traj_cols['longitude']: stop_medoid[0],
                traj_cols['latitude']: stop_medoid[1]
            }
    else:
        if complete_output:
            stop_attr = {
                'start_time': start_time,
                'end_time': end_time,
                traj_cols['x']: stop_medoid[0],
                traj_cols['y']: stop_medoid[1],
                'diameter': diameter_m,
                'n_pings': n_pings,
                'duration': duration,
                'max_gap': max_gap
            }
        else:
            stop_attr = {
                'start_time': start_time,
                'duration': duration,
                traj_cols['x']: stop_medoid[0],
                traj_cols['y']: stop_medoid[1]
            }

    return pd.Series(stop_attr)