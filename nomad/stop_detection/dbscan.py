# TO DO: Normal DBSCAN with simple splitting rule
import pandas as pd
import numpy as np
from collections import defaultdict
import pdb
import warnings
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

def ta_dbscan(data, time_thresh, dist_thresh, min_pts, traj_cols=None, complete_output=False, **kwargs):
    # Load default col names
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    # Set initial schema
    if not traj_cols:
        traj_cols = {}
    
    traj_cols = loader._update_schema(traj_cols, kwargs)
    traj_cols = loader._update_schema(constants.DEFAULT_SCHEMA, traj_cols)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    labels_tadbscan = ta_dbscan_labels(data, time_thresh, dist_thresh, min_pts, False, traj_cols, **kwargs) # see patch on 424

    complete_data = data.join(labels_tadbscan)
    complete_data = complete_data[complete_data['cluster'] != -1]

    stop_table = complete_data.groupby('cluster', as_index=False).apply(
            lambda grouped_data: utils.summarize_stop(grouped_data,
                                           complete_output=complete_output,
                                           traj_cols=traj_cols,
                                           keep_col_names=False,
                                           **kwargs),
            include_groups=False)

    return stop_table
    # return labels_tadbscan, stop_table

def ta_dbscan_labels(data, time_thresh, dist_thresh, min_pts, return_cores=False, traj_cols=None, **kwargs):
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
                    f"The '{traj_cols['timestamp']}' column appears to be in milliseconds. "
                    "This may lead to inconsistencies."
                )
                valid_times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 3
                time_col_name = traj_cols['timestamp']
            elif timestamp_length == 19:
                warnings.warn(
                    f"The '{traj_cols['timestamp']}' column appears to be in nanoseconds. "
                    "This may lead to inconsistencies."
                )
                valid_times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 9
                time_col_name = traj_cols['timestamp']
        else:
            valid_times = data[traj_cols['timestamp']].values
            time_col_name = traj_cols['timestamp']

    data_temp = data.copy()
    data_temp.index = valid_times    

    output = dbscan(data, time_thresh, dist_thresh, min_pts, long_lat, datetime, traj_cols)

    if return_cores:
        output.index = list(data[time_col_name])
        return output
    else:
        labels = output.cluster
        return labels.set_axis(data.index)