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

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
import daphmeIO as loader
import constants as constants

##########################################
######## STOP DETECTION FUNCTIONS ########
##########################################

##########################################
########        Lachesis          ########
##########################################

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
        coords = np.radians(coords)
        distances = _pairwise_haversine(coords)
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
    # traj = traj.copy()
    
    # Check if user wants long and lat
    long_lat = 'latitude' in kwargs and 'longitude' in kwargs and kwargs[
        'latitude'] in traj.columns and kwargs['longitude'] in traj.columns

    # Check if user wants datetime
    datetime = 'datetime' in kwargs and kwargs['datetime'] in traj.column

    # Set initial schema
    if not traj_cols:
        traj_cols = {}

    traj_cols = loader._update_schema(traj_cols, kwargs)
    traj_cols = loader._update_schema(constants.DEFAULT_SCHEMA, traj_cols)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(traj.columns, traj_cols)
    loader._has_time_cols(traj.columns, traj_cols)

    # Setting x and y as defaults if not specified by user in either traj_cols or kwargs
    if traj_cols['x'] in traj.columns and traj_cols['y'] in traj.columns:
        long_lat = False
        coords = traj[[traj_cols['x'], traj_cols['y']]].to_numpy()
    else:
        long_lat = True
        coords = traj[[traj_cols['longitude'], traj_cols['latitude']]].to_numpy()

    # Setting timestamp as default if not specified by user in either traj_cols or kwargs
    if traj_cols['timestamp'] in traj.columns:
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

        d_start = _diameter(coords[i:j_star + 1],
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
                d_update = _update_diameter(coords[j], coords[i:j], d_start,
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
                stop_medoid = _medoid(coords[i:j_star + 1])
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
    
    # Determine the timestamp column to use
    if traj_cols.get('timestamp') in traj.columns:
        datetime = False
        timestamp_col = traj_cols['timestamp']
    elif traj_cols.get('datetime') in traj.columns:
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

##########################################
########         DBSCAN           ########
##########################################

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

    if long_lat:
        coords = np.radians(data[[traj_cols['latitude'], traj_cols['longitude']]].values)
    else:
        coords = data[[traj_cols['x'], traj_cols['y']]].values

    if datetime:
        times = data[traj_cols['datetime']].astype('datetime64[s]').astype(int).values
    else:
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

    # Time threshold calculation using broadcasting
    within_time_thresh = np.triu(time_diffs <= (time_thresh * 60), k=1)
    time_pairs = np.where(within_time_thresh)
    
    # Distance calculation
    if long_lat:
        distances = np.array([_haversine_distance(coords[i], coords[j]) for i, j in zip(*time_pairs)])
    else:
        distances_sq = (coords[time_pairs[0], 0] - coords[time_pairs[1], 0])**2 + \
                       (coords[time_pairs[0], 1] - coords[time_pairs[1], 1])**2
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
    if datetime:
        valid_times = data[traj_cols['datetime']].astype('datetime64[s]').astype(int).values
        original_times = data[traj_cols['datetime']].values
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
                original_times = data[traj_cols['timestamp']].values
            elif timestamp_length == 19:
                warnings.warn(
                    f"The '{timestamp_col_name}' column appears to be in nanoseconds. "
                    "This may lead to inconsistencies."
                )
                valid_times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 9 
                original_times = data[traj_cols['timestamp']].values
        else:
            valid_times = data[traj_cols['timestamp']].values
            original_times = data[traj_cols['timestamp']].values
    
    if not neighbor_dict:
        neighbor_dict = _find_neighbors(data, time_thresh, dist_thresh, long_lat, datetime, traj_cols)
    else:
        neighbor_dict = defaultdict(set,
                                    {k: v.intersection(valid_times) for k, v in
                                     neighbor_dict.items() if
                                     k in valid_times})
    
    cluster_df = pd.Series(-2, index=valid_times, name='cluster')
    core_df = pd.Series(-3, index=valid_times, name='core')

    cid = -1  # Initialize cluster label

    for i, cluster in cluster_df.items():
        if cluster < 0:  # Check if point is not yet in a cluster
            if len(neighbor_dict[i]) < min_pts:
                cluster_df[i] = -1  # Mark as noise if below min_pts
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
    datetime = 'datetime' in kwargs and kwargs['datetime'] in data.column

    # Set initial schema
    if not traj_cols:
        traj_cols = {}

    traj_cols = loader._update_schema(traj_cols, kwargs)
    traj_cols = loader._update_schema(constants.DEFAULT_SCHEMA, traj_cols)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    # Setting x and y as defaults if not specified by user in either traj_cols or kwargs
    if traj_cols['x'] in data.columns and traj_cols['y'] in data.columns:
        long_lat = False
    else:
        long_lat = True

    # Setting timestamp as default if not specified by user in either traj_cols or kwargs
    if traj_cols['timestamp'] in data.columns:
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
    datetime = 'datetime' in kwargs and kwargs['datetime'] in data.column

    # Set initial schema
    if not traj_cols:
        traj_cols = {}

    traj_cols = loader._update_schema(traj_cols, kwargs)
    traj_cols = loader._update_schema(constants.DEFAULT_SCHEMA, traj_cols)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    # Setting x and y as defaults if not specified by user in either traj_cols or kwargs
    if traj_cols['x'] in data.columns and traj_cols['y'] in data.columns:
        long_lat = False
    else:
        long_lat = True

    # Setting timestamp as default if not specified by user in either traj_cols or kwargs
    if traj_cols['timestamp'] in data.columns:
        datetime = False
    else:
        datetime = True

    if datetime:
        valid_times = data[traj_cols['datetime']].astype('datetime64[s]').astype(int).values
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
        stop_medoid = _medoid(coords, metric='haversine')
        diameter_m = _diameter(coords, metric='haversine')
    else:
        coords = grouped_data[[traj_cols['x'], traj_cols['y']]].to_numpy()
        stop_medoid = _medoid(coords, metric='euclidean')
        diameter_m = _diameter(coords, metric='euclidean')

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
                'duration': duration
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
                'duration': duration
            }
        else:
            stop_attr = {
                'start_time': start_time,
                'duration': duration,
                traj_cols['x']: stop_medoid[0],
                traj_cols['y']: stop_medoid[1]
            }

    return pd.Series(stop_attr)