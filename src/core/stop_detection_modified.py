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
from heapq import heappush
from heapq import heappop

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
import daphmeIO as loader
import constants as constants

import warnings
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster
from heapq import heappop, heappush

# ##########################################
# ###############   SPARK   ################
# ##########################################

# from pyspark.sql.functions import col, lit, udf, array
# from pyspark.sql.types import DoubleType
# import numpy as np
# from scipy.spatial.distance import pdist
# from pyspark.sql.functions import collect_list
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, lead, lag, unix_timestamp, when, collect_list, count, min, max
# from pyspark.sql.window import Window
# from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType, TimestampType, ArrayType
# import numpy as np


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
    # getting coordinates based on whether they are geographic coordinates (lon, lat) or catesian (x,y)
    if long_lat:
        coords = np.radians(data[[traj_cols['latitude'], traj_cols['longitude']]].values)
    else:
        coords = data[[traj_cols['x'], traj_cols['y']]].values
    
    # getting times based on whether they are datetime values or timestamps, changed to seconds for calculations
    if datetime:
        times = data[traj_cols['datetime']].astype('datetime64[s]').astype(int).values
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
        distances = np.array([_haversine_distance(coords[i], coords[j]) for i, j in zip(*time_pairs)])
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
        valid_times = data[traj_cols['datetime']].astype('datetime64[s]').astype(int).values
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
    # Find neighbors within the data if they have not been passed into as an arg
    if not neighbor_dict:
        neighbor_dict = _find_neighbors(data, time_thresh, dist_thresh, long_lat, datetime, traj_cols)
    
    # Get initial clusters if cluster_df is not passed into as an arg
    if cluster_df is None:
        cluster_df = dbscan(data, time_thresh, dist_thresh, min_pts, long_lat, datetime, traj_cols, neighbor_dict=neighbor_dict)
    
    # Ensures that a cluster contains at least min_pts points before processing it further
    if len(cluster_df) < min_pts:
        return False

    # Remove noise pings
    cluster_df = cluster_df[cluster_df['cluster'] != -1]

    # There are no clusters
    if len(cluster_df['cluster'].unique()) == 0:
        return False
    
    # All pings are in the same cluster
    elif len(cluster_df['cluster'].unique()) == 1:
        # Rerun dbscan to further refine cluster
        x = dbscan(data = data.loc[cluster_df.index], time_thresh = time_thresh, dist_thresh = dist_thresh,
                   min_pts = min_pts, long_lat = long_lat, datetime = datetime, traj_cols = traj_cols, neighbor_dict = neighbor_dict)
        
        y = x.loc[x['cluster'] != -1]
        z = x.loc[x['core'] != -1]

        # New clusters were found after rerunning
        if len(y) > 0:
            duration = int((y.index.max() - y.index.min()) // 60)

            if duration > min_duration:
                # Create new cluster id
                cid = max(output['cluster']) + 1 
                output.loc[y.index, 'cluster'] = cid
                output.loc[z.index, 'core'] = cid
            
            return True
        # The points despite originally being part of a cluster, are not considered noise.
        elif len(y) == 0:
            return False
   
    # There is more than one cluster
    elif len(cluster_df['cluster'].unique()) > 1:
        # Indices of the "middle" of the data
        i, j = _extract_middle(cluster_df)
        
        # Recursively processes clusters if there is a valid cluster in the middle
        if _process_clusters(data, time_thresh, dist_thresh, min_pts, output, long_lat, datetime, traj_cols, cluster_df = cluster_df[i:j]):
            # Process beginning portion ()
            _process_clusters(data, time_thresh, dist_thresh, min_pts, output,
                             long_lat, datetime, traj_cols, cluster_df = cluster_df[:i])  
            # Process ending portion
            _process_clusters(data, time_thresh, dist_thresh, min_pts, output,
                             long_lat, datetime, traj_cols, cluster_df = cluster_df[j:])
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
                    f"The '{data[traj_cols['timestamp']]}' column appears to be in nanoseconds. "
                    "This may lead to inconsistencies."
                )
                time_col_name = traj_cols['timestamp']
        else:
            time_col_name = traj_cols['timestamp']

    # Getting the labels for each point
    output = _temporal_dbscan_labels(data, time_thresh, dist_thresh, min_pts, traj_cols, **kwargs)
    
    # Filtering out the noise
    output = output[output['cluster'] != -1]
    
    # Merging the labels with the original data
    complete_data = pd.merge(output, data, left_index=True, right_on=time_col_name, how='inner')
    
    # Get stop table metrics
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
                    f"The '{data[traj_cols['timestamp']]}' column appears to be in milliseconds."
                    "This may lead to inconsistencies."
                )
                valid_times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 3
                time_col_name = traj_cols['timestamp']
            elif timestamp_length == 19:
                warnings.warn(
                    f"The '{data[traj_cols['timestamp']]}' column appears to be in nanoseconds."
                    "This may lead to inconsistencies."
                )
                valid_times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 9
                time_col_name = traj_cols['timestamp']
        else:
            valid_times = data[traj_cols['timestamp']].values
            time_col_name = traj_cols['timestamp']

    # Getting the labels through _process_clusters
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

##########################################
########        HDBSCAN           ########
##########################################
def _find_neighbors_hdbscan(data, time_thresh, dist_thresh, long_lat, datetime, traj_cols, alpha=0.5):
    """
    - Computes pairwise distances between points based on space & time.
    - If two points are connected, the matrix stores the weighted distance.
    - Returns a weighted adjacency matrix, where edges represent spatiotemporal proximity.
    """
    if long_lat:
        coords = np.radians(data[[traj_cols['latitude'], traj_cols['longitude']]].values)
    else:
        coords = data[[traj_cols['x'], traj_cols['y']]].values

    if datetime:
        times = data[traj_cols['datetime']].astype('datetime64[s]').astype(int).values
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

    n = len(data)

    adjacency_matrix = {t: {} for t in times}
    
    # Pairwise time differences (convert seconds to minutes)
    time_diffs = np.abs(times[:, np.newaxis] - times) / 60.0

    # Compute spatial distances
    if long_lat:
        distances = np.array([[_haversine_distance(coords[i], coords[j]) for j in range(n)] for i in range(n)])
    else:
        distances = np.sqrt((coords[:, 0][:, np.newaxis] - coords[:, 0])**2 + 
                            (coords[:, 1][:, np.newaxis] - coords[:, 1])**2)

    # Normalize distances
    max_dist = np.max(distances) if np.max(distances) > 0 else 1
    max_time = np.max(time_diffs) if np.max(time_diffs) > 0 else 1

    distances /= max_dist
    time_diffs /= max_time

    # compute weighted combination
    weighted_distances = alpha * distances + (1 - alpha) * time_diffs

    # apply a threshold to keep only valid neighbors
    threshold = alpha * dist_thresh / max_dist + (1 - alpha) * time_thresh / max_time
    
    for i, t1 in enumerate(times):
        for j, t2 in enumerate(times):
            if weighted_distances[i, j] < threshold and i != j:
                adjacency_matrix[t1][t2] = weighted_distances[i, j]
    
    return adjacency_matrix

def _construct_mst(adjacency_matrix):
    """
    - Uses Prim’s algorithm to construct a Minimum Spanning Tree (MST).
    - Works with a dictionary-based adjacency matrix indexed by timestamps.
    - The edges retain spatiotemporal weights computed earlier.
    - Ensures that every point remains part of the same structure.

    Parameters
    ----------
    adjacency_matrix : dict of dict
        A dictionary where keys are timestamps and values are dictionaries of {neighbor_timestamp: weight}.

    Returns
    -------
    list of tuples
        A list of MST edges in the format: (weight, timestamp1, timestamp2).
    """
    timestamps = list(adjacency_matrix.keys())
    mst_edges = []
    visited = {t: False for t in timestamps}
    pq = []

    def find_next_start():
        for t in timestamps:
            if not visited[t] and adjacency_matrix[t]:
                return t
        return None

    start_node = find_next_start()

    while start_node is not None:
        visited[start_node] = True
        for neighbor, weight in adjacency_matrix[start_node].items():
            heappush(pq, (weight, start_node, neighbor))

        while pq:
            weight, u, v = heappop(pq)

            # only skip edges where both nodes are visited
            if visited[u] and visited[v]:
                continue

            mst_edges.append((weight, u, v))

            # visit the unvisited node
            next_node = v if not visited[v] else u
            visited[next_node] = True

            for neighbor, edge_weight in adjacency_matrix[next_node].items():
                if not visited[neighbor]:
                    heappush(pq, (edge_weight, next_node, neighbor))

        start_node = find_next_start()

    return mst_edges

def _extract_hierarchy(mst_edges, all_timestamps):
    """
    - Sorts MST edges by decreasing weight (weakest connections are removed first).
    - Start with individual points as their own clusters.
    - Iteratively merges clusters, creating a tree-like structure (dendrogram).
    - Assigns `-1` to completely isolated timestamps (i.e., those missing from MST).
    
    Parameters
    ----------
    mst_edges : list of tuples
        A list of edges in the format: (weight, timestamp1, timestamp2).
    all_timestamps : set
        The full set of timestamps that should be included in the clustering process.
    
    Returns
    -------
    np.ndarray
        A hierarchical linkage matrix representing the clustering process.
    dict
        A dictionary mapping timestamps to their cluster labels (including `-1` for noise).
    """

    # sort edges by descending weight
    sorted_edges = sorted(mst_edges, reverse=True, key=lambda x: x[0])

    # initialize all timestamps as their own cluster
    clusters = {t: t for t in all_timestamps}
    cluster_tree = []
    cluster_count = max(all_timestamps) + 1

    for weight, u, v in sorted_edges:
        root_u, root_v = clusters[u], clusters[v]

        # only merge if they belong to different clusters
        if root_u != root_v:
            new_cluster = cluster_count
            cluster_count += 1

            # update all points in both clusters to the new cluster ID
            for key in clusters.keys():
                if clusters[key] in (root_u, root_v):
                    clusters[key] = new_cluster

            # compute cluster size
            cluster_size = sum(1 for x in clusters.values() if x == new_cluster)

            # append to the cluster tree
            cluster_tree.append((root_u, root_v, weight, cluster_size))

            
    # identify timestamps that were never merged (completely isolated)
    timestamps_in_mst = set()
    for _, u, v in mst_edges:
        timestamps_in_mst.update([u, v])

    isolated_timestamps = all_timestamps - timestamps_in_mst

    for t in isolated_timestamps:
        clusters[t] = -1

    return np.array(cluster_tree), clusters

def _flat_cut(linkage_matrix, clusters, threshold, min_cluster_size=5):
    """
    Perform a flat cut of the hierarchical clustering at the specified threshold.
    
    Parameters
    ----------
    linkage_matrix : np.ndarray
        The hierarchical linkage matrix from extract_hierarchy.
    clusters : dict
        Dictionary mapping timestamps to their initial cluster IDs.
    threshold : float
        The weight threshold at which to cut the dendrogram.
    min_cluster_size : int, default=5
        Minimum number of points required to form a valid cluster.
        
    Returns
    -------
    dict
        A dictionary mapping timestamps to their final cluster labels after the cut.
        Noise points are labeled as -1.
    """
    result_clusters = clusters.copy()
    
    # If we have an empty linkage matrix, return the original clusters
    if len(linkage_matrix) == 0:
        return result_clusters
    
    # Process the edges of the linkage matrix in order (from low to high weight)
    for i in range(len(linkage_matrix)):
        left, right, weight, size = linkage_matrix[i]
        
        # Only process edges with weights below the threshold
        if weight <= threshold:
            # Update all points in the 'right' cluster to be in the 'left' cluster
            for ts in result_clusters:
                if result_clusters[ts] == right:
                    result_clusters[ts] = left
    
    # Count the sizes of each cluster
    cluster_sizes = {}
    for cluster_id in result_clusters.values():
        if cluster_id == -1:
            continue
        if cluster_id not in cluster_sizes:
            cluster_sizes[cluster_id] = 0
        cluster_sizes[cluster_id] += 1
    
    # Mark clusters smaller than min_cluster_size as noise (-1)
    for ts in result_clusters:
        cluster_id = result_clusters[ts]
        if cluster_id != -1 and cluster_sizes.get(cluster_id, 0) < min_cluster_size:
            result_clusters[ts] = -1
    
    # Relabel the clusters to be consecutive integers starting from 0
    valid_clusters = sorted({c for c in result_clusters.values() if c != -1})
    mapping = {old_id: new_id for new_id, old_id in enumerate(valid_clusters)}
    mapping[-1] = -1  # Keep noise points as -1
    
    for ts in result_clusters:
        result_clusters[ts] = mapping[result_clusters[ts]]
    
    return result_clusters

def _find_optimal_threshold(linkage_matrix, clusters, min_cluster_size=5, percentiles=[5, 10, 25, 50, 75, 90, 95]):
    """
    Find the optimal threshold for flat cutting by evaluating a subset of possible thresholds
    based on percentile selection.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        The hierarchical linkage matrix.
    clusters : dict
        Dictionary mapping timestamps to their cluster IDs.
    min_cluster_size : int, default=5
        Minimum cluster size to consider valid.
    percentiles : list, default=[5, 10, 25, 50, 75, 90, 95]
        Percentile values to select a subset of thresholds.

    Returns
    -------
    float
        The optimal threshold value.
    """
    if len(linkage_matrix) == 0:
        return 0.0  # Return default if no valid threshold is found

    # Step 1: Extract unique weights from the linkage matrix
    all_thresholds = np.unique(linkage_matrix[:, 2])

    if len(all_thresholds) == 0:
        return 0.0  # No valid thresholds

    # Step 2: Select a subset of weights based on percentiles
    sampled_thresholds = np.percentile(all_thresholds, percentiles)

    # Cache results from `_flat_cut` to avoid redundant computation
    flat_cut_results = {}

    results = []
    for threshold in sampled_thresholds:
        # If already computed, reuse result
        if threshold in flat_cut_results:
            flat_clusters = flat_cut_results[threshold]
        else:
            flat_clusters = _flat_cut(linkage_matrix, clusters, threshold, min_cluster_size)
            flat_cut_results[threshold] = flat_clusters  # Cache result

        # Count valid clusters (excluding noise)
        valid_cluster_ids = set(flat_clusters.values()) - {-1}
        num_clusters = len(valid_cluster_ids)

        # Compute noise ratio (fraction of points labeled as noise)
        total_points = len(flat_clusters)
        assigned_points = sum(1 for c in flat_clusters.values() if c != -1)
        noise_ratio = 1.0 - (assigned_points / total_points if total_points > 0 else 0)

        # Compute stability (gap to the next threshold)
        if threshold == sampled_thresholds[-1]:
            stability = 0  # Last threshold has no stability measure
        else:
            next_idx = np.searchsorted(sampled_thresholds, threshold) + 1
            stability = sampled_thresholds[next_idx] - threshold if next_idx < len(sampled_thresholds) else 0

        # Step 3: Compute a score to determine the best threshold
        score = (
            (num_clusters * 0.4) +  # Favor more clusters
            ((1 - noise_ratio) * 0.4) +  # Favor lower noise
            (stability * 5.0 * 0.2)  # Favor stable thresholds
        )

        results.append((threshold, num_clusters, noise_ratio, stability, score))

    # Step 4: Pick the best threshold based on the highest score
    results.sort(key=lambda x: x[4], reverse=True)  # Sort by score descending

    if not results:
        return sampled_thresholds[0] if len(sampled_thresholds) > 0 else 0.0  # Return first threshold if no valid result

    best_threshold = results[0][0]
    return best_threshold

def hierarchical_temporal_dbscan(traj, time_thresh, dist_thresh, min_cluster_size = 10, traj_cols=None, complete_output=False, **kwargs):
    data = traj.copy()
    
    # Check if user wants long and lat
    long_lat = 'latitude' in kwargs and 'longitude' in kwargs and kwargs['latitude'] in data.columns and kwargs['longitude'] in data.columns

    # Check if user wants datetime
    datetime = 'datetime' in kwargs and kwargs['datetime'] in data.columns

    # Set initial schema
    if not traj_cols:
        traj_cols = {}

    traj_cols = loader._update_schema(traj_cols, kwargs)
    traj_cols = loader._update_schema(constants.DEFAULT_SCHEMA, traj_cols)

    # Ensure required spatial and temporal columns exist
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    # Determine if we are using lat/lon or cartesian (meters)
    if traj_cols['x'] in data.columns and traj_cols['y'] in data.columns:
        long_lat = False
    else:
        long_lat = True

    if traj_cols['timestamp'] in data.columns:
        datetime = False
    else:
        datetime = True

    # Timestamp handling
    if datetime:
        time_col_name = traj_cols['datetime']
        # times = data[traj_cols['datetime']].astype('datetime64[s]').astype(int).values
        data['mapped_time'] = data[traj_cols['datetime']].astype('datetime64[s]').astype(int)
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
                data['mapped_time'] = data[traj_cols['timestamp']].values.view('int64') // 10**3
                # times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 3
            elif timestamp_length == 19:
                warnings.warn(
                    f"The '{time_col_name}' column appears to be in nanoseconds. "
                    "This may lead to inconsistencies."
                )
                time_col_name = traj_cols['timestamp']
                data['mapped_time'] = data[traj_cols['timestamp']].values.view('int64') // 10**9
                # times = data[traj_cols['timestamp']].values.view('int64') // 10 ** 9
        else:
            time_col_name = traj_cols['timestamp']
            data['mapped_time'] = data[traj_cols['timestamp']].values.view('int64')
            # times = data[traj_cols['timestamp']].values

    # Build weighted adjacency matrix
    adjacency_matrix = _find_neighbors_hdbscan(data, time_thresh, dist_thresh, long_lat, datetime, traj_cols, alpha=0.5)

    # Construct MST
    mst_edges = _construct_mst(adjacency_matrix)

    # Extract hierarchical tree structure
    cluster_tree, clusters = _extract_hierarchy(mst_edges, set(data['mapped_time'].unique()))

    # Find the optimal threshold for flat cutting
    optimal_threshold = _find_optimal_threshold(cluster_tree, clusters, min_cluster_size)

    # Apply flat cut at optimal threshold
    final_clusters = _flat_cut(cluster_tree, clusters, optimal_threshold, min_cluster_size)

    # Assign cluster labels based on `mapped_time`
    data['cluster'] = data['mapped_time'].map(final_clusters).fillna(-1).astype(int)
    
    hdbscan_labels = data[[time_col_name, 'cluster']]
    hdbscan_labels = hdbscan_labels.set_index(traj_cols['datetime'])
    hdbscan_labels.index.name = None

    # Remove noise (-1 clusters)
    data = data[data['cluster'] != -1]

    # Compute stop metrics per cluster
    stop_table = data.groupby('cluster').apply(
        lambda group: _stop_metrics(group, long_lat, datetime, traj_cols, complete_output), include_groups=False)
    
    stop_table.index.name = None
    # Return stop metrics + timestamps with final cluster labels
    return stop_table, hdbscan_labels