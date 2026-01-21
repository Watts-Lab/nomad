import pandas as pd
import numpy as np
from collections import defaultdict
from nomad.stop_detection import utils
from nomad.filters import to_timestamp
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree # for haverside distance case
import pdb

def _find_temp_neighbors(times, time_thresh, use_datetime):
    """
    Find timestamp pairs that are within time threshold.

    Parameters
    ----------
    times : array of timestamps.
    time_thresh : time threshold for finding what timestamps are close in time.
    use_datetime : Whether to process timestamps as datetime objects.

    Returns
    -------
    time_pairs : list of tuples of timestamps [(t1, t2), ...] that are close in time given time_thresh.

    TC: O(n^2)
    """
    # getting times based on whether they are datetime values or timestamps, changed to seconds for calculations
    if use_datetime:
        times = to_timestamp(times).values
    else:
        times = times.values
        
    # Pairwise time differences
    # times[:, np.newaxis]: from shape (n,) -> to shape (n, 1) – a column vector
    time_diffs = np.abs(times[:, np.newaxis] - times).astype(int)
    
    # Filter by time threshold
    within_time_thresh = np.triu(time_diffs <= (time_thresh * 60), k=1) # keep upper triangle
    i_idx, j_idx = np.where(within_time_thresh)
    
    # Return a list of (timestamp1, timestamp2) tuples
    time_pairs = [(times[i], times[j]) for i, j in zip(i_idx, j_idx)]
    
    return time_pairs, times

def _find_neighbors(data, time_thresh, dist_thresh, use_lon_lat, use_datetime, traj_cols):
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
    use_lon_lat : bool, optional
        Whether to use longitude/latitude coordinates.
    use_datetime : bool, optional
        Whether to process timestamps as datetime objects.
    
    Returns
    -------
    dict
        A dictionary where keys are timestamps, and values are sets of neighboring
        timestamps that satisfy both time and distance thresholds.
    """
    # getting coordinates based on whether they are geographic coordinates (lon, lat) or catesian (x,y)
    if use_lon_lat:
        coords = np.radians(data[[traj_cols['latitude'], traj_cols['longitude']]].values)
    else:
        coords = data[[traj_cols['x'], traj_cols['y']]].values
    
    # getting times based on whether they are datetime values or timestamps, changed to seconds for calculations
    if use_datetime:
        times = to_timestamp(data[traj_cols['datetime']]).values
    else:
        times = data[traj_cols['timestamp']].values
      
    ## TEMPORAL THRESHOLDING ##
    # Pairwise time differences    
    t_kdtree = KDTree(times.reshape(-1, 1))
    time_pairs = t_kdtree.query_pairs(r=time_thresh * 60)
  
    ## DISTANCE THRESHOLDING ##
    if use_lon_lat:
        earth_radius = 6_371_000
        s_balltree = BallTree(coords, metric = 'haversine')
        indices = s_balltree.query_radius(coords, r=dist_thresh/earth_radius, return_distance=False)
        dist_pairs = _adj_arr_to_pairs(indices)
        neighbor_pairs = (time_pairs & dist_pairs)
    else:
        s_kdtree = KDTree(coords)
        dist_pairs = s_kdtree.query_pairs(r=dist_thresh)
        neighbor_pairs = (time_pairs & dist_pairs)
  
    ## BUILD NEIGHBOR DICTIONARY ##
    neighbor_dict = defaultdict(set)
    for i, j in neighbor_pairs:
            neighbor_dict[times[i]].add(times[j])
            neighbor_dict[times[j]].add(times[i])

    return neighbor_dict

def _adj_arr_to_pairs(ind):
    # Vectorized helper to get equal output as kdtree when using balltree
    counts = [len(x) for x in ind]
    row_idx = np.repeat(np.arange(len(ind)), counts)
    col_idx = np.concatenate(ind)
    mask = row_idx < col_idx
    return set(zip(row_idx[mask].tolist(), col_idx[mask].tolist()))
    