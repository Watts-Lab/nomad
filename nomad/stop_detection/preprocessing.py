import pandas as pd
import numpy as np
from collections import defaultdict
from nomad.stop_detection import utils
from nomad.filters import to_timestamp

def _find_spatio_temp_neighbors(times, coords, time_thresh, dist_thresh, use_lon_lat, use_datetime):
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
    use_lon_lat : bool
        Whether to use longitude/latitude coordinates.
    use_datetime : bool
        Whether to cluster using time columns of type pandas.datetime64 if available.
    
    Returns
    -------
    dict
        A dictionary where keys are timestamps, and values are sets of neighboring
        timestamps that satisfy both time and distance thresholds.
    """
    # getting times based on whether they are datetime values or timestamps, changed to seconds for calculations
    times = to_timestamp(times).values if use_datetime else times.values
      
    # Pairwise time differences
    time_diffs = np.abs(times[:, np.newaxis] - times)
    time_diffs = time_diffs.astype(int)
  
    # Filter by time threshold
    within_time_thresh = np.triu(time_diffs <= (time_thresh * 60), k=1)
    i_idx, j_idx = np.where(within_time_thresh)
  
    # Distance calculation
    if use_lon_lat:
        distances = np.array([utils._haversine_distance(coords[i], coords[j]) for i, j in zip(i_idx,j_idx)])
    else:
        distances = np.sqrt((coords[i_idx, 0] - coords[j_idx, 0])**2 + (coords[i_idx, 1] - coords[j_idx, 1])**2)

    # Filter by distance threshold
    neighbor_pairs = distances < dist_thresh
  
    # Building the neighbor dictionary
    neighbor_dict = defaultdict(set)
  
    for i, j in zip(i_idx[neighbor_pairs], j_idx[neighbor_pairs]):
        neighbor_dict[times[i]].add(times[j])
        neighbor_dict[times[j]].add(times[i])

    return neighbor_dict

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
    times = to_timestamp(times).values if use_datetime else times.values
        
    # Pairwise time differences
    # times[:, np.newaxis]: from shape (n,) -> to shape (n, 1) – a column vector
    time_diffs = np.abs(times[:, np.newaxis] - times)
    time_diffs = time_diffs.astype(int)
    
    # Filter by time threshold
    within_time_thresh = np.triu(time_diffs <= (time_thresh * 60), k=1) # keep upper triangle
    i_idx, j_idx = np.where(within_time_thresh)
    
    # Return a list of (timestamp1, timestamp2) tuples
    time_pairs = [(times[i], times[j]) for i, j in zip(i_idx, j_idx)]
    
    return time_pairs, times