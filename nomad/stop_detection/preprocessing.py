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
    if use_datetime:
        times = to_timestamp(times).values
    else:
        times = times.values

    t_kdtree = KDTree(times[:, None])
    time_pairs = t_kdtree.query_pairs(r=time_thresh * 60, output_type='ndarray')
    time_pairs = times[time_pairs]
    return time_pairs, times

def _find_neighbors(data, time_thresh, traj_cols, dist_thresh=None, weighted=False, use_datetime=False, use_lon_lat=False):
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
    if use_lon_lat:
        coords = np.radians(data[[traj_cols['latitude'], traj_cols['longitude']]].values)
    else:
        coords = data[[traj_cols['x'], traj_cols['y']]].values

    # getting times based on whether they are datetime values or timestamps, changed to seconds for calculations
    if use_datetime:
        times = to_timestamp(data[traj_cols['datetime']]).values
    else:
        times = data[traj_cols['timestamp']].values

    # Time   
    t_kdtree = KDTree(times[:, None])
    time_pairs = t_kdtree.query_pairs(r=time_thresh * 60, output_type='ndarray')
    time_pairs = times[time_pairs]
  
    # Distance
    if dist_thresh is None:
        pass
    else:
        if use_lon_lat:
            earth_radius = 6_371_000
            s_balltree = BallTree(coords, metric = 'haversine')
            indices = s_balltree.query_radius(coords, r=dist_thresh/earth_radius, return_distance=False)
            dist_pairs = _adj_arr_to_pairs(indices)
        else:
            s_kdtree = KDTree(coords)
            dist_pairs = s_kdtree.query_pairs(r=dist_thresh)

    neighbor_pairs = (time_pairs & dist_pairs)
  
    # Neighbor dictionary
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
