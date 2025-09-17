import pandas as pd
import numpy as np
import heapq
from collections import defaultdict
import warnings
import nomad.io.base as loader
from nomad.stop_detection import utils
from nomad.filters import to_timestamp

def _build_neighbor_dict(i_idx, j_idx, mask, times):
    """
    Build a neighbor dictionary from index arrays and a boolean mask.

    Parameters
    ----------
    i_idx, j_idx : array-like
        Arrays of indices representing pairs.
    mask : array-like
        Boolean array indicating which pairs are valid.
    times : array-like
        Array of timestamps.

    Returns
    -------
    neighbor_dict : defaultdict(set)
        Dictionary mapping each timestamp to its neighbors.
    """
    neighbor_dict = defaultdict(set)
    for i, j in zip(i_idx[mask], j_idx[mask]):
        neighbor_dict[times[i]].add(times[j])
        neighbor_dict[times[j]].add(times[i])
    return neighbor_dict

def _cache_neighbors(times, time_thresh, use_datetime, coords=None, dist_thresh=None, use_lon_lat=False):
    """
    Find neighboring timestamps within a time threshold, and optionally filter by spatial proximity.

    Parameters
    ----------
    times : pd.Series or np.ndarray
        Array of timestamps or datetime objects.
    time_thresh : float
        Time threshold (in minutes) for considering neighbors.
    use_datetime : bool
        If True, convert times to timestamps using `to_timestamp`.
    coords : pd.DataFrame or np.ndarray, optional
        Array of coordinates (e.g., latitude/longitude or x/y) for spatial filtering.
    dist_thresh : float, optional
        Distance threshold for spatial neighbors. Required if `coords` is provided.
    use_lon_lat : bool, default=False
        If True, coordinates are treated as longitude/latitude and haversine distance is used.

    Returns
    -------
    neighbor_dict : defaultdict(set)
        Dictionary mapping each timestamp to its neighboring timestamps (if `coords` is provided).
    OR
    time_pairs : list of tuple
        List of (timestamp1, timestamp2) pairs within the time threshold (if `coords` is None).
    times : np.ndarray
        Array of processed timestamps.

    Notes
    -----
    - If `coords` and `dist_thresh` are provided, only pairs within both time and distance thresholds are considered neighbors.
    - If only time filtering is used, all pairs within the time threshold are returned.
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
    
    # Only for TA-DBSCAN
    if coords is not None:
        coords = np.radians(coords) if use_lon_lat else coords.values

        # Distance calculation
        if use_lon_lat:
            distances = np.array([utils._haversine_distance(coords[i], coords[j]) for i, j in zip(i_idx, j_idx)])
        else:
            distances_sq = (coords[i_idx, 0] - coords[j_idx, 0])**2 + (coords[i_idx, 1] - coords[j_idx, 1])**2
            distances = np.sqrt(distances_sq)

        # Filter by distance threshold
        valid = distances < dist_thresh
        
        # Build neighbor graph
        neighbor_dict = _build_neighbor_dict(i_idx, j_idx, valid, times)

        return neighbor_dict

    # Return a list of (timestamp1, timestamp2) tuples
    time_pairs = [(times[i], times[j]) for i, j in zip(i_idx, j_idx)]
    
    return time_pairs, times