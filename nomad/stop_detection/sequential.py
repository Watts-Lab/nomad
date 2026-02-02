import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm
import nomad.io.base as loader
from nomad.stop_detection import utils
from nomad.filters import to_timestamp


def haversine_dist(lat1, lon1, lat2, lon2):
    # remove this and use what's in utils.py 
    """
    Calculate haversine distance between two points in meters.

    Parameters
    ----------
    lat1, lon1 : float or array-like
        Latitude and longitude of first point(s)
    lat2, lon2 : float or array-like
        Latitude and longitude of second point(s)

    Returns
    -------
    float or array-like
        Distance in meters
    """
    R = 6371000  # Earth radius in meters

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c


def detect_stop_labels(
    data,
    delta_roam=100,
    dt_max=15.0,
    dur_min=5.0,
    method='sliding',
    traj_cols=None,
    **kwargs
):
    """
    Scan a trajectory and assign each point to a stop cluster index or -1 for noise.
    
    Uses a sliding window approach where points are grouped into stops based on:
    - Spatial constraint: all points within delta_roam of first point in window
    - Temporal constraint: no gaps > dt_max between consecutive points
    - Duration constraint: total duration >= dur_min
    
    Parameters
    ----------
    data : pd.DataFrame or gpd.GeoDataFrame
        Input trajectory with spatial and temporal columns
    delta_roam : float, default 100
        Maximum distance threshold in meters (for haversine) or map units (for euclidean)
    dt_max : float, default 15.0
        Maximum allowed gap in minutes between consecutive points in a stop
    dur_min : float, default 5.0
        Minimum duration in minutes for a valid stop
    method : str, default 'sliding'
        Method to use ('sliding' currently supported)
    traj_cols : dict, optional
        Mapping for 'x', 'y', 'longitude', 'latitude', 'timestamp', or 'datetime'
    **kwargs
        Passed along to column detection helper
        
    Returns
    -------
    pd.Series
        One integer label per row, -1 for non-stop points, 0..K for stops
    """
    if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
        raise TypeError("Input 'data' must be a pandas DataFrame or GeoDataFrame.")
    
    if data.empty:
        return pd.Series([], dtype=int, name='cluster')
    
    # Get column mappings
    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(
        data.columns, traj_cols, kwargs
    )
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    
    # Validate spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)
    
    # Determine distance metric
    metric = 'haversine' if use_lon_lat else 'euclidean'
    
    # Extract coordinates and time
    coords = data[[traj_cols[coord_key1], traj_cols[coord_key2]]].to_numpy(dtype='float64')
    time_series = to_timestamp(data[traj_cols[t_key]]) if use_datetime else data[traj_cols[t_key]]
    
    # Initialize all labels as noise (-1)
    n = len(data)
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    
    i = 0
    while i < n:
        j = i + 1
        
        # Get starting point info
        if use_lon_lat:
            start_lat, start_lon = coords[i, 0], coords[i, 1]
        start_time = time_series.iloc[i]
        
        # Slide window forward
        while j < n:
            # Check for temporal gap
            time_gap = (time_series.iloc[j] - time_series.iloc[j-1]) / 60  # Convert to minutes
            if time_gap > dt_max:
                break
            
            # Calculate distance from first point (anchor)
            if method == 'sliding':
                if use_lon_lat:
                    curr_lat, curr_lon = coords[j, 0], coords[j, 1]
                    dist = haversine_dist(start_lat, start_lon, curr_lat, curr_lon)
                else:
                    dist = np.linalg.norm(coords[j] - coords[i])
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Check if moved beyond distance threshold
            if dist > delta_roam:
                break
            
            j += 1
        
        # Check if we have a valid stop (enough time spent)
        time_spent = (time_series.iloc[j-1] - start_time) / 60  # Convert to minutes
        
        if time_spent >= dur_min:
            # Assign cluster label to all points in this stop
            labels[i:j] = cluster_id
            cluster_id += 1
            # Move to the point that broke the stop
            i = j
        else:
            # Not enough time spent, move to next point
            i += 1
    
    return pd.Series(labels, index=data.index, name='cluster')


def applyParallel(groups, func, n_jobs=1, print_progress=False, **kwargs):
    """
    Apply function to groups in parallel.

    Parameters
    ----------
    groups : DataFrameGroupBy
        Grouped dataframe
    func : callable
        Function to apply to each group
    n_jobs : int
        Number of parallel jobs
    print_progress : bool
        Whether to show progress bar
    **kwargs
        Additional arguments to pass to func

    Returns
    -------
    list
        List of results from applying func to each group
    """
    if n_jobs == 1:
        # Sequential processing
        if print_progress:
            results = [func(group, **kwargs) for group in tqdm(groups, desc="Processing users")]
        else:
            results = [func(group, **kwargs) for group in groups]
    else:
        # Parallel processing
        group_list = list(groups)
        if print_progress:
            results = Parallel(n_jobs=n_jobs)(
                delayed(func)(group, **kwargs) for group in tqdm(group_list, desc="Processing users")
            )
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(func)(group, **kwargs) for group in group_list
            )

    return results


def detect_stops(
    data,
    delta_roam=100,
    dt_max=15.0,
    dur_min=5.0,
    method='sliding',
    complete_output=False,
    passthrough_cols=[],
    keep_col_names=True,
    traj_cols=None,
    **kwargs
):
    """
    Sequential stop detection using sliding window approach.
    
    Analogous to lachesis function but uses sliding window method.

    Parameters
    ----------
    data : pd.DataFrame or GeoDataFrame
        Input trajectory with spatial and temporal columns.
    delta_roam : float, default 100
        Maximum distance threshold in meters (for haversine) or map units (for euclidean).
    dt_max : float, default 15.0
        Maximum allowed gap in minutes between consecutive points in a stop.
    dur_min : float, default 5.0
        Minimum duration in minutes for a valid stop.
    method : str, default 'sliding'
        Method to use ('sliding' currently supported).
    complete_output : bool, default False
        If True, include additional summary statistics in output.
    passthrough_cols : list, optional
        Columns to retain (and summarize/propagate) per stop.
    keep_col_names : bool, default True
        Whether to keep original column names in output.
    traj_cols : dict, optional
        Mapping for 'x', 'y', 'longitude', 'latitude', 'timestamp', or 'datetime'.
    **kwargs
        Passed along to column detection helper.

    Returns
    -------
    pd.DataFrame
        Stop table with one row per detected stop.

    Raises
    ------
    ValueError if multiple users found; use detect_stops_per_user instead.
    """
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' in traj_cols_temp and traj_cols_temp['user_id'] in data.columns:
        uid_col = data[traj_cols_temp['user_id']]
        arr = uid_col.values
        if len(arr) > 0:
            first = arr[0]
            if any(x != first for x in arr[1:]):
                raise ValueError("Multi-user data? Use detect_stops_per_user instead.")
            passthrough_cols = passthrough_cols + [traj_cols_temp['user_id']]
    else:
        uid_col = None

    labels = detect_stop_labels(
        data=data,
        delta_roam=delta_roam,
        dt_max=dt_max,
        dur_min=dur_min,
        method=method,
        traj_cols=traj_cols,
        **kwargs
    )
    merged = data.join(labels)
    merged = merged[merged.cluster != -1]

    if merged.empty:
        # Get column names by calling summarize function on dummy data
        cols = utils._get_empty_stop_columns(
            data.columns, complete_output, passthrough_cols, traj_cols, 
            keep_col_names=keep_col_names, is_grid_based=False, **kwargs
        )
        return pd.DataFrame(columns=cols, dtype=object)

    stop_table = merged.groupby('cluster', as_index=False, sort=False).apply(
        lambda grp: utils.summarize_stop(
            grp,
            complete_output=complete_output,
            traj_cols=traj_cols,
            keep_col_names=keep_col_names,
            passthrough_cols=passthrough_cols,
            **kwargs
        ),
        include_groups=False
    )

    return stop_table


def detect_stops_per_user(
    data,
    delta_roam=100,
    dt_max=15.0,
    dur_min=5.0,
    method='sliding',
    complete_output=False,
    passthrough_cols=[],
    keep_col_names=True,
    traj_cols=None,
    **kwargs
):
    """
    Run detect_stops on each user separately, then concatenate results.
    
    Parameters
    ----------
    data : pd.DataFrame or GeoDataFrame
        Input trajectory with spatial and temporal columns.
    delta_roam : float, default 100
        Maximum distance threshold in meters (for haversine) or map units (for euclidean).
    dt_max : float, default 15.0
        Maximum allowed gap in minutes between consecutive points in a stop.
    dur_min : float, default 5.0
        Minimum duration in minutes for a valid stop.
    method : str, default 'sliding'
        Method to use ('sliding' currently supported).
    complete_output : bool, default False
        If True, include additional summary statistics in output.
    passthrough_cols : list, optional
        Columns to retain (and summarize/propagate) per stop.
    keep_col_names : bool, default True
        Whether to keep original column names in output.
    traj_cols : dict, optional
        Mapping for 'x', 'y', 'longitude', 'latitude', 'timestamp', or 'datetime'.
    **kwargs
        Passed along to column detection helper.

    Returns
    -------
    pd.DataFrame
        Concatenated stop table with stops from all users.
        
    Raises
    ------
    ValueError if 'user_id' not in traj_cols or missing from data.
    """
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("detect_stops_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']
    
    pt_cols = passthrough_cols + [uid]
    
    results = [
        detect_stops(
            group.reset_index(drop=True),
            delta_roam=delta_roam,
            dt_max=dt_max,
            dur_min=dur_min,
            method=method,
            complete_output=complete_output,
            passthrough_cols=pt_cols,
            keep_col_names=keep_col_names,
            traj_cols=traj_cols,
            **kwargs
        )
        for _, group in data.groupby(uid, sort=False)
    ]
    return pd.concat(results, ignore_index=True)