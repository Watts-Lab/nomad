import pandas as pd
import numpy as np
import geopandas as gpd
import nomad.io.base as loader
from nomad.stop_detection import utils
from nomad.filters import to_timestamp
from nomad.stop_detection.utils import _haversine_distance

def detect_stops_labels(
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
        Method to use ('sliding' or 'centroid') for the anchor point of the active stop
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
        return pd.Series(dtype='int64', name='cluster')
    
    # Get column mappings
    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(
        data.columns, traj_cols, kwargs
    )
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    
    # Validate spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    # Extract coordinates and time
    coords = data[[traj_cols[coord_key1], traj_cols[coord_key2]]].to_numpy(dtype='float64')
    times = to_timestamp(data[traj_cols[t_key]]) if use_datetime else data[traj_cols[t_key]]
    
    # Initialize all labels as noise (-1)
    n = len(data)
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    
    i = 0
    while i < n:
        j = i + 1
        anchor_coords = coords[i]
        start_time = times.iloc[i]
        
        # Slide window forward
        while j < n:
            # Check for temporal gap
            time_gap = (times.iloc[j] - times.iloc[j-1]) / 60  # Convert to minutes
            if time_gap > dt_max:
                break

            if method == "sliding" or method == "centroid":
                if use_lon_lat:
                    dist = _haversine_distance(anchor_coords, coords[j], radians=False)
                else:
                    dist = np.linalg.norm(coords[j] - anchor_coords)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Check if moved beyond distance threshold
            if dist > delta_roam:
                break
            
            # Update centroid if using centroid method
            if method == 'centroid':
                anchor_coords = ((j-i) * anchor_coords + coords[j]) / (j - i + 1)
            else:
                pass
            
            j += 1
        
        # Check if we have a valid stop (enough time spent)
        time_spent = (times.iloc[j-1] - start_time) / 60  # Convert to minutes
        
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
    return utils.applyParallel(
        groups,
        func,
        n_jobs=n_jobs,
        print_progress=print_progress,
        **kwargs,
    )


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
            if traj_cols_temp['user_id'] not in passthrough_cols:
                passthrough_cols = passthrough_cols + [traj_cols_temp['user_id']]
    else:
        uid_col = None

    labels = detect_stops_labels(
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
        return utils._get_empty_stop_df(
            data.columns,
            complete_output,
            passthrough_cols,
            traj_cols,
            keep_col_names=keep_col_names,
            is_grid_based=False,
            **kwargs,
        )

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
    ).reset_index(drop=True)

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
    n_jobs=1,
    print_progress=False,
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
    n_jobs : int, default 1
        Number of parallel jobs to use. 1 means sequential processing.
    print_progress : bool, default False
        Whether to show progress bar during processing.
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
    
    pt_cols = passthrough_cols if uid in passthrough_cols else passthrough_cols + [uid]
    
    def process_user_group(group):
        """Helper function to process a single user group."""
        return detect_stops(
            group[1].reset_index(drop=True),
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
    
    # Use applyParallel to process groups in parallel
    grouped = data.groupby(uid, sort=False, as_index=False)
    results = applyParallel(
        grouped,
        process_user_group,
        n_jobs=n_jobs,
        print_progress=print_progress
    )
    
    return pd.concat(results, ignore_index=True)


def detect_stops_labels_per_user(
    data,
    delta_roam=100,
    dt_max=15.0,
    dur_min=5.0,
    method='sliding',
    traj_cols=None,
    n_jobs=1,
    print_progress=False,
    **kwargs
):
    """
    Run detect_stops_labels on each user separately and concatenate labels.

    Raises if 'user_id' not in traj_cols or missing from data.
    """
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("detect_stops_labels_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']

    def process_user_group(group):
        return detect_stops_labels(
            data=group[1],
            delta_roam=delta_roam,
            dt_max=dt_max,
            dur_min=dur_min,
            method=method,
            traj_cols=traj_cols,
            **kwargs,
        )

    grouped = data.groupby(uid, sort=False)
    results = applyParallel(
        grouped,
        process_user_group,
        n_jobs=n_jobs,
        print_progress=print_progress,
    )

    return pd.concat(results).reindex(data.index)