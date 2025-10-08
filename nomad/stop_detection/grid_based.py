import pandas as pd
import numpy as np
import nomad.io.base as loader
from nomad.filters import to_timestamp
from nomad.stop_detection import utils
from nomad.stop_detection.utils import _fallback_time_cols
import pdb

def grid_based_labels(data, time_thresh=np.inf, min_cluster_size=1, dur_min=0, traj_cols=None, **kwargs):
    """
    Detects stops in trajectory data based on time and each ping's location.

    Parameters
    ----------
    data : pd.DataFrame
        Input trajectory data containing temporal columns and a location column.
    time_thresh : int
        Maximum allowed time difference (in minutes) between consecutive pings within a stop. time_thresh should be greater than dur_min.
    min_cluster_size : int
        Minimum number of points required to form a stop.
    dur_min : int
        Minimum duration (in minutes) for a valid stop.
    traj_cols : dict, optional
        A dictionary defining column mappings for 'timestamp', 'datetime' or 'location_id'.
        Defaults to None.

    Returns
    -------
    pd.Series
        Integer cluster labels aligned with `data.index`. Noise gets labels of –1.
    """
    if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
        raise TypeError("Input 'data' must be a pandas DataFrame or GeoDataFrame.")
    if data.empty:
        return pd.DataFrame()
    # Decide on temporal column to use
    t_key, use_datetime = _fallback_time_cols(data.columns, traj_cols, kwargs)
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs) # load defaults

    if traj_cols['location_id'] not in data.columns:
            raise ValueError(f"Missing {traj_cols['location_id']} column in {data.columns}."
                            "pass `location_id` as keyword argument or in traj_cols."
                            )

    if traj_cols['user_id'] in data.columns:
        arr = data[traj_cols['user_id']].values
        first = arr[0]
        if any(x != first for x in arr[1:]):
            raise ValueError("grid_based cannot be run on multi-user data. Use grid_based_per_user instead.")

    ts = to_timestamp(data[traj_cols[t_key]]) if use_datetime else data[traj_cols[t_key]]
    loc = data[traj_cols['location_id']]
        
    labels = pd.Series(-1, index=data.index)
    labels.name = 'cluster'
    
    i= 0 # index to traverse data
    c = 0 # cluster label counter
    n = len(data)

    while i < n - 1:
        t_i, loc_i = ts.iloc[i], loc.iloc[i]
        
        if pd.isna(loc.iloc[i]):
            i += 1
            continue
        
        # find first index where location changes or gap exceeds threshold
        j = i + 1
        while j < n:
            gap = (ts.iloc[j] - ts.iloc[j-1]) // 60
            if pd.isna(loc.iloc[j]) or loc.iloc[j] != loc_i or gap > time_thresh:
                break
            j += 1

        if j - i >= min_cluster_size:
            if (ts.iloc[j-1] - t_i) // 60 >= dur_min:
                labels.iloc[i:j] = c
                c += 1
        i = j
    
    return labels

def grid_based(
    data,
    time_thresh=120,
    min_cluster_size=2,
    dur_min=5,
    complete_output=False,
    passthrough_cols=[],
    traj_cols=None,
    **kwargs
):
    """
    Detect stops in trajectory data using a grid/location-based segmentation, then summarize them.

    Parameters
    ----------
    data : pd.DataFrame
        Input trajectory data with temporal and location columns.
    time_thresh : int, optional
        Maximum allowed time gap (in minutes) between consecutive pings within a stop. Default is 5.
    min_cluster_size : int, optional
        Minimum number of points required to form a stop. Default is 2.
    dur_min : int, optional
        Minimum duration in minutes for a valid stop. Default is 5.
    complete_output : bool, optional
        If True, include additional stop statistics in the output.
    traj_cols : dict, optional
        Mapping for 'timestamp', 'datetime', or 'location_id' column names.
    **kwargs
        Passed through to helper functions for flexible column mapping.

    Returns
    -------
    pd.DataFrame
        One row per stop, summarizing its centroid/medoid, duration, and optionally full stats.
    """
    labels = grid_based_labels(
        data,
        time_thresh=time_thresh,
        min_cluster_size=min_cluster_size,
        dur_min=dur_min,
        traj_cols=traj_cols,
        **kwargs
    )
       
    merged = data.join(labels)
    merged = merged[merged.cluster != -1]

    stop_table = merged.groupby('cluster', as_index=False, sort=False).apply(
        lambda grp: utils.summarize_stop_grid(
            grp,
            complete_output=complete_output,
            traj_cols=traj_cols,
            keep_col_names=True,
            passthrough_cols=passthrough_cols,
            **kwargs
        ),
        include_groups=False
    )

    if complete_output:
        pass #implement diameter, centroid for location_id being an h3_cell
        
    return stop_table

def grid_based_per_user(
    data,
    time_thresh=120,
    min_cluster_size=2,
    dur_min=5,
    complete_output=False,
    passthrough_cols=[], 
    traj_cols=None,
    **kwargs
):
    """
    Run grid_based stop detection on each user separately, then concatenate results.
    Raises an error if 'user_id' is not in traj_cols or kwargs.
    """
    # Parse user_id
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if traj_cols_temp['user_id'] not in data.columns:
        raise ValueError(f"No 'user_id' column found in Index {data.columns} or specified in traj_cols or kwargs.")

    uid = traj_cols_temp['user_id']
    passthrough_cols += [uid, traj_cols_temp['date']]
    
    results = []
    for _, group in data.groupby(uid, sort=False):

        stop_table = grid_based(
            group,
            time_thresh=time_thresh,
            min_cluster_size=min_cluster_size,
            dur_min=dur_min,
            complete_output=complete_output,
            passthrough_cols=passthrough_cols,
            traj_cols=traj_cols,
            **kwargs
        )
        results.append(stop_table)
        
    return pd.concat(results, ignore_index=True)