import pandas as pd
import numpy as np
import nomad.io.base as loader
import nomad.constants as constants
from nomad.filters import to_timestamp
from nomad.stop_detection.utils import _fallback_time_cols

def grid_based_labels(data, time_thresh=np.inf, min_cluster_size=0, dur_min=0, traj_cols=None, **kwargs):
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
    # Decide on temporal column to use
    t_key, use_datetime = _fallback_time_cols(data.columns, traj_cols, kwargs)
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs) # load defaults
    if traj_cols['location_id'] not in data.columns:
            raise ValueError(f"Missing {traj_cols['location_id']} column in {data.columns}."
                            "pass `location_id` as keyword argument or in traj_cols."
                            )
    ts = to_timestamp(data[traj_cols[t_key]]) if use_datetime else data[traj_cols[t_key]]
    loc = data[traj_cols['location_id']]
        
    labels = pd.Series(-1, index=data.index)
    labels.name = 'cluster'
    
    i= 0 # numerical index to traverse data
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
