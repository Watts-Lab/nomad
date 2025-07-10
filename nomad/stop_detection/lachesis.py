import pandas as pd
from scipy.spatial.distance import pdist, cdist
import numpy as np
import math
import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import warnings
import geopandas as gpd
import nomad.io.base as loader
import nomad.constants as constants
from nomad.stop_detection import utils
from nomad.filters import to_timestamp

##########################################
########        Lachesis          ########
##########################################

def _lachesis_labels(data, dt_max, delta_roam, dur_min=5, traj_cols=None, **kwargs):
    """
    Scan a trajectory and assign each ping to a stop‐cluster index or -1 for noise.

    Parameters
    ----------
    data : pd.DataFrame or GeoDataFrame
        Input trajectory with spatial and temporal columns.
    dt_max : int
        Maximum allowed gap in minutes between consecutive pings in a stop.
    delta_roam : float
        Maximum spatial diameter for a stop.
    dur_min : int
        Minimum duration in minutes for a valid stop.
    traj_cols : dict, optional
        Mapping for 'x', 'y', 'longitude', 'latitude', 'timestamp', or 'datetime'.
    **kwargs
        Passed along to the column‐detection helper.

    Returns
    -------
    pd.Series
        One integer label per row, -1 for non‐stop points, 0..K for stops.
    """
    if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
         raise TypeError("Input 'data' must be a pandas DataFrame or GeoDataFrame.")
    if data.empty:
        return pd.DataFrame()

    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(data.columns, traj_cols, kwargs)        
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    metric = 'haversine' if use_lon_lat else 'euclidean'    
    coords = data[[traj_cols[coord_key1], traj_cols[coord_key2]]].to_numpy(dtype='float64')
    
    # Parse if necessary
    time_series = to_timestamp(data[traj_cols[t_key]]) if use_datetime else data[traj_cols[t_key]]
  
    stops = []
    i = 0
    n = len(data)
    
    # all cluster labels initialized as noise
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    while i < n - 1:
        t_i = time_series.iloc[i]
        j_star = next((j for j in range(i, n) if (time_series.iloc[j] - t_i) >= dur_min * 60), -1)
        if j_star == -1:
            break

        d_start = utils._diameter(coords[i:j_star + 1], metric=metric)
        time_diffs = np.diff(time_series.iloc[i:j_star + 1].values)
        if (time_diffs >= dt_max * 60).any() or d_start > delta_roam:
            i += 1
            continue

        j_final = j_star
        for j in range(j_star, n):
            d_update = utils._update_diameter(coords[j], coords[i:j], d_start, metric=metric)
            cc_diff = time_series.iloc[j] - time_series.iloc[j - 1]
            if d_update > delta_roam or cc_diff > dt_max * 60:
                j_final = j - 1
                break
            d_start = d_update
        else:
            j_final = n - 1

        duration = (time_series.iloc[j_final] - time_series.iloc[i]) // 60
        if duration >= dur_min:
            labels[i : j_final + 1] = cluster_id
            cluster_id += 1

        i = j_final + 1

    return pd.Series(labels, index=data.index, name='cluster')

def lachesis(data, dt_max, delta_roam, dur_min=5, complete_output=False, traj_cols=None, **kwargs):
    """
    Detect stops with the sequential Lachesis algorithm and then summarize them.

    Parameters
    ----------
    data : pd.DataFrame or GeoDataFrame
        Trajectory data.
    dur_min : int
        Minimum duration in minutes for a stop.
    dt_max : int
        Maximum time gap in minutes allowed within a stop.
    delta_roam : float
        Maximum spatial diameter for a stop.
    traj_cols : dict, optional
        Column mappings.
    complete_output : bool
        If True, include diameter, n_pings, max_gap in the summary.
    **kwargs
        Passed through to both the labeler and summarizer.

    Returns
    -------
    pd.DataFrame
        One row per stop, summarizing its medoid, duration, and optionally full stats.
    """
    labels = _lachesis_labels(
        data=data,
        dur_min=dur_min,
        dt_max=dt_max,
        delta_roam=delta_roam,
        traj_cols=traj_cols,
        **kwargs
    )

    merged = data.join(labels)
    merged = merged[merged.cluster != -1]

    stop_table = merged.groupby('cluster', as_index=False).apply(
        lambda grp: utils.summarize_stop(
            grp,
            complete_output=complete_output,
            traj_cols=traj_cols,
            keep_col_names=False,
            **kwargs
        ),
        include_groups=False
    )

    return stop_table