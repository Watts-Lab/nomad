import geopandas as gpd
import numpy as np
import pandas as pd
import nomad.io.base as loader
from nomad.filters import to_timestamp
from nomad.stop_detection.postprocessing import merge_stops
from nomad.stop_detection import utils
from nomad.stop_detection.utils import _haversine_distance
from nomad.visit_attribution.visit_attribution import detect_locations


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
        Cluster labels.
    """
    if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
        raise TypeError("Input 'data' must be a pandas DataFrame or GeoDataFrame.")
    
    # Get column mappings
    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(
        data.columns, traj_cols, kwargs
    )
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    
    # Validate spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    if data.empty:
        return utils._get_empty_aux_df()
    if method not in {"sliding", "centroid"}:
        raise ValueError(f"Unknown method: {method}")

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

            if use_lon_lat:
                dist = _haversine_distance(anchor_coords, coords[j], radians=False)
            else:
                dist = np.linalg.norm(coords[j] - anchor_coords)
            
            # Check if moved beyond distance threshold
            if dist > delta_roam:
                break
            
            # Update centroid if using centroid method
            if method == 'centroid':
                anchor_coords = ((j-i) * anchor_coords + coords[j]) / (j - i + 1)
            
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


def detect_stops(
    data,
    delta_roam=100,
    dt_max=15.0,
    dur_min=5.0,
    method='sliding',
    complete_output=False,
    passthrough_cols=None,
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
    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
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
    labels = detect_stops_labels(
        data=data,
        delta_roam=delta_roam,
        dt_max=dt_max,
        dur_min=dur_min,
        method=method,
        traj_cols=traj_cols,
        **kwargs
    )
    return utils.summarize_stops(
        data,
        labels,
        complete_output=complete_output,
        passthrough_cols=passthrough_cols,
        keep_col_names=keep_col_names,
        traj_cols=traj_cols,
        **kwargs,
    )


def detect_stops_per_user(
    data,
    delta_roam=100,
    dt_max=15.0,
    dur_min=5.0,
    method='sliding',
    complete_output=False,
    passthrough_cols=None,
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
    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("detect_stops_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']
    
    pt_cols = passthrough_cols if uid in passthrough_cols else passthrough_cols + [uid]
    
    grouped = data.groupby(uid, sort=False, as_index=False)
    results = utils.applyParallel(
        grouped,
        detect_stops,
        {
            "delta_roam": delta_roam,
            "dt_max": dt_max,
            "dur_min": dur_min,
            "method": method,
            "complete_output": complete_output,
            "passthrough_cols": pt_cols,
            "keep_col_names": keep_col_names,
            "traj_cols": traj_cols,
            **kwargs,
        },
        reset_index=True,
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

    grouped = data.groupby(uid, sort=False)
    results = utils.applyParallel(
        grouped,
        detect_stops_labels,
        {
            "delta_roam": delta_roam,
            "dt_max": dt_max,
            "dur_min": dur_min,
            "method": method,
            "traj_cols": traj_cols,
            **kwargs,
        },
        n_jobs=n_jobs,
        print_progress=print_progress,
    )

    return pd.concat(results).reindex(data.index)
########        Lachesis          ########
##########################################

def lachesis_labels(data, dt_max, delta_roam, dur_min=5, traj_cols=None, return_anchors=False, **kwargs):
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
    return_anchors : bool, default False
        Return the earliest endpoint of each accepted prefix's maximum-diameter
        pair as ``anchor_time``.
    traj_cols : dict, optional
        Mapping for 'x', 'y', 'longitude', 'latitude', 'timestamp', or 'datetime'.
    **kwargs
        Passed along to the column‐detection helper.

    Returns
    -------
    pd.Series or pd.DataFrame
        Labels, or labels with aligned diameter-witness ``anchor_time``.
    """
    if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
         raise TypeError("Input 'data' must be a pandas DataFrame or GeoDataFrame.")

    t_key, coord_key1, coord_key2, use_datetime, use_lon_lat = utils._fallback_st_cols(data.columns, traj_cols, kwargs)        
    traj_cols = loader._parse_traj_cols(data.columns, traj_cols, kwargs)

    # Tests to check for spatial and temporal columns
    loader._has_spatial_cols(data.columns, traj_cols)
    loader._has_time_cols(data.columns, traj_cols)

    if data.empty:
        return utils._get_empty_aux_df(
            data[traj_cols[t_key]], return_anchors=return_anchors
        )

    metric = 'haversine' if use_lon_lat else 'euclidean'    
    coords = data[[traj_cols[coord_key1], traj_cols[coord_key2]]].to_numpy(dtype='float64')
    
    # Parse if necessary
    time_series = to_timestamp(data[traj_cols[t_key]]) if use_datetime else data[traj_cols[t_key]]

    i = 0
    n = len(data)
    
    # all cluster labels initialized as noise
    labels = np.full(n, -1, dtype=int)
    if return_anchors:
        anchor_times = pd.Series(
            index=data.index,
            dtype=data[traj_cols[t_key]].dtype if use_datetime else 'Int64',
        )
    cluster_id = 0
    while i < n - 1:
        t_i = time_series.iloc[i]
        j_star = next((j for j in range(i, n) if (time_series.iloc[j] - t_i) >= dur_min * 60), -1)
        if j_star == -1:
            break

        d_start = utils._diameter(coords[i:j_star + 1], metric=metric)
        time_diffs = np.diff(time_series.iloc[i:j_star + 1].values)
        if (time_diffs > dt_max * 60).any() or d_start > delta_roam:
            i += 1
            continue

        j_final = j_star
        for j in range(j_star + 1, n):
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
            if return_anchors:
                diameter, witness = utils._diameter(
                    coords[i:i + 1], metric=metric, witness=True
                )
                for anchor_index in range(i + 1, j_final + 1):
                    diameter, witness = utils._update_diameter(
                        coords[anchor_index],
                        coords[i:anchor_index],
                        diameter,
                        metric=metric,
                        witness=True,
                        witness_index=witness,
                    )
                    anchor_times.iloc[anchor_index] = data[traj_cols[t_key]].iloc[i + witness]
            cluster_id += 1

        i = j_final + 1

    result = pd.Series(labels, index=data.index, name='cluster')
    if return_anchors:
        return pd.DataFrame({"cluster": result, "anchor_time": anchor_times})
    return result

def lachesis(
    data,
    delta_roam,
    dt_max = 60,
    dur_min=5,
    complete_output=False,
    passthrough_cols=None,
    keep_col_names=True,
    postprocessing=None,
    postprocessing_kwargs=None,
    merge_kwargs=None,
    traj_cols=None,
    **kwargs
):
    """
    Sequential stop detection with diameter stopping criterion

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
    passthrough_cols : list, optional
        Columns to retain (and summarize/propagate) per stop.
    postprocessing : {None, 'none', 'dbscan', 'infomap'}, optional
        Destination-detection method applied after sequential stop detection.
    postprocessing_kwargs : dict, optional
        Arguments passed to DBSCAN destination detection.
    merge_kwargs : dict, optional
        Arguments passed to visit merging after destination detection.

    Returns
    -------
    pd.Series
        One integer label per row, -1 for non‐stop points, 0..K for stops.

    Raises
    ------
    ValueError if multiple users found; use lachesis_per_user instead.
    """
    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
    traj_cols_temp = loader._parse_traj_cols(
        data.columns, traj_cols, kwargs, warn=False
    )
    if 'user_id' in traj_cols_temp and traj_cols_temp['user_id'] in data.columns:
        uid_col = data[traj_cols_temp['user_id']]
        arr = uid_col.values
        if len(arr) > 0:
            first = arr[0]
            if any(x != first for x in arr[1:]):
                raise ValueError("Multi-user data? Use lachesis_per_user instead.")
            if traj_cols_temp['user_id'] not in passthrough_cols:
                passthrough_cols = passthrough_cols + [traj_cols_temp['user_id']]
    labels = lachesis_labels(
        data=data,
        dur_min=dur_min,
        dt_max=dt_max,
        delta_roam=delta_roam,
        traj_cols=traj_cols,
        **kwargs
    )
    stop_table = utils.summarize_stops(
        data,
        labels,
        complete_output=complete_output,
        passthrough_cols=passthrough_cols,
        keep_col_names=keep_col_names,
        traj_cols=traj_cols,
        **kwargs,
    )
    return _postprocess_lachesis_stops(
        stop_table,
        postprocessing=postprocessing,
        postprocessing_kwargs=postprocessing_kwargs,
        merge_kwargs=merge_kwargs,
        traj_cols=traj_cols,
        **kwargs
    )


def _postprocess_lachesis_stops(
    stops,
    postprocessing=None,
    postprocessing_kwargs=None,
    merge_kwargs=None,
    traj_cols=None,
    **kwargs
):
    """Apply destination detection and visit merging to Lachesis stops."""
    if postprocessing in (None, 'none'):
        return stops
    if postprocessing == 'infomap':
        raise NotImplementedError("Lachesis postprocessing method 'infomap' is not implemented")
    if postprocessing != 'dbscan':
        raise ValueError("postprocessing must be one of: None, 'none', 'dbscan', 'infomap'")

    traj_cols_temp = loader._parse_traj_cols(
        stops.columns, traj_cols, kwargs, warn=False
    )
    location_col = traj_cols_temp['location_id']
    if stops.empty:
        stops = stops.copy()
        stops[location_col] = pd.Series(index=stops.index, dtype='Int64')
        return stops

    # Recognize recurring destinations, then combine interrupted visits to them.
    location_options = dict(postprocessing_kwargs or {})
    location_options['return_locations'] = True
    location_ids, locations = detect_locations(
        stops,
        traj_cols=traj_cols,
        **kwargs,
        **location_options
    )
    labeled_stops = stops.copy()
    labeled_stops[location_col] = location_ids.to_numpy()
    merge_options = dict(merge_kwargs or {})
    merge_options.setdefault('location_col', location_col)
    visits = merge_stops(
        labeled_stops,
        traj_cols=traj_cols,
        **kwargs,
        **merge_options
    )

    # Report each visit at its recurring destination's center.
    coord_key1, coord_key2, _ = loader._fallback_spatial_cols(
        labeled_stops.columns, traj_cols_temp, kwargs
    )
    centers = locations.set_index(location_col).center
    visits[traj_cols_temp[coord_key1]] = visits[location_col].map(centers.x)
    visits[traj_cols_temp[coord_key2]] = visits[location_col].map(centers.y)
    return visits

def lachesis_per_user(
    data,
    dt_max,
    delta_roam,
    dur_min=5,
    complete_output=False,
    passthrough_cols=None,
    postprocessing=None,
    postprocessing_kwargs=None,
    merge_kwargs=None,
    traj_cols=None,
    n_jobs=1,
    print_progress=False,
    **kwargs
):
    """
    Run lachesis on each user separately, then concatenate results.
    
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
    complete_output : bool, default False
        If True, include additional summary statistics in output.
    passthrough_cols : list, optional
        Columns to retain (and summarize/propagate) per stop.
    postprocessing : {None, 'none', 'dbscan', 'infomap'}, optional
        Destination-detection method applied after all users' stops are detected.
    postprocessing_kwargs : dict, optional
        Arguments passed to DBSCAN destination detection.
    merge_kwargs : dict, optional
        Arguments passed to visit merging after destination detection.
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
    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
    traj_cols_temp = loader._parse_traj_cols(
        data.columns, traj_cols, kwargs, warn=False
    )
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("lachesis_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']
    
    pt_cols = passthrough_cols if uid in passthrough_cols else passthrough_cols + [uid]
    
    grouped = data.groupby(uid, sort=False)
    results = utils.applyParallel(
        grouped,
        lachesis,
        {
            "dt_max": dt_max,
            "delta_roam": delta_roam,
            "dur_min": dur_min,
            "complete_output": complete_output,
            "passthrough_cols": pt_cols,
            "postprocessing": None,
            "traj_cols": traj_cols,
            **kwargs,
        },
        reset_index=True,
        n_jobs=n_jobs,
        print_progress=print_progress
    )
    
    stops = pd.concat(results, ignore_index=True)
    return _postprocess_lachesis_stops(
        stops,
        postprocessing=postprocessing,
        postprocessing_kwargs=postprocessing_kwargs,
        merge_kwargs=merge_kwargs,
        traj_cols=traj_cols,
        **kwargs
    )


def lachesis_labels_per_user(
    data,
    dt_max,
    delta_roam,
    dur_min=5,
    traj_cols=None,
    return_anchors=False,
    n_jobs=1,
    print_progress=False,
    **kwargs
):
    """
    Run lachesis_labels on each user separately and concatenate labels.

    Raises if 'user_id' not in traj_cols or missing from data.
    """
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("lachesis_labels_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']

    grouped = data.groupby(uid, sort=False)
    results = utils.applyParallel(
        grouped,
        lachesis_labels,
        {
            "dt_max": dt_max,
            "delta_roam": delta_roam,
            "dur_min": dur_min,
            "return_anchors": return_anchors,
            "traj_cols": traj_cols,
            **kwargs,
        },
        n_jobs=n_jobs,
        print_progress=print_progress,
    )

    return pd.concat(results).reindex(data.index)
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
        return utils._get_empty_aux_df()
    # Decide on temporal column to use
    t_key, use_datetime = loader._fallback_time_cols_dt(data.columns, traj_cols, kwargs)
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

    while i < n:
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
    passthrough_cols=None,
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
    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
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

    empty_stops = utils._get_empty_stop_df(
        data,
        complete_output,
        passthrough_cols,
        traj_cols,
        keep_col_names=True,
        is_grid_based=True,
        **kwargs,
    )

    if merged.empty:
        return empty_stops

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
    ).reset_index(drop=True)

    if complete_output:
        pass #implement diameter, centroid for location_id being an h3_cell
        
    return utils._cast_to_stop_schema(stop_table, empty_stops)

def grid_based_per_user(
    data,
    time_thresh=120,
    min_cluster_size=2,
    dur_min=5,
    complete_output=False,
    passthrough_cols=None,
    traj_cols=None,
    n_jobs=1,
    print_progress=False,
    **kwargs
):
    """
    Run grid_based stop detection on each user separately, then concatenate results.
    Raises an error if 'user_id' is not in traj_cols or kwargs.
    """
    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
    # Parse user_id
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if traj_cols_temp['user_id'] not in data.columns:
        raise ValueError(f"No 'user_id' column found in Index {data.columns} or specified in traj_cols or kwargs.")

    uid = traj_cols_temp['user_id']
    pt_cols = passthrough_cols.copy()
    for col in [uid, traj_cols_temp['date']]:
        if col not in pt_cols:
            pt_cols.append(col)
    
    grouped = data.groupby(uid, sort=False, as_index=False)
    results = utils.applyParallel(
        grouped,
        grid_based,
        {
            "time_thresh": time_thresh,
            "min_cluster_size": min_cluster_size,
            "dur_min": dur_min,
            "complete_output": complete_output,
            "passthrough_cols": pt_cols,
            "traj_cols": traj_cols,
            **kwargs,
        },
        reset_index=True,
        n_jobs=n_jobs,
        print_progress=print_progress,
    )
        
    return pd.concat(results, ignore_index=True)


def grid_based_labels_per_user(
    data,
    time_thresh=np.inf,
    min_cluster_size=1,
    dur_min=0,
    traj_cols=None,
    n_jobs=1,
    print_progress=False,
    **kwargs
):
    """
    Run grid_based_labels on each user separately and concatenate labels.

    Raises if 'user_id' not in traj_cols or missing from data.
    """
    traj_cols_temp = loader._parse_traj_cols(data.columns, traj_cols, kwargs)
    if 'user_id' not in traj_cols_temp or traj_cols_temp['user_id'] not in data.columns:
        raise ValueError("grid_based_labels_per_user requires a 'user_id' column specified in traj_cols or kwargs.")
    uid = traj_cols_temp['user_id']

    grouped = data.groupby(uid, sort=False)
    results = utils.applyParallel(
        grouped,
        grid_based_labels,
        {
            "time_thresh": time_thresh,
            "min_cluster_size": min_cluster_size,
            "dur_min": dur_min,
            "traj_cols": traj_cols,
            **kwargs,
        },
        n_jobs=n_jobs,
        print_progress=print_progress,
    )

    return pd.concat(results).reindex(data.index)


__all__ = [
    "detect_stops_labels",
    "detect_stops",
    "detect_stops_per_user",
    "detect_stops_labels_per_user",
    "lachesis_labels",
    "lachesis",
    "lachesis_per_user",
    "lachesis_labels_per_user",
    "grid_based_labels",
    "grid_based",
    "grid_based_per_user",
    "grid_based_labels_per_user",
]
