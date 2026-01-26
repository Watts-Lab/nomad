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


def generate_staypoints(
    positionfixes,
    method="sliding",
    distance_metric="haversine",
    dist_threshold=100,
    time_threshold=5.0,
    gap_threshold=15.0,
    print_progress=False,
    exclude_duplicate_pfs=True,
    n_jobs=1,
):
    """
    Generate staypoints from positionfixes using sliding window approach.

    Parameters
    ----------
    positionfixes : pd.DataFrame or gpd.GeoDataFrame
        Position fixes with columns: user_id, timestamp, geometry

    method : {'sliding'}
        Method to create staypoints. 'sliding' applies a sliding window over the data.

    distance_metric : {'haversine', 'euclidean'}
        The distance metric used by the applied method.

    dist_threshold : float, default 100
        The distance threshold for the 'sliding' method, i.e., how far someone has to travel to
        generate a new staypoint. Units depend on distance_metric. If 'haversine': meters.

    time_threshold : float, default 5.0 (minutes)
        The time threshold for the 'sliding' method in minutes.

    gap_threshold : float, default 15.0 (minutes)
        Maximum temporal gap between consecutive points. Consecutive points with
        temporal gaps larger than 'gap_threshold' will be excluded from staypoints.

    print_progress: boolean, default False
        Show per-user progress if set to True.

    exclude_duplicate_pfs: boolean, default True
        Filters duplicate positionfixes before generating staypoints. Duplicates can lead to problems in later
        processing steps (e.g., when generating triplegs). It is not recommended to set this to False.

    n_jobs: int, default 1
        The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel
        computing code is used at all.

    Returns
    -------
    pfs: pd.DataFrame or gpd.GeoDataFrame
        The original positionfixes with a new column ``[`staypoint_id`]``.

    sp: gpd.GeoDataFrame
        The generated staypoints.

    References
    ----------
    Zheng, Y. (2015). Trajectory data mining: an overview. ACM Transactions on Intelligent Systems
    and Technology (TIST), 6(3), 29.

    Li, Q., Zheng, Y., Xie, X., Chen, Y., Liu, W., & Ma, W. Y. (2008, November). Mining user
    similarity based on location history. In Proceedings of the 16th ACM SIGSPATIAL international
    conference on Advances in geographic information systems (p. 34). ACM.
    """
    # Validate required columns
    required_cols = ['user_id', 'timestamp']
    for col in required_cols:
        if col not in positionfixes.columns:
            raise ValueError(f"Missing required column: {col}")

    if not isinstance(positionfixes, gpd.GeoDataFrame):
        raise TypeError("positionfixes must be a GeoDataFrame with geometry column")

    # Copy the original pfs for adding 'staypoint_id' column
    pfs = positionfixes.copy()

    if exclude_duplicate_pfs:
        len_org = pfs.shape[0]
        pfs = pfs.drop_duplicates()
        nb_dropped = len_org - pfs.shape[0]
        if nb_dropped > 0:
            warn_str = (
                f"{nb_dropped} duplicates were dropped from your positionfixes. Dropping duplicates is"
                + " recommended but can be prevented using the 'exclude_duplicate_pfs' flag."
            )
            warnings.warn(warn_str)

    # If the positionfixes already have a column "staypoint_id", we drop it
    if "staypoint_id" in pfs.columns:
        pfs.drop(columns="staypoint_id", inplace=True)

    geo_col = pfs.geometry.name
    
    # Setup traj_cols based on distance_metric
    if distance_metric == 'haversine':
        # Extract lat/lon from geometry for haversine
        pfs['_temp_lat'] = pfs[geo_col].apply(lambda g: g.y)
        pfs['_temp_lon'] = pfs[geo_col].apply(lambda g: g.x)
        traj_cols = {'latitude': '_temp_lat', 'longitude': '_temp_lon', 'timestamp': 'timestamp', 'user_id': 'user_id'}
    else:
        # Use x/y coordinates for euclidean
        pfs['_temp_x'] = pfs[geo_col].apply(lambda g: g.x)
        pfs['_temp_y'] = pfs[geo_col].apply(lambda g: g.y)
        traj_cols = {'x': '_temp_x', 'y': '_temp_y', 'timestamp': 'timestamp', 'user_id': 'user_id'}

    if method != "sliding":
        raise ValueError(f"Unknown method: {method}. Only 'sliding' is supported.")

    # Process each user and get labels
    def process_user(user_group):
        user_id, user_data = user_group
        labels = detect_stop_labels(
            data=user_data,
            delta_roam=dist_threshold,
            dt_max=gap_threshold,
            dur_min=time_threshold,
            method=method,
            traj_cols=traj_cols
        )
        return user_data.index, labels

    # Apply to all users using applyParallel
    results = applyParallel(
        pfs.groupby('user_id'),
        process_user,
        n_jobs=n_jobs,
        print_progress=print_progress
    )

    # Combine all labels into a single series
    all_labels = pd.Series(dtype=int)
    cluster_offset = 0
    for indices, labels in results:
        # Offset cluster IDs to make them globally unique across users
        adjusted_labels = labels.copy()
        mask = adjusted_labels != -1
        adjusted_labels[mask] = adjusted_labels[mask] + cluster_offset
        
        all_labels = pd.concat([all_labels, pd.Series(adjusted_labels.values, index=indices)])
        
        # Update offset for next user
        if (labels != -1).any():
            cluster_offset = adjusted_labels.max() + 1

    # Add labels to pfs
    pfs['_cluster'] = all_labels

    # Create staypoints from labeled data
    labeled_pfs = pfs[pfs['_cluster'] != -1].copy()
    
    if labeled_pfs.empty:
        warnings.warn("No staypoints can be generated, returning empty sp.")
        pfs['staypoint_id'] = pd.Series(dtype='Int64')
        sp = gpd.GeoDataFrame(columns=['user_id', 'started_at', 'finished_at', geo_col], 
                              geometry=geo_col, crs=pfs.crs)
        # Cleanup temporary columns
        pfs = pfs.drop(columns=[col for col in pfs.columns if col.startswith('_temp_') or col == '_cluster'])
        return pfs, sp

    # Aggregate by cluster to create staypoints
    sp_list = []
    cluster_ids = []
    for cluster_id, group in labeled_pfs.groupby('_cluster'):
        cluster_ids.append(cluster_id)
        # Calculate centroid using spherical mean for haversine, simple mean for euclidean
        if distance_metric == 'haversine':
            lats = group['_temp_lat'].values
            lons = group['_temp_lon'].values
            
            lats_rad = np.radians(lats)
            lons_rad = np.radians(lons)
            
            x = np.cos(lats_rad) * np.cos(lons_rad)
            y = np.cos(lats_rad) * np.sin(lons_rad)
            z = np.sin(lats_rad)
            
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            mean_z = np.mean(z)
            
            mean_lon = np.arctan2(mean_y, mean_x)
            hyp = np.sqrt(mean_x**2 + mean_y**2)
            mean_lat = np.arctan2(mean_z, hyp)
            
            center_lat = np.degrees(mean_lat)
            center_lon = np.degrees(mean_lon)
        else:
            center_lon = group['_temp_x'].mean()
            center_lat = group['_temp_y'].mean()
        
        sp_dict = {
            'user_id': group['user_id'].iloc[0],
            'started_at': group['timestamp'].min(),
            'finished_at': group['timestamp'].max(),
            geo_col: Point(center_lon, center_lat)
        }
        
        sp_list.append(sp_dict)

    sp = pd.DataFrame(sp_list)
    
    # Add staypoint IDs
    sp['staypoint_id'] = sp.index
    sp.index.name = 'id'
    
    # Map staypoint IDs back to original positionfixes (use cluster_ids to maintain correct mapping)
    cluster_to_sp_id = dict(zip(cluster_ids, sp.index))
    pfs['staypoint_id'] = pfs['_cluster'].map(cluster_to_sp_id)
    
    # Cleanup temporary columns
    pfs = pfs.drop(columns=[col for col in pfs.columns if col.startswith('_temp_') or col == '_cluster'])
    
    # Convert to GeoDataFrame
    sp_columns = ['user_id', 'started_at', 'finished_at', geo_col]
    sp = gpd.GeoDataFrame(sp, columns=sp_columns, geometry=geo_col, crs=pfs.crs)

    # dtype consistency
    sp.index = sp.index.astype("int64")
    pfs["staypoint_id"] = pfs["staypoint_id"].astype("Int64")
    sp["user_id"] = sp["user_id"].astype(pfs["user_id"].dtype)

    return pfs, sp